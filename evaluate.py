"""Evaluate the best validation-loss checkpoint on the held-out test split."""
import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    from torch.utils.data import SequentialSampler
    from prepare_data import prepare_data
    from utils import ROOT, load_checkpoint
except ModuleNotFoundError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install requirements.txt in your Python environment.") from error


def save_sample_predictions(dataset, test_indices, truth, predicted, confidence, seed, path):
    """Show seeded examples, including a success and error when both exist.

    This is an illustrative selection, not an estimate of overall accuracy.
    All labels and scores come from the single completed test pass.
    """
    rng = np.random.default_rng(seed)
    selection = rng.choice(len(truth), min(9, len(truth)), replace=False)
    correct_mask = np.array(truth) == np.array(predicted)
    if correct_mask.any() and (~correct_mask).any():
        if correct_mask[selection].all():
            selection[-1] = rng.choice(np.flatnonzero(~correct_mask))
        elif not correct_mask[selection].any():
            selection[-1] = rng.choice(np.flatnonzero(correct_mask))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Illustrative test examples; correct and incorrect when available")
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, selection):
        image_path, _ = dataset.samples[test_indices[index]]
        with dataset.loader(image_path) as image:
            ax.imshow(image)
        correct = truth[index] == predicted[index]
        status = "CORRECT" if correct else "INCORRECT"
        ax.set_title(
            f"{status}\nTrue: {dataset.classes[truth[index]]}\n"
            f"Pred: {dataset.classes[predicted[index]]} ({confidence[index]:.0%})",
            color="green" if correct else "red", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset-resized")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/wastevision_stage3.pth")
    args = parser.parse_args()
    model_path, data_dir = args.model_path.resolve(), args.data_dir.resolve()
    split_path = ROOT / "results/splits.json"
    if not model_path.is_file():
        parser.error(f"Best checkpoint not found: {model_path}. Evaluation stopped; no training is started.")
    if not data_dir.is_dir():
        parser.error(f"Dataset folder not found: {data_dir}. Restore the original TrashNet dataset.")
    if not split_path.is_file():
        parser.error("Saved dataset split missing: results/splits.json. Restore the split used for training.")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, checkpoint = load_checkpoint(model_path, device)
        manifest = json.loads(split_path.read_text())
        if checkpoint["split"] != manifest:
            raise ValueError("The saved split differs from the split used by this checkpoint.")
        if checkpoint.get("selection_metric") != "val_loss":
            raise ValueError("Expected a checkpoint selected by validation loss.")
        if checkpoint.get("evaluation_preprocessing") != "resize_224":
            raise ValueError("Checkpoint preprocessing does not match the test loader.")
        _, dataset, loaders = prepare_data(data_dir, batch_size=32, seed=checkpoint["seed"])
        classes = dataset.classes
        if checkpoint["classes"] != classes:
            raise ValueError("Checkpoint class order differs from ImageFolder's class order.")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(f"Cannot prepare evaluation: {error}")
    loader = loaders["test"]
    test_indices = loader.dataset.indices
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        parser.error("The test loader must use sequential sampling with drop_last=False.")
    if not test_indices or len(test_indices) != len(set(test_indices)):
        parser.error("The test split is empty or contains duplicate indices.")
    results = ROOT / "results/stage4"
    try:
        results.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"Cannot create evaluation output folder: {error}")
    print(f"Checkpoint: {model_path}", flush=True)
    print(f"Selected epoch: {checkpoint['epoch']}; validation loss: {checkpoint['val_loss']:.6f}", flush=True)
    print(f"Device: {device}; test images: {len(test_indices)}", flush=True)
    print(f"Dataset class order: {classes}", flush=True)
    model.eval()
    truth, predicted, confidence = [], [], []
    # Exactly one pass over test batches; train/validation loaders are not used.
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite model scores encountered; evaluation stopped.")
            scores, choices = logits.softmax(dim=1).max(dim=1)
            truth.extend(labels.tolist())
            predicted.extend(choices.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
    expected_labels = [dataset.targets[index] for index in test_indices]
    if truth != expected_labels:
        raise RuntimeError("Test coverage/order mismatch: each saved test image must be evaluated once.")
    labels = list(range(len(classes)))
    report = classification_report(truth, predicted, labels=labels,
        target_names=classes, output_dict=True, zero_division=0)
    report_text = classification_report(truth, predicted, labels=labels,
        target_names=classes, zero_division=0, digits=4)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    records = []
    for index, actual, guess, score in zip(test_indices, truth, predicted, confidence):
        relative_path = Path(dataset.samples[index][0]).relative_to(data_dir).as_posix()
        records.append({"image": relative_path, "true_label": classes[actual],
                        "predicted_label": classes[guess], "confidence": score})
    metrics = {
        "checkpoint": model_path.name, "selected_epoch": checkpoint["epoch"],
        "selection_metric": "val_loss", "validation_loss": checkpoint["val_loss"],
        "seed": checkpoint["seed"], "classes": classes, "test_images": len(truth),
        "accuracy": accuracy_score(truth, predicted), "macro_average": report["macro avg"],
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "predictions": records,
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (results / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (results / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label / predicted_label", *classes])
        for name, row in zip(classes, matrix):
            writer.writerow([name, *row.tolist()])
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set(title="Waste classification — held-out test counts", xlabel="Predicted label", ylabel="True label")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    save_sample_predictions(dataset, test_indices, truth, predicted, confidence,
                            checkpoint["seed"], results / "sample_predictions.png")
    print(report_text, flush=True)
    print(f"Test accuracy: {metrics['accuracy']:.2%}", flush=True)
    for metric in ("precision", "recall", "f1-score"):
        print(f"Macro {metric}: {report['macro avg'][metric]:.2%}", flush=True)
    print(f"Test coverage: {len(truth)} images, exactly once", flush=True)
    for name in ("classification_report.txt", "metrics.json", "confusion_matrix.csv",
                 "confusion_matrix.png", "sample_predictions.png"):
        print(f"Saved: {results / name}", flush=True)


if __name__ == "__main__":
    main()
