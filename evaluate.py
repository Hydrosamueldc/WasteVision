"""Evaluate the saved checkpoint on its untouched test split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageOps

from utils import ROOT, CLASSES, load_checkpoint, load_dataset, split_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset-resized")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(ROOT / "models/wastevision.pth", device)
    dataset = load_dataset(args.data_dir)
    test_indices = split_indices(dataset, checkpoint["split"], "test")
    loader = DataLoader(Subset(dataset, test_indices), batch_size=32, num_workers=0)
    truth, predicted, confidence = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = model(images.to(device)).softmax(dim=1)
            scores, choices = probabilities.max(dim=1)
            truth.extend(labels.tolist())
            predicted.extend(choices.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
    report = classification_report(truth, predicted, labels=list(range(6)),
        target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {"accuracy": accuracy_score(truth, predicted), "test_images": len(truth),
        "best_epoch": checkpoint["epoch"], "classification_report": report}
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report_text = classification_report(truth, predicted, labels=list(range(6)),
        target_names=CLASSES, zero_division=0, digits=4)
    (results / "classification_report.txt").write_text(report_text)
    print(report_text)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=list(range(6)),
        display_labels=CLASSES, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("TrashNet test set — counts")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    # A fixed random selection avoids choosing only successful examples.
    selection = np.random.default_rng(checkpoint["seed"]).choice(len(truth), min(9, len(truth)), replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, selection):
        with Image.open(dataset.samples[test_indices[index]][0]) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        ax.set_title(f"True: {CLASSES[truth[index]]}\nPred: {CLASSES[predicted[index]]} ({confidence[index]:.0%})",
            color="green" if truth[index] == predicted[index] else "red", fontsize=10)
    fig.tight_layout()
    fig.savefig(results / "sample_predictions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
