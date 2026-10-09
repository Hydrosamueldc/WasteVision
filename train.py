"""Train a six-class ResNet18 classifier on a fixed TrashNet split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from utils import ROOT, CLASSES, build_model, load_dataset, seed_everything, split_indices


def run_epoch(model, loader, criterion, device, optimizer=None):
    # Keep frozen backbone BatchNorm statistics fixed, including during training.
    model.eval()
    model.fc.train(optimizer is not None)
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset-resized")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    seed_everything(args.seed)
    torch.set_num_threads(4)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    dataset = load_dataset(args.data_dir)
    manifest_path = results / "splits.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["seed"] != args.seed:
            raise ValueError("Use the saved split seed, or move results/splits.json to start a new experiment.")
    else:
        indices = np.arange(len(dataset))
        train_idx, held_out = train_test_split(indices, test_size=0.30,
            stratify=dataset.targets, random_state=args.seed)
        val_idx, test_idx = train_test_split(held_out, test_size=0.50,
            stratify=np.array(dataset.targets)[held_out], random_state=args.seed)
        manifest = {"seed": args.seed, "classes": CLASSES}
        for name, ids in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            manifest[name] = [Path(dataset.samples[i][0]).relative_to(args.data_dir).as_posix() for i in ids]
        manifest_path.write_text(json.dumps(manifest, indent=2))
    train_dataset = load_dataset(args.data_dir, training=True)
    train_loader = DataLoader(Subset(train_dataset, split_indices(dataset, manifest, "train")),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Subset(dataset, split_indices(dataset, manifest, "val")),
        batch_size=args.batch_size, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; train/val/test: " + "/".join(str(len(manifest[k])) for k in ("train", "val", "test")), flush=True)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = {key: [] for key in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]}
    best_accuracy = -1.0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, [train_loss, val_loss, train_accuracy, val_accuracy]):
            history[key].append(value)
        print(f"Epoch {epoch + 1}/{args.epochs}: train loss={train_loss:.4f}, accuracy={train_accuracy:.3f}; val loss={val_loss:.4f}, accuracy={val_accuracy:.3f}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES,
                "epoch": epoch + 1, "val_accuracy": val_accuracy, "seed": args.seed,
                "split": manifest, "architecture": "resnet18", "frozen_backbone": True},
                ROOT / "models/wastevision.pth")
    (results / "history.json").write_text(json.dumps(history, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, args.epochs + 1)
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[f"train_{metric}"], label="Train")
        ax.plot(epochs, history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), xticks=list(epochs))
        ax.legend()
    fig.tight_layout()
    fig.savefig(results / "training_curves.png", dpi=150)
    plt.close(fig)
    print("Saved best validation checkpoint. Next: python evaluate.py", flush=True)


if __name__ == "__main__":
    main()
