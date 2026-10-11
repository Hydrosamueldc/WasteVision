"""Train the ResNet18 classification head and validate after each epoch."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from prepare_data import prepare_data
from utils import ROOT, build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Update the head during training; measure validation without gradients."""
    training = optimizer is not None
    if training:
        model.train()
        # Frozen parameters do not freeze BatchNorm's running statistics.
        # Keep every backbone module in eval mode; only fc stays in train mode.
        for name, module in model.named_children():
            if name != "fc":
                module.eval()
    else:
        model.eval()
    loss_sum = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; training stopped.")
            if training:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    if total == 0:
        raise ValueError("The DataLoader contains no samples.")
    return loss_sum / total, correct / total


def plot_history(history, path):
    """Save the measured training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric.capitalize(), xticks=epochs)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset-resized")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/wastevision_stage3.pth")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10 or args.batch_size < 1:
        parser.error("epochs must be between 1 and 10; batch-size must be positive")
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"Dataset folder not found: {data_dir}. Restore TrashNet; no dataset is downloaded.")
    manifest_path = ROOT / "results/splits.json"
    if not manifest_path.is_file():
        parser.error("Dataset split not found: results/splits.json. Run prepare_data.py first.")
    try:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        if args.model_path.exists() and not args.model_path.is_file():
            parser.error(f"Model output path is not a file: {args.model_path}")
    except OSError as error:
        parser.error(f"Cannot create model output directory: {error}")
    torch.set_num_threads(4)
    try:
        _, dataset, loaders = prepare_data(data_dir, args.batch_size, args.seed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    # prepare_data sets the seeds and checks class order and split assignments.
    manifest = json.loads(manifest_path.read_text())
    train_loader, val_loader = loaders["train"], loaders["val"]
    # Use only training and validation data to select the checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seed: {args.seed}", flush=True)
    print(f"Train: {len(train_loader.dataset)}; validation: {len(val_loader.dataset)}; test: unused", flush=True)
    print("Optimizer: Adam; learning rate: 0.001; loss: CrossEntropyLoss", flush=True)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_accuracy = 0.0
    history = []
    history_path = ROOT / "results/history.csv"
    plot_path = ROOT / "results/stage3_training_curves.png"
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
            row = dict(zip(fields, [epoch, train_loss, train_accuracy, val_loss, val_accuracy]))
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch:02d}/{args.epochs}: train loss={train_loss:.4f}, accuracy={train_accuracy:.2%}; val loss={val_loss:.4f}, accuracy={val_accuracy:.2%}", flush=True)
            if val_loss < best_loss:
                best_loss, best_accuracy, best_epoch = val_loss, val_accuracy, epoch
                torch.save({
                    "model_state_dict": model.state_dict(), "classes": dataset.classes,
                    "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy,
                    "seed": args.seed, "split": manifest, "architecture": "resnet18",
                    "frozen_backbone": True, "selection_metric": "val_loss",
                    "evaluation_preprocessing": "resize_224",
                }, args.model_path)
    plot_history(history, plot_path)
    print(f"Epochs completed: {len(history)}", flush=True)
    print(f"Best checkpoint: epoch {best_epoch}; validation loss={best_loss:.6f}; validation accuracy={best_accuracy:.2%}", flush=True)
    print(f"Best model: {args.model_path.resolve()}", flush=True)
    print(f"Training history: {history_path}", flush=True)
    print(f"Training curves: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
