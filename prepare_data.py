"""Stage 1 only: inspect TrashNet, create splits, and load image batches.

Run: python prepare_data.py
This script does not create a model or train anything.
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from utils import ROOT, get_splits, load_dataset, seed_everything, split_indices, evaluation_preprocessing


def prepare_data(data_dir, batch_size=32, seed=42):
    """Return the original dataset, cleaned dataset, and three DataLoaders."""
    seed_everything(seed)
    # ImageFolder assigns labels from alphabetically sorted class folders.
    original = datasets.ImageFolder(data_dir)
    # Reuse the project's existing exclusion of six ambiguously labeled copies.
    dataset = load_dataset(data_dir)
    dataset.transform = evaluation_preprocessing()
    # Existing training transforms: resize, random crop, flip, tensor, normalize.
    train_dataset = load_dataset(data_dir, training=True)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    manifest = get_splits(dataset, seed, results / "splits.json")
    loaders = {}
    for name in ("train", "val", "test"):
        source = train_dataset if name == "train" else dataset
        indices = split_indices(dataset, manifest, name)
        loaders[name] = DataLoader(
            Subset(source, indices),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,  # Simple and compatible with Windows.
            generator=torch.Generator().manual_seed(seed),
        )
    return original, dataset, loaders


def save_examples(dataset, path, seed):
    """Display one original RGB image per class without normalized colors."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    targets = np.array(dataset.targets)
    for label, ax in enumerate(axes.flat):
        index = int(rng.choice(np.flatnonzero(targets == label)))
        image_path, _ = dataset.samples[index]
        with dataset.loader(image_path) as image:
            ax.imshow(image)
        ax.set_title(dataset.classes[label])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset-resized")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    torch.set_num_threads(4)
    data_dir = args.data_dir.resolve()
    original, dataset, loaders = prepare_data(data_dir, args.batch_size, args.seed)
    print(f"Dataset: {data_dir}")
    print(f"Original images: {len(original)}")
    print("Classes: " + ", ".join(dataset.classes))
    original_counts, retained_counts = Counter(original.targets), Counter(dataset.targets)
    print("Class counts (original -> retained):")
    for label, name in enumerate(dataset.classes):
        print(f"  {name}: {original_counts[label]} -> {retained_counts[label]}")
    print(f"Excluded ambiguous duplicate images: {len(original) - len(dataset)}")
    print(f"Images used for splitting: {len(dataset)}")
    print(f"Random seed: {args.seed}")
    print("Requested split: 70% train / 15% validation / 15% test (rounded to whole images)")
    print("Final split counts:")
    for name, loader in loaders.items():
        print(f"  {name}: {len(loader.dataset)}")
    # Check that every retained image occurs in exactly one split.
    index_sets = [set(loader.dataset.indices) for loader in loaders.values()]
    assert sum(map(len, index_sets)) == len(set.union(*index_sets)) == len(dataset)
    print("Split overlap: none; all retained images are covered")
    print("First batch shapes (images; labels):")
    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, 224, 224)
        assert torch.isfinite(images).all()
        print(f"  {name}: {tuple(images.shape)}; {tuple(labels.shape)}")
    figure_path = ROOT / "results/dataset_examples.png"
    save_examples(dataset, figure_path, args.seed)
    print("Saved visualization: results/dataset_examples.png")
    print("Stage 1 complete. No model created or trained.")


if __name__ == "__main__":
    main()
