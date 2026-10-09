"""Shared model, preprocessing, and saved split helpers."""
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# Three SHA-256-identical pairs have conflicting labels; omit both copies.
AMBIGUOUS_IMAGES = {
    "glass/glass115.jpg", "metal/metal91.jpg",
    "glass/glass176.jpg", "plastic/plastic152.jpg",
    "glass/glass389.jpg", "plastic/plastic332.jpg",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocessing(training=False):
    weights = models.ResNet18_Weights.DEFAULT
    if not training:
        return weights.transforms()
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(pretrained=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint class order does not match this project.")
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def load_dataset(data_dir, training=False):
    dataset = datasets.ImageFolder(data_dir, transform=preprocessing(training))
    if dataset.classes != CLASSES:
        raise ValueError(f"Expected folders {CLASSES}, found {dataset.classes}")
    dataset.samples = [(path, label) for path, label in dataset.samples
                       if Path(path).relative_to(data_dir).as_posix() not in AMBIGUOUS_IMAGES]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    return dataset


def split_indices(dataset, manifest, subset):
    # Resolve saved relative paths, rather than depending on file enumeration order.
    lookup = {Path(path).relative_to(dataset.root).as_posix(): i
              for i, (path, _) in enumerate(dataset.samples)}
    saved_paths = [path for name in ("train", "val", "test") for path in manifest[name]]
    if len(saved_paths) != len(set(saved_paths)) or set(saved_paths) != set(lookup):
        raise ValueError("Dataset files differ from the saved split, or splits overlap. Restore the original dataset.")
    return [lookup[path] for path in manifest[subset]]
