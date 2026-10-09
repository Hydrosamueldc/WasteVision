"""Shared model, preprocessing, and saved split helpers."""
import json
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import numpy as np
import torch
from sklearn.model_selection import train_test_split
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


def evaluation_preprocessing():
    """The deterministic Stage 1 validation/test transform, also for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(pretrained=False):
    """Return frozen ResNet18 with a new head in CLASSES order.

    Use pretrained=True to initialize for transfer learning. False is used
    when rebuilding the architecture before loading a saved checkpoint.
    The forward pass returns raw class scores (logits), not probabilities.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # Freeze the existing ImageNet parameters before replacing the head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    # A new Linear layer is trainable by default: 512 features -> six scores.
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


def get_splits(dataset, seed, manifest_path):
    """Reuse saved stratified splits, or create a 70/15/15 split once."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["seed"] != seed:
            raise ValueError("Use the saved split seed, or move results/splits.json to start a new experiment.")
    else:
        indices = np.arange(len(dataset))
        train_idx, held_out = train_test_split(indices, test_size=0.30,
            stratify=dataset.targets, random_state=seed)
        val_idx, test_idx = train_test_split(held_out, test_size=0.50,
            stratify=np.array(dataset.targets)[held_out], random_state=seed)
        manifest = {"seed": seed, "classes": CLASSES}
        for name, ids in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            manifest[name] = [Path(dataset.samples[i][0]).relative_to(dataset.root).as_posix() for i in ids]
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if manifest["classes"] != dataset.classes:
        raise ValueError("Saved split classes do not match the dataset.")
    for subset in ("train", "val", "test"):
        split_indices(dataset, manifest, subset)
    return manifest


def predict_image(image, model_path=ROOT / "models/wastevision_stage3.pth", device=None,
                  checkpoint_loader=None):
    """Predict one image path or PIL image using the saved Stage 3 checkpoint.

    Returns a dictionary with predicted_class, score, and top_three.
    Scores are softmax outputs, not calibrated probabilities of correctness.
    The dataset is not required: class names come from the checkpoint.
    An optional checkpoint_loader lets an interface cache the model resource.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Best model checkpoint not found: {model_path}. Restore the trained checkpoint.")
    try:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            with Image.open(image_path) as source:
                rgb_image = source.convert("RGB")
        elif isinstance(image, Image.Image):
            # Match ImageFolder's RGB loading, including grayscale/RGBA inputs.
            rgb_image = image.convert("RGB")
        else:
            raise TypeError("image must be a file path or an already loaded PIL image.")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        if isinstance(error, FileNotFoundError):
            raise
        raise ValueError(f"Cannot read the input as an image: {error}") from error
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    try:
        loader = checkpoint_loader if checkpoint_loader is not None else load_checkpoint
        model, checkpoint = loader(model_path, device)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise ValueError(f"Cannot load checkpoint {model_path}: {error}") from error
    if checkpoint.get("evaluation_preprocessing") != "resize_224":
        raise ValueError("Expected the Stage 3 checkpoint with resize_224 preprocessing.")
    classes = checkpoint["classes"]
    if classes != checkpoint["split"]["classes"]:
        raise ValueError("Checkpoint class order differs from the saved Stage 1 class order.")
    model.eval()
    batch = evaluation_preprocessing()(rgb_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise RuntimeError("The model returned non-finite scores.")
        probabilities = logits.softmax(dim=1)[0]
        scores, indices = probabilities.topk(min(3, len(classes)))
    top_three = [{"class": classes[index], "score": score}
                 for index, score in zip(indices.cpu().tolist(), scores.cpu().tolist())]
    return {"predicted_class": top_three[0]["class"], "score": top_three[0]["score"],
            "top_three": top_three}


def check_model():
    """Stage 2 only: initialize pretrained ResNet18 and verify its output."""
    seed_everything(42)
    torch.set_num_threads(4)
    dataset = load_dataset(ROOT / "data/dataset-resized")
    # load_dataset already checks that ImageFolder's class order equals CLASSES.
    saved_classes = json.loads((ROOT / "results/splits.json").read_text())["classes"]
    if saved_classes != dataset.classes:
        raise ValueError("Stage 1's saved class order differs from the dataset.")
    model = build_model(pretrained=True)
    # Evaluation mode also keeps BatchNorm statistics fixed for this check.
    model.eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    assert trainable_names == ["fc.weight", "fc.bias"]
    print("Architecture: ResNet18")
    print("Pretrained weights: ResNet18_Weights.IMAGENET1K_V1 (DEFAULT)")
    print(f"Classes: {dataset.classes}")
    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Final layer: {model.fc}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    print("Trainable tensors: " + ", ".join(trainable_names))
    dummy_image = torch.zeros(1, 3, 224, 224)
    # No gradients or parameter updates are needed for a shape check.
    with torch.inference_mode():
        logits = model(dummy_image)
    assert logits.shape == (1, len(dataset.classes))
    assert torch.isfinite(logits).all()
    print(f"Dummy input shape: {list(dummy_image.shape)}")
    print(f"Output shape: {list(logits.shape)}")
    print("Output contains finite logits: True")
    print("Stage 2 complete. No training, optimizer, or loss function used.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check the Stage 2 model or predict one image.")
    parser.add_argument("--image", type=Path, help="Run Stage 5 inference on this image path.")
    args = parser.parse_args()
    if args.image is None:
        check_model()
    else:
        torch.set_num_threads(4)
        try:
            result = predict_image(args.image)
        except (FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
            parser.error(str(error))
        print(f"Image: {args.image.resolve()}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Model score: {result['score']:.4%}")
        print("Top three predictions:")
        for item in result["top_three"]:
            print(f"  {item['class']}: {item['score']:.4%}")
