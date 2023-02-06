"""
Single-image inference script for trained transfer-learning model.

Given an image path, this script:
- loads the best checkpoint,
- preprocesses the image using ImageNet-style transforms,
- runs model inference,
- prints predicted class and confidence.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from config import CFG
from data import build_dataloaders
from model import build_model, get_device, set_seed


def build_infer_transform(image_size: int) -> transforms.Compose:
    """
    Build deterministic transform pipeline for single-image inference.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def preprocess_image(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess one image to model input tensor shape: [1, C, H, W].
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to RGB to handle grayscale or RGBA images robustly.
    image = Image.open(image_path).convert("RGB")
    transform = build_infer_transform(image_size)
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor.to(device)


def load_model_for_inference(
    checkpoint_path: Path,
    class_names: List[str],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build model architecture and load trained weights for inference.
    """
    model = build_model(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_one(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_names: List[str],
) -> Tuple[str, float]:
    """
    Predict class label and confidence score for one preprocessed image.
    """
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = class_names[pred_idx.item()]
    confidence = conf.item()
    return pred_class, confidence


def main() -> None:
    """
    Main inference entrypoint.
    """
    set_seed(CFG.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Build dataloaders once to recover class names from training directory.
    # This guarantees prediction index -> class name mapping consistency.
    _, class_names, _ = build_dataloaders()

    model = load_model_for_inference(
        checkpoint_path=CFG.best_checkpoint_path,
        class_names=class_names,
        device=device,
    )

    input_tensor = preprocess_image(
        image_path=CFG.infer_image_path,
        image_size=CFG.image_size,
        device=device,
    )

    pred_class, confidence = predict_one(
        model=model,
        input_tensor=input_tensor,
        class_names=class_names,
    )

    print(f"Image: {CFG.infer_image_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
