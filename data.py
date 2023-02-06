"""
Data loading utilities for transfer learning with torchvision datasets.

This module provides:
- train/val/test transforms,
- DataLoaders for all splits,
- class name extraction for reporting and inference.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CFG


def get_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    """
    Build data transforms for train, validation, and test sets.

    Why different transforms?
    - Train transform includes augmentation (random crop/flip) to improve
      generalization and reduce overfitting.
    - Validation/Test transforms are deterministic to measure true performance.
    """
    # Normalization values are ImageNet statistics because pretrained ResNet
    # expects input distributions similar to ImageNet-preprocessed images.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose(
        [
            # Random resized crop creates scale/position variation.
            transforms.RandomResizedCrop(image_size),
            # Random horizontal flip adds left-right invariance.
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            # Resize + center crop is standard for deterministic ResNet eval.
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


def _build_imagefolder(path: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    """
    Create an ImageFolder dataset with a safety check for path existence.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. "
            "Please update paths in config.py."
        )
    return datasets.ImageFolder(root=str(path), transform=transform)


def build_dataloaders() -> Tuple[Dict[str, DataLoader], List[str], int]:
    """
    Create train/val/test dataloaders.

    Returns:
        dataloaders: dict with `train`, `val`, `test` DataLoader objects
        class_names: sorted class labels from folder names
        num_classes: number of classes
    """
    tfms = get_transforms(CFG.image_size)

    train_ds = _build_imagefolder(CFG.train_dir, tfms["train"])
    val_ds = _build_imagefolder(CFG.val_dir, tfms["val"])
    test_ds = _build_imagefolder(CFG.test_dir, tfms["test"])

    # Ensure class mapping is consistent across all splits.
    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError(
            "Class mapping mismatch across train/val/test directories. "
            "Ensure each split has the same class folder names."
        )

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=CFG.batch_size,
            shuffle=True,  # shuffle for SGD stability and better generalization
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=CFG.pin_memory,
        ),
    }

    class_names = train_ds.classes
    num_classes = len(class_names)
    return dataloaders, class_names, num_classes


def get_num_samples(dataloaders: Dict[str, DataLoader]) -> Dict[str, int]:
    """
    Utility helper for logging dataset sizes per split.
    """
    return {split: len(loader.dataset) for split, loader in dataloaders.items()}


def maybe_move_to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device):
    """
    Move a `(images, labels)` batch to target device.

    Keeping this helper makes training/evaluation loops cleaner.
    """
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
