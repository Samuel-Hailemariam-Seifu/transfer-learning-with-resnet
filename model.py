"""
Model utilities for transfer learning with ResNet.

This module demonstrates two transfer-learning stages:
1) Feature extraction: freeze pretrained backbone, train classifier head only.
2) Fine-tuning: unfreeze last residual block and continue training with lower LR.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from config import CFG


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Note:
    - Perfect reproducibility can still vary across hardware/CUDA versions.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These flags prioritize reproducibility over maximum speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Choose GPU if available, else CPU.
    """
    if CFG.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    """
    Build a pretrained ResNet and replace its final classification layer.

    Why replace the final layer?
    - Pretrained model's original head is for ImageNet (1000 classes).
    - Your custom dataset usually has different class count.
    """
    model_name = CFG.model_name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if CFG.use_pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if CFG.use_pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("Unsupported model_name. Use 'resnet18' or 'resnet50'.")

    # Retrieve input features of original classifier.
    in_features = model.fc.in_features

    # Replace classifier head with a small custom head.
    # Works for both binary (2 classes) and multi-class (N classes).
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


def freeze_backbone_for_feature_extraction(model: nn.Module) -> None:
    """
    Freeze all backbone params, keep classifier head trainable.

    Why freezing helps:
    - Early layers already encode generic visual patterns (edges, textures).
    - Freezing reduces compute and memory cost.
    - It lowers overfitting risk on small datasets by training fewer parameters.
    """
    # First freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze classifier head parameters.
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_last_block_for_finetuning(model: nn.Module) -> None:
    """
    Unfreeze the last residual block (`layer4`) and the classifier head.

    Why fine-tuning can help:
    - If your dataset differs from ImageNet (domain gap), adapting high-level
      features improves task-specific representation.
    - We unfreeze only late layers to balance adaptation and overfitting risk.
    """
    # Keep everything frozen first, then selectively unfreeze desired parts.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final residual stage.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Keep classifier trainable as well.
    for param in model.fc.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module):
    """
    Return an iterator over trainable parameters only.

    This ensures optimizer updates only parameters with requires_grad=True.
    """
    return (p for p in model.parameters() if p.requires_grad)


def create_criterion_optimizer_phase1(model: nn.Module) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Create loss and optimizer for feature-extraction phase.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=CFG.lr_phase1,
        weight_decay=CFG.weight_decay,
    )
    return criterion, optimizer


def create_criterion_optimizer_phase2(model: nn.Module) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Create loss and optimizer for fine-tuning phase with lower learning rate.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        get_trainable_params(model),
        lr=CFG.lr_phase2,  # lower LR to avoid destroying pretrained features
        weight_decay=CFG.weight_decay,
    )
    return criterion, optimizer
