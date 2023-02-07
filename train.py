"""
Training script for two-phase transfer learning with ResNet.

Phase 1: Feature extraction
    - Freeze backbone
    - Train classifier head only

Phase 2: Fine-tuning
    - Unfreeze last residual block + classifier
    - Train with lower learning rate

Outputs:
    - Best checkpoint based on validation accuracy
    - Console logs for train/validation loss and accuracy
"""

from pathlib import Path
from typing import Dict, Tuple

import torch

from config import CFG
from data import build_dataloaders, get_num_samples, maybe_move_to_device
from model import (
    build_model,
    create_criterion_optimizer_phase1,
    create_criterion_optimizer_phase2,
    freeze_backbone_for_feature_extraction,
    get_device,
    set_seed,
    unfreeze_last_block_for_finetuning,
)


def _safe_config_for_checkpoint() -> Dict[str, object]:
    """
    Build a serialization-safe config dict for checkpoint metadata.

    Why this helper exists:
    - Dataclass fields may contain `Path` objects.
    - In newer PyTorch versions with stricter safe loading defaults,
      non-primitive objects can complicate checkpoint loading.
    - Converting `Path` to `str` keeps metadata readable and portable.
    """
    safe_cfg: Dict[str, object] = {}
    for key, value in CFG.__dict__.items():
        safe_cfg[key] = str(value) if isinstance(value, Path) else value
    return safe_cfg


def run_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    """
    Run one epoch for either training or validation.

    Returns:
        avg_loss, avg_accuracy
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    # Context manager to disable gradients during validation for speed/memory.
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch in loader:
            images, labels = maybe_move_to_device(batch, device)

            # Forward pass.
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimizer update only during training.
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update metrics.
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    avg_acc = running_correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def train_phase(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_epoch: int,
    num_epochs: int,
    best_val_acc: float,
    checkpoint_path: Path,
) -> Tuple[int, float]:
    """
    Generic training loop for one phase.

    Saves checkpoint whenever validation accuracy improves.

    Returns:
        next_epoch_index, updated_best_val_acc
    """
    epoch_idx = start_epoch
    for _ in range(num_epochs):
        epoch_idx += 1

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        print(
            f"Epoch {epoch_idx:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Save best model by validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": _safe_config_for_checkpoint(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved new best checkpoint to: {checkpoint_path} (val_acc={val_acc:.4f})")

    return epoch_idx, best_val_acc


def main() -> None:
    """
    Main training entrypoint.
    """
    # Set random seed for reproducibility.
    set_seed(CFG.seed)

    # Select runtime device (GPU if available, otherwise CPU).
    device = get_device()
    print(f"Using device: {device}")

    # Ensure checkpoint directory exists.
    CFG.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloaders and inspect dataset.
    dataloaders, class_names, num_classes = build_dataloaders()
    sample_counts = get_num_samples(dataloaders)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Split sizes: {sample_counts}")

    # Build transfer-learning model and move to device.
    model = build_model(num_classes=num_classes).to(device)

    # ---------------- Phase 1: Feature extraction ----------------
    print("\nPhase 1: Feature extraction (freeze backbone, train head)")
    freeze_backbone_for_feature_extraction(model)
    criterion1, optimizer1 = create_criterion_optimizer_phase1(model)

    best_val_acc = 0.0
    last_epoch, best_val_acc = train_phase(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion1,
        optimizer=optimizer1,
        device=device,
        start_epoch=0,
        num_epochs=CFG.epochs_phase1,
        best_val_acc=best_val_acc,
        checkpoint_path=CFG.best_checkpoint_path,
    )

    # ---------------- Phase 2: Fine-tuning ----------------
    print("\nPhase 2: Fine-tuning (unfreeze layer4 + head, lower LR)")
    unfreeze_last_block_for_finetuning(model)
    criterion2, optimizer2 = create_criterion_optimizer_phase2(model)

    _, best_val_acc = train_phase(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion2,
        optimizer=optimizer2,
        device=device,
        start_epoch=last_epoch,
        num_epochs=CFG.epochs_phase2,
        best_val_acc=best_val_acc,
        checkpoint_path=CFG.best_checkpoint_path,
    )

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint path: {CFG.best_checkpoint_path}")


if __name__ == "__main__":
    main()
