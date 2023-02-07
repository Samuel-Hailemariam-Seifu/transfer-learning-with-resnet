"""
Evaluation script for trained transfer-learning model.

Reports:
- Test loss
- Test accuracy
- Confusion matrix
- Classification report (precision/recall/F1)

Supports binary and multi-class classification automatically.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import CFG
from data import build_dataloaders, maybe_move_to_device
from model import build_model, get_device, set_seed


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run model on a dataloader and compute loss, accuracy, and predictions.

    Returns:
        avg_loss, avg_acc, y_true, y_pred
    """
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    true_labels: List[int] = []
    pred_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            images, labels = maybe_move_to_device(batch, device)
            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy().tolist())
            pred_labels.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / total if total > 0 else 0.0
    avg_acc = running_correct / total if total > 0 else 0.0

    return avg_loss, avg_acc, np.array(true_labels), np.array(pred_labels)


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """
    Load saved checkpoint weights into model.
    """
    # PyTorch 2.6+ defaults to `weights_only=True`, which can fail for
    # older/custom checkpoint dictionaries containing non-tensor objects.
    # We explicitly set `weights_only=False` because this project loads
    # checkpoints generated locally by `train.py` (trusted source).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main() -> None:
    """
    Main evaluation entrypoint.
    """
    set_seed(CFG.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Rebuild dataloaders to get consistent class names and num_classes.
    dataloaders, class_names, num_classes = build_dataloaders()
    test_loader = dataloaders["test"]

    # Build same architecture and load best checkpoint.
    model = build_model(num_classes=num_classes).to(device)
    checkpoint = load_checkpoint_into_model(
        model=model,
        checkpoint_path=str(CFG.best_checkpoint_path),
        device=device,
    )
    print(
        f"Loaded checkpoint: {CFG.best_checkpoint_path} "
        f"(saved_epoch={checkpoint.get('epoch', 'N/A')}, "
        f"saved_val_acc={checkpoint.get('val_acc', 'N/A')})"
    )

    # Evaluate on test split.
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Confusion matrix is useful for class-wise error patterns.
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report supports binary and multi-class automatically.
    # `target_names` gives readable class labels from folder names.
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
