"""
Central configuration for the transfer learning project.

This file keeps all "knobs" in one place so you can:
- quickly change dataset paths,
- switch between ResNet18 and ResNet50,
- control training hyperparameters,
- and keep scripts (`train.py`, `evaluate.py`, `infer.py`) clean.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """
    Project configuration container.

    Using a dataclass keeps configuration explicit and type-friendly.
    """

    # ---------------------------------------------------------------------
    # Dataset paths:
    # Expected structure:
    # data_root/
    #   train/
    #     class_a/
    #     class_b/
    #   val/
    #     class_a/
    #     class_b/
    #   test/
    #     class_a/
    #     class_b/
    # ---------------------------------------------------------------------
    data_root: Path = Path("./dataset")
    train_dir: Path = data_root / "train"
    val_dir: Path = data_root / "val"
    test_dir: Path = data_root / "test"

    # ---------------------------------------------------------------------
    # Model setup:
    # Choose `resnet18` (faster, lighter) or `resnet50` (larger, often better).
    # ---------------------------------------------------------------------
    model_name: str = "resnet18"  # "resnet18" or "resnet50"
    use_pretrained: bool = True

    # ---------------------------------------------------------------------
    # Input shape:
    # ResNet models are commonly trained on 224x224 images.
    # ---------------------------------------------------------------------
    image_size: int = 224

    # ---------------------------------------------------------------------
    # Data loading:
    # Increase `num_workers` based on your CPU for faster loading.
    # ---------------------------------------------------------------------
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    # ---------------------------------------------------------------------
    # Optimization:
    # Phase 1 trains classifier head only.
    # Phase 2 fine-tunes the last residual block + head with lower LR.
    # ---------------------------------------------------------------------
    epochs_phase1: int = 5
    epochs_phase2: int = 5
    lr_phase1: float = 1e-3
    lr_phase2: float = 1e-4
    weight_decay: float = 1e-4

    # ---------------------------------------------------------------------
    # Reproducibility:
    # Setting a seed makes runs more repeatable.
    # ---------------------------------------------------------------------
    seed: int = 42

    # ---------------------------------------------------------------------
    # Runtime:
    # `device` auto-selects GPU if available, otherwise CPU.
    # ---------------------------------------------------------------------
    device: str = "cuda"  # script will fallback to CPU if CUDA unavailable

    # ---------------------------------------------------------------------
    # Checkpointing:
    # We save the best model (highest validation accuracy).
    # ---------------------------------------------------------------------
    checkpoint_dir: Path = Path("./checkpoints")
    best_checkpoint_name: str = "best_model.pth"

    # ---------------------------------------------------------------------
    # Inference settings:
    # File path to test one image in `infer.py`.
    # ---------------------------------------------------------------------
    infer_image_path: Path = Path("./sample.jpg")

    @property
    def best_checkpoint_path(self) -> Path:
        """
        Convenience property to build full checkpoint path in one place.
        """
        return self.checkpoint_dir / self.best_checkpoint_name


# Singleton-style config object for easy import in scripts.
CFG = Config()
