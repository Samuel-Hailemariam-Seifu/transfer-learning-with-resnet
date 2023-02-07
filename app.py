"""
Streamlit UI for the transfer-learning ResNet project.

This app gives a simple interface to:
- understand transfer learning concepts,
- inspect dataset splits/classes,
- trigger training/evaluation scripts,
- run single-image inference from uploaded files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import torch
from PIL import Image

from config import CFG
from data import build_dataloaders
from infer import predict_one, preprocess_image
from model import build_model, get_device


def run_python_script(script_name: str) -> Tuple[int, str, str]:
    """
    Run one of the project scripts (`train.py` or `evaluate.py`) as subprocess.

    Returning stdout/stderr lets the UI show full script logs to the user.
    """
    command = [sys.executable, script_name]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


@st.cache_resource(show_spinner=False)
def load_inference_model(class_names: List[str], device: torch.device) -> torch.nn.Module:
    """
    Load checkpointed model once and cache it for faster repeated predictions.
    """
    checkpoint_path = CFG.best_checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at `{checkpoint_path}`. Train the model first."
        )

    model = build_model(num_classes=len(class_names)).to(device)

    # `weights_only=False` keeps compatibility with local project checkpoints.
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_class_names() -> List[str]:
    """
    Resolve class names from dataset folders using existing data pipeline code.
    """
    _, class_names, _ = build_dataloaders()
    return class_names


def show_overview() -> None:
    """
    Render educational overview for transfer learning workflow.
    """
    st.subheader("What is transfer learning?")
    st.write(
        "Transfer learning means we start from a model already trained on a large "
        "dataset (ImageNet), then adapt it to our own task. This is faster and "
        "usually works better than training from scratch on small datasets."
    )

    st.subheader("Why freezing helps first")
    st.write(
        "In phase 1, we freeze the pretrained convolutional backbone and train only "
        "the final classifier head. This reduces overfitting and training cost because "
        "we update far fewer parameters."
    )

    st.subheader("When fine-tuning helps")
    st.write(
        "In phase 2, we unfreeze the last residual block (`layer4`) and train with a "
        "lower learning rate. This is useful when your dataset has some domain gap from "
        "ImageNet and benefits from adapting higher-level visual features."
    )


def show_dataset_info() -> None:
    """
    Show split sizes and class names so users can verify dataset setup quickly.
    """
    st.subheader("Dataset Information")
    try:
        dataloaders, class_names, _ = build_dataloaders()
        split_counts = {split: len(loader.dataset) for split, loader in dataloaders.items()}

        st.write(f"**Classes:** {class_names}")
        st.write("**Split Sizes:**")
        st.json(split_counts)
    except Exception as exc:
        st.error(
            "Could not load dataset. Check folder structure in `config.py` "
            f"and dataset paths.\n\nDetails: {exc}"
        )


def show_training_controls() -> None:
    """
    Render button to run `train.py` and display logs.
    """
    st.subheader("Train Model")
    st.caption("Runs two-phase training: feature extraction then fine-tuning.")

    if st.button("Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            code, stdout, stderr = run_python_script("train.py")

        if code == 0:
            st.success("Training finished successfully.")
        else:
            st.error(f"Training failed with exit code {code}.")

        st.text_area("Training Output", value=stdout or "(no stdout)", height=260)
        if stderr:
            st.text_area("Training Errors", value=stderr, height=180)


def show_evaluation_controls() -> None:
    """
    Render button to run `evaluate.py` and display metrics output.
    """
    st.subheader("Evaluate Model")
    st.caption("Computes test loss/accuracy, confusion matrix, and classification report.")

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating on test set..."):
            code, stdout, stderr = run_python_script("evaluate.py")

        if code == 0:
            st.success("Evaluation completed.")
        else:
            st.error(f"Evaluation failed with exit code {code}.")

        st.text_area("Evaluation Output", value=stdout or "(no stdout)", height=260)
        if stderr:
            st.text_area("Evaluation Errors", value=stderr, height=180)


def show_inference_ui() -> None:
    """
    Render uploaded-image inference UI and prediction result.
    """
    st.subheader("Single Image Inference")
    st.caption("Upload a JPG/PNG image and get class prediction + confidence.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    # Show preview so user confirms the selected image.
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", width=280)

    try:
        device = get_device()
        class_names = resolve_class_names()
        model = load_inference_model(class_names=class_names, device=device)

        # Save to temporary local file because existing preprocessing expects path.
        temp_dir = Path("./.tmp_ui")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        pil_image.save(temp_path)

        input_tensor = preprocess_image(
            image_path=temp_path,
            image_size=CFG.image_size,
            device=device,
        )
        pred_class, confidence = predict_one(model, input_tensor, class_names)

        st.success(f"Prediction: **{pred_class}**")
        st.write(f"Confidence: **{confidence:.4f}**")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")


def main() -> None:
    """
    Streamlit app entrypoint.
    """
    st.set_page_config(
        page_title="Transfer Learning with ResNet",
        page_icon="🧠",
        layout="wide",
    )

    st.title("Transfer Learning with ResNet - Educational UI")
    st.write(
        "This interface wraps the project scripts and helps you train, evaluate, and "
        "predict with your custom image classification dataset."
    )

    tab_overview, tab_data, tab_train, tab_eval, tab_infer = st.tabs(
        ["Overview", "Dataset", "Train", "Evaluate", "Infer"]
    )

    with tab_overview:
        show_overview()

    with tab_data:
        show_dataset_info()

    with tab_train:
        show_training_controls()

    with tab_eval:
        show_evaluation_controls()

    with tab_infer:
        show_inference_ui()


if __name__ == "__main__":
    main()
