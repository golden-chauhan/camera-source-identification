
from pathlib import Path
import sys
import tempfile

import numpy as np
import streamlit as st
import torch

from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "vision_deit_tiny"
)

SRC_DIR = EXPERIMENT_DIR / "src"
PRNU_SRC_DIR = EXPERIMENT_DIR / "prnu" / "src"

CHECKPOINT_PATH = (
    EXPERIMENT_DIR
    / "checkpoints"
    / "best_deit_tiny_balanced_finetune.pth"
)

FINGERPRINT_DIR = EXPERIMENT_DIR / "prnu" / "fingerprints_deploy"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PRNU_SRC_DIR))


# ============================================================
# IMPORT EXISTING PROJECT MODULES
# ============================================================

from dataset import get_transforms
from model import create_model

from matching import load_fingerprints

from fusion import (
    calculate_image_prnu_scores,
    normalize_deit,
    normalize_prnu
)

# Separate experimental PRNU-only registered-device matcher.
import individual_device_experiment as individual_prnu


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CamTrace AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEIT_WEIGHT = 0.65
PRNU_WEIGHT = 0.35

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #172554,
            #111827,
            #1e293b
        );
        border: 1px solid #334155;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 15px;
    }

    .result-card {
        padding: 22px;
        border-radius: 14px;
        background: #172033;
        border: 1px solid #334155;
        text-align: center;
    }

    .result-label {
        color: #94a3b8;
        font-size: 13px;
    }

    .result-value {
        color: #38bdf8;
        font-size: 25px;
        font-weight: bold;
        margin-top: 8px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DEIT MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading DeiT-Tiny model...")
def load_deit_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = create_model(
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, class_names


# ============================================================
# LOAD PRNU FINGERPRINTS
# ============================================================

@st.cache_resource(show_spinner="Loading PRNU fingerprints...")
def load_prnu_fingerprints():

    if not FINGERPRINT_DIR.exists():
        raise FileNotFoundError(
            f"Fingerprint folder not found: {FINGERPRINT_DIR}"
        )

    fingerprints = load_fingerprints()

    if not fingerprints:
        raise RuntimeError(
            "No PRNU fingerprints were loaded."
        )

    return fingerprints


# ============================================================
# DEIT PREDICTION
# ============================================================

def predict_deit(image, model, class_names):

    transform = get_transforms(training=False)

    image_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):
            outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs.float(),
            dim=1
        )[0]

    return probabilities.cpu().numpy()


# ============================================================
# PRNU PREDICTION
# ============================================================

def predict_prnu(
    image_path,
    fingerprints,
    class_names
):

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    scores = calculate_image_prnu_scores(
        image_path,
        fingerprints,
        class_to_index,
        len(class_names)
    )

    return scores


# ============================================================
# FUSION
# ============================================================

def fuse_predictions(
    deit_probabilities,
    prnu_scores
):

    deit_normalized = normalize_deit(
        deit_probabilities.reshape(1, -1)
    )

    prnu_normalized = normalize_prnu(
        prnu_scores.reshape(1, -1)
    )

    fused_scores = (
        DEIT_WEIGHT * deit_normalized
        + PRNU_WEIGHT * prnu_normalized
    )

    return (
        deit_normalized[0],
        prnu_normalized[0],
        fused_scores[0]
    )


# ============================================================
# TOP PREDICTIONS
# ============================================================

def get_top_predictions(scores, class_names, count=3):

    indices = np.argsort(scores)[::-1][:count]

    return [
        {
            "device": class_names[index],
            "score": float(scores[index])
        }
        for index in indices
    ]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_predictions(
    title,
    scores,
    class_names,
    percentage=False
):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    top_predictions = get_top_predictions(
        scores,
        class_names
    )

    for rank, prediction in enumerate(
        top_predictions,
        start=1
    ):

        score = prediction["score"]

        st.write(
            f"**{rank}. {prediction['device']}**"
        )

        st.progress(
            float(
                np.clip(score, 0.0, 1.0)
            )
        )

        if percentage:
            st.caption(
                f"Score: {score * 100:.2f}%"
            )
        else:
            st.caption(
                f"Normalized score: {score:.4f}"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("CamTrace AI")

    st.caption(
        "AI-Based Camera Model and Individual "
        "Source Device Identification"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Image Analysis",
            "Model Performance",
            "Individual Device PRNU",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    st.write(
        "GPU:",
        "Available" if DEVICE.type == "cuda" else "CPU"
    )

    st.write("Device:", DEVICE.type.upper())

    st.write("Supported classes: 22")

    st.divider()

    st.caption(
        "DeiT + PRNU Hybrid Identification System"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>CamTrace AI</h1>
        <p>
            Identify the camera model and source device
            using Deep Learning and Photo Response
            Non-Uniformity fingerprint analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE ANALYSIS PAGE
# ============================================================

if page == "Image Analysis":

    st.subheader("Upload an Image")

    st.write(
        "Upload a JPEG or PNG image to analyze its "
        "camera source."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            st.caption(
                f"Image dimensions: {image.width} Ã— {image.height}"
            )

        with right:

            st.markdown("### Analysis Configuration")

            st.write("Deep learning model: DeiT-Tiny")

            st.write("Fingerprint method: PRNU")

            st.write("Fusion weights: 65% DeiT / 35% PRNU")

            st.write("Device classes: 22")

            analyze = st.button(
                "Analyze Image",
                type="primary",
                use_container_width=True
            )

        if analyze:

            temporary_path = None

            try:

                with st.spinner(
                    "Loading models and analyzing image..."
                ):

                    model, class_names = load_deit_model()

                    fingerprints = load_prnu_fingerprints()

                    # Verify that the fingerprint classes match.
                    if set(fingerprints.keys()) != set(class_names):
                        raise ValueError(
                            "PRNU fingerprint classes do not match "
                            "the DeiT checkpoint classes."
                        )

                    # Save original uploaded bytes temporarily.
                    suffix = Path(uploaded_file.name).suffix.lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temporary_file:

                        temporary_file.write(
                            uploaded_file.getvalue()
                        )

                        temporary_path = Path(
                            temporary_file.name
                        )

                    # Run DeiT.
                    deit_probabilities = predict_deit(
                        image,
                        model,
                        class_names
                    )

                    # Run PRNU.
                    prnu_scores = predict_prnu(
                        temporary_path,
                        fingerprints,
                        class_names
                    )

                    # Fuse predictions.
                    (
                        deit_normalized,
                        prnu_normalized,
                        fused_scores
                    ) = fuse_predictions(
                        deit_probabilities,
                        prnu_scores
                    )

                # ------------------------------------------------
                # FINAL RESULT
                # ------------------------------------------------

                st.divider()

                st.subheader("Identification Result")

                final_index = int(
                    np.argmax(fused_scores)
                )

                final_device = class_names[final_index]

                top3 = get_top_predictions(
                    fused_scores,
                    class_names,
                    3
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">
                                IDENTIFIED DEVICE
                            </div>
                            <div class="result-value">
                                {final_device}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col2:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">
                                FUSED NORMALIZED SCORE
                            </div>
                            <div class="result-value">
                                {fused_scores[final_index]:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.info(
                    "The fused score is a ranking score, "
                    "not a calibrated probability."
                )

                # ------------------------------------------------
                # COMPONENT PREDICTIONS
                # ------------------------------------------------

                st.divider()

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.markdown("### DeiT-Tiny")

                    deit_top = get_top_predictions(
                        deit_probabilities,
                        class_names
                    )

                    for item in deit_top:

                        st.write(
                            f"**{item['device']}**"
                        )

                        st.progress(
                            float(
                                np.clip(item["score"], 0, 1)
                            )
                        )

                        st.caption(
                            f"{item['score'] * 100:.2f}% "
                            "softmax score"
                        )

                with col2:

                    st.markdown("### PRNU")

                    prnu_top = get_top_predictions(
                        prnu_scores,
                        class_names
                    )

                    for item in prnu_top:

                        st.write(
                            f"**{item['device']}**"
                        )

                        st.caption(
                            f"PCE: {item['score']:.4f}"
                        )

                with col3:

                    st.markdown("### Fused Prediction")

                    for item in top3:

                        st.write(
                            f"**{item['device']}**"
                        )

                        st.caption(
                            f"Normalized score: "
                            f"{item['score']:.4f}"
                        )

                # ------------------------------------------------
                # PREDICTION TABLE
                # ------------------------------------------------

                st.divider()

                st.subheader("Top-3 Device Predictions")

                result_rows = [
                    {
                        "Rank": index + 1,
                        "Device": item["device"],
                        "Fused Score": round(item["score"], 5)
                    }
                    for index, item in enumerate(top3)
                ]

                table_markdown = (
                    "| Rank | Device | Fused Score |\n"
                    "|---:|---|---:|\n"
                    + "\n".join(
                        f"| {row['Rank']} | {row['Device']} | {row['Fused Score']:.5f} |"
                        for row in result_rows
                    )
                )

                st.markdown(table_markdown)
                st.success(
                    "Image analysis completed successfully."
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )

                st.exception(error)

            finally:

                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.subheader("Model Evaluation Results")

    st.caption(
        "Reported evaluation metrics from the completed "
        "500-image test experiment."
    )

    st.info(
        "These are dataset evaluation results, not "
        "confidence values for the currently uploaded image."
    )

    metrics = {
        "DeiT-Tiny": {
            "Top-1 Accuracy": 80.60,
            "Top-3 Accuracy": 95.40,
            "Macro F1": 77.30,
            "Weighted F1": 80.16
        },
        "PRNU": {
            "Top-1 Accuracy": 87.40,
            "Top-3 Accuracy": 90.80,
            "Macro F1": 85.24,
            "Weighted F1": 85.79
        },
        "Fusion": {
            "Top-1 Accuracy": 96.20,
            "Top-3 Accuracy": 99.20,
            "Macro F1": 95.84,
            "Weighted F1": 96.13
        }
    }

    for model_name, values in metrics.items():

        st.markdown(f"### {model_name}")

        columns = st.columns(4)

        for column, (metric_name, value) in zip(
            columns,
            values.items()
        ):

            column.metric(
                metric_name,
                f"{value:.2f}%"
            )

        st.divider()

    st.markdown("### Experimental Individual-Device PRNU Matching")
    exp_col1, exp_col2 = st.columns(2)
    exp_col1.metric("Top-1 Accuracy", "89.98%")
    exp_col2.metric("Top-3 Accuracy", "93.79%")
    st.caption(
        "499 successfully evaluated query images across 22 known VISION device "
        "classes; one corrupted query image was excluded. This is a separate "
        "experimental PRNU-only result, not the DeiT + PRNU fusion result."
    )
    st.warning(
        "Closed-set experimental matching only. PCE scores are ranking statistics, "
        "not calibrated probabilities or forensic proof of physical-device identity."
    )
    st.divider()

    st.subheader("Fusion Configuration")

    st.write("DeiT weight: 65%")
    st.write("PRNU weight: 35%")
    st.write("Test images: 500")
    st.write("Device classes: 22")


# ============================================================
# INDIVIDUAL DEVICE PRNU PAGE (EXPERIMENTAL)
# ============================================================

elif page == "Individual Device PRNU":

    st.subheader("Registered Device Matching — Experimental")
    st.write(
        "Upload an image and compare its PRNU residual against 22 reference "
        "fingerprints generated from the separate 1,100-image reference subset. "
        "This experiment is independent of the DeiT + PRNU fusion prediction."
    )
    st.warning(
        "Closed-set experiment: the image is ranked only against 22 registered "
        "VISION device classes. PCE is a matching statistic, not a probability "
        "or forensic-grade attribution. Unknown-device rejection is not implemented."
    )

    individual_uploaded = st.file_uploader(
        "Upload a query image for individual-device PRNU matching",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="individual_device_upload"
    )

    if individual_uploaded is not None:
        try:
            individual_image = Image.open(individual_uploaded).convert("RGB")
            image_col, details_col = st.columns([1, 1])
            with image_col:
                st.image(individual_image, caption="Query image", use_container_width=True)
                st.caption(
                    f"Dimensions: {individual_image.width} × "
                    f"{individual_image.height} pixels"
                )
            with details_col:
                st.markdown("### Experiment details")
                st.write("Reference fingerprints: 22")
                st.write("Reference subset: 1,100 images")
                st.write("Evaluation subset: 499 valid query images")
                st.write("Matching: PRNU residual + PCE/NCC ranking")
                st.write("Crop: 512 × 512, without resizing")

            run_individual = st.button(
                "Run Individual Device Matching",
                type="primary",
                use_container_width=True,
                key="run_individual_prnu"
            )

            if run_individual:
                temporary_individual_path = None
                try:
                    with st.spinner("Extracting PRNU and comparing 22 fingerprints..."):
                        if not individual_prnu.FINGERPRINT_ROOT.exists():
                            raise FileNotFoundError(
                                "Experimental fingerprints folder is missing: "
                                f"{individual_prnu.FINGERPRINT_ROOT}"
                            )
                        fingerprints = individual_prnu.load_fingerprints()
                        if len(fingerprints) != 22:
                            raise RuntimeError(
                                f"Expected 22 fingerprints; loaded {len(fingerprints)}."
                            )

                        suffix = Path(individual_uploaded.name).suffix.lower() or ".jpg"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(individual_uploaded.getvalue())
                            temporary_individual_path = Path(tmp.name)

                        ranked_matches = individual_prnu.identify(
                            temporary_individual_path, fingerprints
                        )

                    if not ranked_matches:
                        st.error("No compatible registered fingerprints were found.")
                    else:
                        st.divider()
                        st.subheader("Registered Device Match Results")
                        best = ranked_matches[0]
                        best_col, score_col = st.columns(2)
                        best_col.metric("Top matching candidate", best["device"])
                        score_col.metric("Top candidate PCE", f"{best['pce']:.6f}")

                        st.markdown("### Top-3 registered candidates")
                        for rank, item in enumerate(ranked_matches[:3], start=1):
                            st.write(f"**{rank}. {item['device']}**")
                            st.caption(
                                f"PCE: {item['pce']:.6f} | NCC: {item['ncc']:.6f}"
                            )

                        st.markdown("### Full ranking")
                        ranking_table = (
                            "| Rank | Device | PCE | NCC |\n"
                            "|---:|---|---:|---:|\n"
                            + "\n".join(
                                f"| {rank} | {item['device']} | "
                                f"{item['pce']:.6f} | {item['ncc']:.6f} |"
                                for rank, item in enumerate(ranked_matches, start=1)
                            )
                        )
                        st.markdown(ranking_table)
                        st.info(
                            "The first row is the highest-scoring registered candidate. "
                            "A high PCE score alone does not establish identity."
                        )
                except Exception as error:
                    st.error(f"Individual-device matching failed: {error}")
                    st.exception(error)
                finally:
                    if temporary_individual_path is not None:
                        temporary_individual_path.unlink(missing_ok=True)
        except Exception as error:
            st.error(f"Could not read the uploaded image: {error}")


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About Project":

    st.subheader("About the Project")

    st.write(
        """
        This project identifies the camera model and source
        device responsible for producing a digital image.

        It combines two complementary techniques:

        **1. DeiT-Tiny (Data-efficient Image Transformer)**

        A transformer-based deep learning model trained
        to classify images into 22 known device classes.

        **2. PRNU (Photo Response Non-Uniformity)**

        A sensor fingerprinting technique that extracts
        sensor noise patterns using wavelet denoising and
        compares them against reference fingerprints.

        **3. Hybrid Score Fusion**

        The normalized scores from both branches are
        combined using validation-selected weights:
        65% DeiT and 35% PRNU.

        **4. Experimental Registered-Device PRNU Matching**

        A separate PRNU-only demonstration compares uploaded images
        against 22 reference fingerprints. It is a closed-set experiment
        and has not established forensic-grade physical-device attribution.
        """
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - Python
        - Streamlit
        - PyTorch
        - DeiT-Tiny
        - TIMM
        - NumPy
        - Pillow
        - PyWavelets
        - SciPy
        - Scikit-learn
        """
    )

    st.subheader("Research Limitation")

    st.warning(
        "The model identifies among the 22 device classes "
        "represented in its training dataset. It has not "
        "yet been validated for previously unseen camera "
        "devices or forensic-grade attribution."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Camera Model and Individual Source Device "
    "Identification from Digital Images"
)



