"""
app.py — Avocado Leaf Disease Classifier (PyTorch)
Production-grade Streamlit web application.

Run with:
    streamlit run app.py
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

MODEL_PATH = "avocado_model.pth"
MAX_UPLOAD_MB    = 10
ACCEPTED_TYPES   = ["jpg", "jpeg", "png"]
DISEASE_COLOR    = "#FF4B4B"
HEALTHY_COLOR    = "#21C55D"
CONFIDENCE_WARN  = 70.0          # show warning below this threshold

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PAGE CONFIG  (must be the FIRST Streamlit call)
# ──────────────────────────────────────────────

st.set_page_config(
    page_title = "Avocado Disease Detector",
    page_icon  = "🥑",
    layout     = "centered",
)


# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }

        .banner {
            background: linear-gradient(135deg, #1a5c2a 0%, #2d8a47 60%, #4caf70 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            color: white;
        }
        .banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 .25rem 0; }
        .banner p  { font-size: 1rem; opacity: 0.88; margin: 0; }

        .stFileUploader > div > div {
            border: 2px dashed #2d8a47 !important;
            border-radius: 12px !important;
            background: #f6fdf7 !important;
        }

        .result-card {
            border-radius: 14px;
            padding: 1.4rem 1.8rem;
            margin-top: 1.2rem;
            border-left: 6px solid;
        }
        .card-disease { background: #fff5f5; border-color: #FF4B4B; }
        .card-healthy { background: #f0fdf4; border-color: #21C55D; }
        .card-label   { font-size: 1.6rem; font-weight: 700; margin: 0 0 .2rem 0; }
        .card-conf    { font-size: 1rem; opacity: 0.75; font-family: 'DM Mono', monospace; }

        .low-conf-badge {
            background: #fffbeb;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: .5rem .9rem;
            font-size: .85rem;
            color: #92400e;
            margin-top: .8rem;
        }

        .info-tile {
            background: #f9fafb;
            border-radius: 10px;
            padding: .9rem 1.2rem;
            font-size: .85rem;
            color: #374151;
            border: 1px solid #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# MODEL LOADING  (cached for the app lifetime)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_classifier():
    """
    Load AvocadoClassifier once and cache it.
    Returns the classifier instance, or None if the model file is missing.
    """
    try:
        from predict import AvocadoClassifier
        return AvocadoClassifier(model_path=MODEL_PATH)
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.error("Model load error: %s", exc)
        return None


# ──────────────────────────────────────────────
# IMAGE VALIDATION
# ──────────────────────────────────────────────

def validate_upload(file) -> Optional[Image.Image]:
    """
    Validate the uploaded file object.
    Returns a PIL Image on success, None (+ st.error) on failure.
    """
    if file is None:
        return None

    size_mb = file.size / (1024 ** 2)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Maximum: {MAX_UPLOAD_MB} MB.")
        return None

    try:
        raw  = file.read()
        img  = Image.open(io.BytesIO(raw))
        img.verify()                          # detect corrupt / truncated files
        img  = Image.open(io.BytesIO(raw))   # re-open after verify()
        return img.convert("RGB")
    except Exception:
        st.error(
            "The uploaded file could not be read as a valid image. "
            "Please upload a clear JPG or PNG photograph of an avocado leaf."
        )
        return None


# ──────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <div class="banner">
          <h1>🥑 Avocado Disease Detector</h1>
          <p>Upload a leaf photograph to instantly detect signs of disease using AI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_tiles() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="info-tile">🧠 <b>Model</b><br>MobileNetV2<br>Transfer Learning</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-tile">📐 <b>Input size</b><br>224 × 224 px<br>RGB colour</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-tile">🏷️ <b>Classes</b><br>Disease<br>Healthy</div>',
            unsafe_allow_html=True,
        )


def render_result(label: str, confidence: float) -> None:
    card_cls = "card-disease" if label == "Disease" else "card-healthy"
    icon     = "🔴" if label == "Disease" else "🟢"
    color    = DISEASE_COLOR if label == "Disease" else HEALTHY_COLOR

    st.markdown(
        f"""
        <div class="result-card {card_cls}">
          <p class="card-label" style="color:{color};">{icon} {label}</p>
          <p class="card-conf">Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence < CONFIDENCE_WARN:
        st.markdown(
            f"""
            <div class="low-conf-badge">
              ⚠️ Low confidence ({confidence:.1f}%). The image may be blurry, poorly lit,
              or not a clear close-up of a leaf. Try re-taking the photo in good lighting.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Confidence meter**")
    st.progress(int(confidence))


def render_model_missing() -> None:
    st.error(
        "**Model checkpoint not found.**\n\n"
        f"Expected: `{MODEL_PATH}`\n\n"
        "Please ensure `avocado_model.pth` is uploaded to GitHub repo root."
    )
    st.stop()


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main() -> None:
    _inject_css()
    render_header()

    # ── Model check ───────────────────────────
    classifier = load_classifier()
    if classifier is None:
        render_model_missing()

    # ── Info tiles ────────────────────────────
    render_info_tiles()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload widget ─────────────────────────
    st.subheader("Upload a leaf image")
    uploaded_file = st.file_uploader(
        label            = "Drag & drop or click to browse",
        type             = ACCEPTED_TYPES,
        label_visibility = "collapsed",
    )

    if uploaded_file is None:
        st.markdown("_Supported formats: JPG, JPEG, PNG  ·  Max size: 10 MB_")
        return

    # ── Validate ──────────────────────────────
    img = validate_upload(uploaded_file)
    if img is None:
        return

    # ── Layout ────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img, caption="Uploaded leaf image", use_column_width=True)

    with col_result:
        st.subheader("Analysis result")
        with st.spinner("Analysing leaf…"):
            try:
                label, confidence = classifier.predict_image(img)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                log.error("Prediction error: %s", exc, exc_info=True)
                return

        render_result(label, confidence)

    # ── Technical details ─────────────────────
    with st.expander("🔍 Prediction details"):
        # Derive raw prob from label + confidence
        raw_prob = (confidence / 100.0) if label == "Healthy" else (1.0 - confidence / 100.0)
        st.markdown(f"- **Raw sigmoid output P(Healthy):** `{raw_prob:.4f}`")
        st.markdown(f"- **Decision threshold:** `0.50`")
        st.markdown(f"- **Predicted class:** `{label}`")
        st.markdown(f"- **Class confidence:** `{confidence:.2f}%`")
        st.caption(
            "The model outputs a sigmoid probability in [0, 1]. "
            "Values ≥ 0.5 are classified as *Healthy*; values < 0.5 as *Disease*. "
            "Confidence always reflects the score assigned to the winning class."
        )

    st.divider()
    st.caption(
        "Avocado Disease Detector  ·  Powered by PyTorch / MobileNetV2  ·  "
        "For research use only — consult an agronomist for a definitive diagnosis."
    )


if __name__ == "__main__":
    main()
