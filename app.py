import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Constants
CLASS_INDICES = {"FAKE": 0, "REAL": 1}
LABELS = {v: k for k, v in CLASS_INDICES.items()}
IMG_SIZE = (256, 256)

# css styles
st.markdown(
    """
<style>
/* Hero banner */
.hero {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%), 
                url('https://images.unsplash.com/photo-1518709268805-4e9042af2176?auto=format&fit=crop&w=1200&q=80');
    background-size: cover;
    background-blend-mode: overlay;
    color:#fff;
    text-align:center;
    padding:4rem 2rem;
    border-radius:15px;
    margin-bottom:3rem;
}
.hero h1{font-size:3rem;margin:0 0 1rem 0;text-shadow:2px 2px 4px rgba(0,0,0,.3);}
.hero p{font-size:1.15rem;margin:0 0 1.5rem 0;opacity:.9;}

.cta-btn{
    display:inline-block;
    padding:1rem 2rem;
    background:linear-gradient(45deg,#28a745,#20c997);
    color:#fff;font-weight:700;border-radius:50px;
    box-shadow:0 4px 15px rgba(40,167,69,.35);
    text-decoration:none;
    transition:transform .2s,box-shadow .2s;
}
.cta-btn:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(40,167,69,.5);}

/* Result cards */
.real{background:linear-gradient(135deg,#28a745 0%,#20c997 100%);}
.fake{background:linear-gradient(135deg,#dc3545 0%,#e74c3c 100%);}
.result-card{
    color:#fff;
    padding:1rem;
    border-radius:10px;
    text-align:center;
    margin:1rem 0;
    box-shadow:0 3px 12px rgba(0,0,0,.2);
}

/* Misc */
.conf-box{background:#f8f9fa;padding:1rem;border-radius:8px;margin:1rem 0;border-left:4px solid #667eea;}
.upload-card{background:#fff;padding:2rem;border-radius:15px;box-shadow:0 4px 15px rgba(0,0,0,.1);border-left:5px solid #667eea;margin-top:1rem;}
</style>
""",
    unsafe_allow_html=True,
)

# loading the model
@st.cache_resource
def load_detection_model():
    return load_model("cnn_model.h5")


try:
    MODEL = load_detection_model()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    st.error(f"Model load failed: {e}")

# hero section
st.markdown(
    """
<div class="hero">
    <h1>🔍 AI Deepfake Detection Tool</h1>
    <p>Advanced deep learning to spot manipulated images instantly.</p>
    <p>✨ <strong>190 K+ training images</strong> • 🎯 <strong>76.3 % accuracy</strong></p>
    <a class="cta-btn">🚀 Upload your image below</a>
</div>
""",
    unsafe_allow_html=True,
)

# sidebar stuffs
with st.sidebar:
    st.header("📊 About")
    st.info(
        "Detect potential deepfakes with a CNN trained on 190 K+ images.\n\n"
        "Supported formats: JPG, JPEG, PNG."
    )
    st.header("🛡 How it works")
    st.markdown(
        "1. Upload an image\n"
        "2. The model analyses visual patterns\n"
        "3. You get a prediction & confidence score"
    )
    st.header("⚠️ Disclaimer")
    st.warning("Predictions are probabilistic. Verify critical content via multiple sources.")

# ─────────────────── 7. MAIN CONTENT ───────────────────
col_upload, col_result = st.columns(2)

# upload 
with col_upload:
    st.subheader("📤 Upload Image")
    file = st.file_uploader("Choose an image", ["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.expander("File details"):
            st.write(f"**Name:** {file.name}")
            st.write(f"**Size:** {file.size/1024:.1f} KB")
            st.write(f"**Dimensions:** {img.width}×{img.height}px")

# results
with col_result:
    if file and MODEL_READY:
        st.subheader("🔬 Analysis")
        with st.spinner("Processing..."):
            time.sleep(0.8)  # UX pause
            # preprocess
            img_resized = img.resize(IMG_SIZE)
            arr = image.img_to_array(img_resized) / 255.0
            pred = MODEL.predict(np.expand_dims(arr, 0), verbose=0)[0][0]
            pred_class = int(np.round(pred))
            conf = pred if pred_class == 1 else 1 - pred

        label = LABELS[pred_class]
        card_class = "real" if label == "REAL" else "fake"
        card_text = "✅ AUTHENTIC IMAGE" if label == "REAL" else "⚠️ POTENTIAL DEEPFAKE"

        st.markdown(
            f'<div class="result-card {card_class}"><h3>{card_text}</h3></div>',
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#28a745" if label == "REAL" else "#dc3545"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gold"},
                        {"range": [80, 100], "color": "lightgreen"},
                    ],
                },
            )
        )
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Confidence explainer
        st.markdown('<div class="conf-box"><h4>🧐 Confidence explained</h4></div>', unsafe_allow_html=True)
        pct = conf * 100
        if pct >= 90:
            st.success(f"Very high confidence ({pct:.1f} %). Result is highly reliable.")
        elif pct >= 80:
            st.info(f"High confidence ({pct:.1f} %). Result is generally reliable.")
        elif pct >= 60:
            st.warning(f"Moderate confidence ({pct:.1f} %). Consider further checks.")
        else:
            st.error(f"Low confidence ({pct:.1f} %). Verify with other methods.")
        st.caption("Higher scores mean clearer features matching the predicted class.")

    elif file and not MODEL_READY:
        st.error("Model unavailable – please try again later.")
    elif not file:
        st.markdown(
            """
<div class="upload-card">
    <h4>Ready for analysis</h4>
    <p>Upload an image to begin deepfake detection.</p>
</div>
""",
            unsafe_allow_html=True,
        )

# stats 
st.markdown("---")
stats1, stats2, stats3 = st.columns(3)
stats1.metric("Model accuracy", "76.3 %")
stats2.metric("Precision", "79.7 %")
stats3.metric("Training images", "190 K+")

st.markdown("---")
st.markdown(
    """
<div style="text-align:center;color:#666;padding:1.5rem 0;">
    🔒 Images are processed in-memory and never stored.<br>
    Built for the EATC deep learning assignment.
</div>
""",
    unsafe_allow_html=True,
)