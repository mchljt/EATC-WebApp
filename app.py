import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .detection-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .real-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .fake-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 5px;
        margin: 10px 0;
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_detection_model():
    return load_model("cnn_model.h5")

# Initialize model
try:
    model = load_detection_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {str(e)}")

# Configuration
class_indices = {'FAKE': 0, 'REAL': 1}
labels = {v: k for k, v in class_indices.items()}
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 EATC AI-Powered Deepfake Detection Tool</h1>
    <p>Advanced Deep learning technology to identify manipulated images</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About This Tool")
    st.info("""
    This tool analyzes uploaded images to detect potential deepfakes, using a deeplearning model trained on over 190,000 images.
    
    **Supported formats:** JPG, JPEG, PNG
    """)
    
    st.markdown("### 🛡️ How It Works")
    st.write("""
    1. Upload your image
    2. Our AI model processes the image
    3. Get instant detection results
    4. View confidence metrics
    """)
    
    st.markdown("### ⚠️ Disclaimer")
    st.warning("This tool provides prediction based on AI analysis. Always verify results through multiple sources.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)
        
        # File info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "Image dimensions": f"{img.size[0]} x {img.size[1]} pixels"
        }
        
        with st.expander("📋 File Information"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

with col2:
    if uploaded_file is not None and model_loaded:
        st.markdown("### 🔬 Analysis Results")
        
        # Processing indicator
        with st.spinner('🧠 Analyzing your image...'):
            time.sleep(1)  # Simulate processing time for UX
            
            # Preprocess image
            img_processed = img.convert('RGB')
            img_processed = img_processed.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = image.img_to_array(img_processed)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)[0][0]
            predicted_class = int(np.round(prediction))
            confidence = prediction if predicted_class == 1 else 1 - prediction
            
        # Results display
        result_label = labels[predicted_class]
        
        if result_label == "REAL":
            st.markdown(f"""
            <div class="real-result">
                <h3>✅ AUTHENTIC IMAGE</h3>
                <p>This image appears to be genuine</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fake-result">
                <h3>⚠️ POTENTIAL DEEPFAKE</h3>
                <p>This image may be artificially generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence visualization
        st.markdown("#### 📊 Confidence Analysis")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Level (%)"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea" if result_label == "REAL" else "#f5576c"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical details in expandable section
        with st.expander("🔧 Technical Details"):
            col_tech1, col_tech2 = st.columns(2)
            with col_tech1:
                st.metric("Raw Prediction Score", f"{prediction:.4f}")
                st.metric("Classification Threshold", "0.5")
            with col_tech2:
                st.metric("Model Confidence", f"{confidence*100:.1f}%")
                st.metric("Predicted Class Index", predicted_class)
    
    elif uploaded_file is not None and not model_loaded:
        st.error("❌ Model not available. Please check the model file.")
    else:
        st.markdown("""
        <div class="detection-card">
            <h4>🎯 Ready for Analysis</h4>
            <p>Upload an image to begin deepfake detection analysis.</p>
            <ul>
                <li>Supports JPG, JPEG, and PNG formats</li>
                <li>Optimal image size: 256x256 pixels</li>
                <li>Clear, well-lit images work best</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Additional features section
st.markdown("---")
st.markdown("### 📈 Model Performance Statistics")

col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.metric("Model Accuracy", "76.3%")
with col_stats2:
    st.metric("Precision Score", "79.7%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔒 Your images will not be stored on our servers.</p>
    <p>Built with Deeplearning for EATC Assignment.</p>
</div>
""", unsafe_allow_html=True)