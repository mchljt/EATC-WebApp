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

# Enhanced CSS for better styling with landing banner
st.markdown("""
<style>
    /* Landing Banner/Hero Section */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%),
                    url('https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80');
        background-blend-mode: overlay;
        background-size: cover;
        background-position: center;
        text-align: center;
        padding: 4rem 2rem;
        color: white;
        border-radius: 15px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(102, 126, 234, 0.8);
        z-index: 1;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .cta-button {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 10px;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.6);
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
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    
    .fake-result {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);
    }
    
    .confidence-explanation {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
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
    
    .feature-highlight {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
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

# Enhanced Landing Banner/Hero Section
with st.container():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-content">
                <h1 class="hero-title">🔍 AI Deepfake Detection Tool</h1>
                <p class="hero-subtitle">Advanced Deep Learning Technology to Identify Manipulated Images</p>
                <div class="feature-highlight">
                    <p>✨ <strong>Trained on 140,000+ Images</strong> • 🎯 <strong>76.3% Accuracy</strong> • 🚀 <strong>Instant Results</strong></p>
                </div>
                <div style="margin-top: 2rem;">
                    <span class="cta-button">🚀 Try It Now – Upload Your Image Below</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About This Tool")
    st.info("""
    This tool analyzes uploaded images to detect potential deepfakes, using a deeplearning model trained on over 140,000 images.
    
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
                'bar': {'color': "#28a745" if result_label == "REAL" else "#dc3545"},
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
        
        # Inline explanation of confidence - NEW FEATURE
        confidence_percentage = confidence * 100
        st.markdown("""
        <div class="confidence-explanation">
            <h4>🤔 What does this confidence score mean?</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence_percentage >= 90:
            st.success(f"**Very High Confidence ({confidence_percentage:.1f}%):** The model is very certain about this prediction. Results above 90% are considered highly reliable.")
        elif confidence_percentage >= 80:
            st.info(f"**High Confidence ({confidence_percentage:.1f}%):** The model is confident about this prediction. Results between 80-90% are generally reliable.")
        elif confidence_percentage >= 60:
            st.warning(f"**Moderate Confidence ({confidence_percentage:.1f}%):** The model has some uncertainty. Consider additional verification for important decisions.")
        else:
            st.error(f"**Low Confidence ({confidence_percentage:.1f}%):** The model is uncertain about this prediction. This result should be treated with caution and verified through other means.")
        
        # Additional explanation
        st.caption("💡 **Tip:** Higher confidence scores indicate the model found clearer patterns to make its decision. Lower scores suggest the image has ambiguous features.")
        
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

col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.metric("Model Accuracy", "76.3%")
with col_stats2:
    st.metric("Precision Score", "79.7%")
with col_stats3:
    st.metric("Trained on", "140,000+ images")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔒 Your images will not be stored on our servers.</p>
    <p>Built with Deeplearning for EATC Assignment.</p>
</div>
""", unsafe_allow_html=True)
