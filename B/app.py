import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "tire_classifier_model.h5"  
IMG_SIZE = (64, 64)
CLASS_NAMES = ["Danger", "Safe tires", "Warning"]

# Class info for results
CLASS_INFO = {
    "Danger": {
        "color": "#EF4444",
        "bg": "#FEE2E2",
        "icon": "🔴",
        "title": "Danger — Replace Immediately",
        "message": "Your tire tread is critically worn and unsafe to drive on. The tread depth is below the legal minimum. Driving on these tires poses a serious risk of hydroplaning, blowouts, and loss of control.",
        "action": "Stop driving on this tire and replace it as soon as possible. Do not drive in wet conditions.",
        "risk": 95,
    },
    "Warning": {
        "color": "#F59E0B",
        "bg": "#FEF3C7",
        "icon": "🟡",
        "title": "Warning — Replace Soon",
        "message": "Your tire tread is getting low and approaching the wear limit. The tire still works on dry roads but is losing its ability to grip in wet conditions. Replacement should be planned soon.",
        "action": "Schedule a tire replacement within the next few weeks. Avoid driving in heavy rain if possible.",
        "risk": 55,
    },
    "Safe tires": {
        "color": "#10B981",
        "bg": "#D1FAE5",
        "icon": "🟢",
        "title": "Safe — Tire is in Good Condition",
        "message": "Your tire tread depth is sufficient and provides good grip on both dry and wet roads. The tread pattern is intact and able to channel water effectively.",
        "action": "No action needed. Continue regular tire maintenance and check again in a few months.",
        "risk": 10,
    },
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Tire Condition Analysis",
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Hide streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark theme */
    .stApp {
        background: linear-gradient(180deg, #0B0B1E 0%, #131332 100%);
    }
    
    /* Top bar */
    .top-bar {
        background: linear-gradient(135deg, #6C5CE7, #A855F7);
        padding: 20px 30px;
        border-radius: 0 0 20px 20px;
        margin: -80px -80px 30px -80px;
        text-align: center;
    }
    .top-bar h1 {
        color: white;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
    }
    .top-bar p {
        color: rgba(255,255,255,0.7);
        font-size: 14px;
        margin: 5px 0 0 0;
    }
    
    /* Upload area */
    .upload-area {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(168,85,247,0.3);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s;
    }
    .upload-area:hover {
        border-color: rgba(168,85,247,0.6);
        background: rgba(255,255,255,0.06);
    }
    
    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .result-title {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-message {
        font-size: 14px;
        line-height: 1.6;
        opacity: 0.8;
    }
    
    /* Info cards */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .info-card h3 {
        color: #A855F7;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .info-card p {
        color: rgba(255,255,255,0.6);
        font-size: 12px;
    }
    
    /* Risk meter */
    .risk-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 14px;
        width: 100%;
        margin: 8px 0;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    /* Action box */
    .action-box {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin: 12px 0;
    }
    
    /* History item */
    .history-item {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    /* Stats */
    .stat-box {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stat-number {
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 4px;
    }
    .stat-label {
        font-size: 12px;
        color: rgba(255,255,255,0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C5CE7, #A855F7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(168,85,247,0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255,255,255,0.6);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C5CE7, #A855F7);
        color: white;
    }
    
    /* Divider */
    .divider {
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 24px 0;
    }
    
    /* Hide file uploader label */
    .stFileUploader label {
        color: rgba(255,255,255,0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained tire classification model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        return None

def predict_tire(model, image):
    """Run prediction on an uploaded tire image."""
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Handle grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    all_probs = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, all_probs

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#6C5CE7,#A855F7);padding:24px 32px;border-radius:0 0 20px 20px;margin:-16px -16px 24px -16px;text-align:center;">
    <h1 style="color:white;font-size:28px;font-weight:800;margin:0;">🛞 Tire Condition Analysis</h1>
    <p style="color:rgba(255,255,255,0.7);font-size:14px;margin:6px 0 0 0;">AI-powered tire safety checker — scan your tire to check its condition</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN TABS
# ============================================================
tab_scan, tab_history, tab_info = st.tabs(["📷 Scan Tire", "📋 History", "ℹ️ How It Works"])

# ============================================================
# TAB 1: SCAN
# ============================================================
with tab_scan:
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model not found. Make sure your model file is at: " + MODEL_PATH)
        st.info("The app will show a demo result for testing purposes.")
    
    st.markdown("")
    
    # Upload section
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.markdown("### 📸 Upload Tire Photo")
        st.markdown('<p style="color:rgba(255,255,255,0.5);font-size:13px;margin-bottom:16px;">Take a photo of your tire from the side showing the tread surface</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a tire image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded tire image", use_container_width=True)
            
            analyze_btn = st.button("🔍 Analyze Tire Condition", use_container_width=True)
        else:
            analyze_btn = False
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04);border:2px dashed rgba(168,85,247,0.3);border-radius:16px;padding:48px 24px;text-align:center;margin:16px 0;">
                <p style="font-size:40px;margin-bottom:12px;">📷</p>
                <p style="color:rgba(255,255,255,0.6);font-size:14px;">Upload a photo of your tire to get started</p>
                <p style="color:rgba(255,255,255,0.3);font-size:12px;margin-top:8px;">Supports JPG, PNG, WEBP</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_result:
        st.markdown("### 📊 Analysis Result")
        
        if uploaded_file is not None and analyze_btn:
            # Loading animation
            with st.spinner("Analyzing tire condition..."):
                time.sleep(1.5)  # Simulated delay for UX
                
                if model is not None:
                    predicted_class, confidence, all_probs = predict_tire(model, image)
                else:
                    # Demo mode when no model
                    predicted_class = np.random.choice(CLASS_NAMES)
                    confidence = np.random.uniform(0.7, 0.98)
                    remaining = 1 - confidence
                    other_classes = [c for c in CLASS_NAMES if c != predicted_class]
                    split = np.random.dirichlet([1, 1]) * remaining
                    all_probs = {predicted_class: confidence}
                    for i, c in enumerate(other_classes):
                        all_probs[c] = float(split[i])
            
            info = CLASS_INFO[predicted_class]
            st.session_state.scan_count += 1
            
            # Save to history
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M — %d %b %Y"),
                "result": predicted_class,
                "confidence": confidence,
                "icon": info["icon"],
            })
            
            # Result display
            st.markdown(f"""
            <div style="background:{info['bg']}10;border:2px solid {info['color']}40;border-radius:16px;padding:24px;margin-bottom:16px;">
                <div style="font-size:40px;margin-bottom:8px;">{info['icon']}</div>
                <div style="font-size:22px;font-weight:800;color:{info['color']};margin-bottom:8px;">{info['title']}</div>
                <div style="font-size:13px;color:rgba(255,255,255,0.7);line-height:1.6;">{info['message']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence
            st.markdown(f"""
            <div style="margin:16px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:rgba(255,255,255,0.6);font-size:12px;">Confidence</span>
                    <span style="color:{info['color']};font-weight:700;font-size:14px;">{confidence:.0%}</span>
                </div>
                <div style="background:rgba(255,255,255,0.1);border-radius:10px;height:12px;overflow:hidden;">
                    <div style="background:{info['color']};height:100%;border-radius:10px;width:{confidence*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk level
            risk = info["risk"]
            st.markdown(f"""
            <div style="margin:16px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:rgba(255,255,255,0.6);font-size:12px;">Risk Level</span>
                    <span style="color:{info['color']};font-weight:700;font-size:14px;">{risk}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.1);border-radius:10px;height:12px;overflow:hidden;">
                    <div style="background:linear-gradient(90deg, #10B981, #F59E0B, #EF4444);height:100%;border-radius:10px;width:{risk}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action box
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:16px 20px;border-left:4px solid {info['color']};margin:16px 0;">
                <div style="font-size:12px;font-weight:700;color:{info['color']};margin-bottom:4px;">RECOMMENDED ACTION</div>
                <div style="font-size:13px;color:rgba(255,255,255,0.7);line-height:1.5;">{info['action']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Class probabilities
            st.markdown('<div style="margin-top:16px;"><p style="color:rgba(255,255,255,0.5);font-size:12px;font-weight:600;margin-bottom:8px;">ALL PROBABILITIES</p></div>', unsafe_allow_html=True)
            for cls_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                cls_color = CLASS_INFO[cls_name]["color"]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="color:rgba(255,255,255,0.6);font-size:12px;width:80px;">{cls_name}</span>
                    <div style="flex:1;background:rgba(255,255,255,0.08);border-radius:6px;height:8px;overflow:hidden;">
                        <div style="background:{cls_color};height:100%;width:{prob*100}%;border-radius:6px;"></div>
                    </div>
                    <span style="color:rgba(255,255,255,0.5);font-size:11px;width:40px;text-align:right;">{prob:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:48px 24px;text-align:center;">
                <p style="font-size:40px;margin-bottom:12px;">📊</p>
                <p style="color:rgba(255,255,255,0.5);font-size:14px;">Upload a tire photo and click analyze to see results here</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: HISTORY
# ============================================================
with tab_history:
    st.markdown("### 📋 Scan History")
    
    if len(st.session_state.history) == 0:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:48px 24px;text-align:center;margin:20px 0;">
            <p style="font-size:40px;margin-bottom:12px;">📋</p>
            <p style="color:rgba(255,255,255,0.5);font-size:14px;">No scans yet. Go to the Scan tab to analyze a tire.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Stats row
        total = len(st.session_state.history)
        danger_count = sum(1 for h in st.session_state.history if h["result"] == "Danger")
        warning_count = sum(1 for h in st.session_state.history if h["result"] == "Warning")
        safe_count = sum(1 for h in st.session_state.history if h["result"] == "Safe tires")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#A855F7;">{total}</div>
                <div class="stat-label">Total Scans</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#EF4444;">{danger_count}</div>
                <div class="stat-label">Danger</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#F59E0B;">{warning_count}</div>
                <div class="stat-label">Warning</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number" style="color:#10B981;">{safe_count}</div>
                <div class="stat-label">Safe</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # History list (newest first)
        for item in reversed(st.session_state.history):
            info = CLASS_INFO[item["result"]]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px 20px;margin:8px 0;display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:24px;">{item['icon']}</span>
                    <div>
                        <div style="color:{info['color']};font-weight:700;font-size:14px;">{item['result']}</div>
                        <div style="color:rgba(255,255,255,0.4);font-size:11px;">{item['time']}</div>
                    </div>
                </div>
                <div style="color:rgba(255,255,255,0.5);font-size:13px;">{item['confidence']:.0%} confidence</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 3: HOW IT WORKS
# ============================================================
with tab_info:
    st.markdown("### ℹ️ How It Works")
    st.markdown("")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="info-card">
            <p style="font-size:36px;margin-bottom:8px;">📸</p>
            <h3>1. Take a Photo</h3>
            <p>Take a clear photo of your tire from the side, showing the tread surface. Make sure the image is well-lit and in focus.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="info-card">
            <p style="font-size:36px;margin-bottom:8px;">🤖</p>
            <h3>2. AI Analysis</h3>
            <p>Our deep learning model analyzes the tread pattern, depth, and wear to classify the tire condition into one of three categories.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="info-card">
            <p style="font-size:36px;margin-bottom:8px;">✅</p>
            <h3>3. Get Results</h3>
            <p>Receive an instant classification with a confidence score and recommended action to keep you safe on the road.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🚦 Classification Categories")
    st.markdown("")
    
    for cls_name, info in CLASS_INFO.items():
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-left:4px solid {info['color']};border-radius:0 12px 12px 0;padding:16px 20px;margin:10px 0;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:20px;">{info['icon']}</span>
                <span style="color:{info['color']};font-weight:700;font-size:15px;">{cls_name}</span>
            </div>
            <p style="color:rgba(255,255,255,0.6);font-size:13px;line-height:1.5;">{info['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);border-radius:12px;padding:20px;text-align:center;">
        <p style="color:rgba(255,255,255,0.5);font-size:12px;line-height:1.6;">
            <strong style="color:#A855F7;">Disclaimer:</strong> This app provides an AI-based estimation and should not replace professional tire inspection. 
            Always consult a qualified mechanic for tire safety decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center;padding:24px 0 12px 0;margin-top:40px;border-top:1px solid rgba(255,255,255,0.06);">
    <p style="color:rgba(255,255,255,0.25);font-size:11px;">Tire Condition Analysis App — Lucas van Male — Breda University 2026</p>
</div>
""", unsafe_allow_html=True)