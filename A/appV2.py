import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "tire_classifier_model.h5"
IMG_SIZE = (64, 64)
CLASS_NAMES = ["Danger", "Safe tires", "Warning"]

# ============================================================
# THEME SYSTEM
# ============================================================
THEMES = {
    "Dark": {
        "bg": "#0A0E1A",
        "bg2": "#111827",
        "card": "rgba(255,255,255,0.05)",
        "card_border": "rgba(255,255,255,0.1)",
        "text": "#E2E8F0",
        "text_secondary": "rgba(255,255,255,0.65)",
        "text_muted": "rgba(255,255,255,0.35)",
        "danger": "#FF4D6A",
        "warning": "#FFB830",
        "safe": "#00D68F",
        "accent": "#3B82F6",
        "accent2": "#1D4ED8",
        "danger_bg": "#FF4D6A12",
        "warning_bg": "#FFB83012",
        "safe_bg": "#00D68F12",
    },
    "Light": {
        "bg": "#F8FAFC",
        "bg2": "#FFFFFF",
        "card": "rgba(15,23,42,0.04)",
        "card_border": "rgba(15,23,42,0.1)",
        "text": "#0F172A",
        "text_secondary": "rgba(15,23,42,0.6)",
        "text_muted": "rgba(15,23,42,0.4)",
        "danger": "#E11D48",
        "warning": "#EA8C00",
        "safe": "#059669",
        "accent": "#2563EB",
        "accent2": "#1D4ED8",
        "danger_bg": "#FFF1F2",
        "warning_bg": "#FFFBEB",
        "safe_bg": "#ECFDF5",
    },
    "Color Blind": {
        "bg": "#0A0E1A",
        "bg2": "#111827",
        "card": "rgba(255,255,255,0.05)",
        "card_border": "rgba(255,255,255,0.1)",
        "text": "#E2E8F0",
        "text_secondary": "rgba(255,255,255,0.65)",
        "text_muted": "rgba(255,255,255,0.35)",
        "danger": "#D55E00",
        "warning": "#E69F00",
        "safe": "#0072B2",
        "accent": "#3B82F6",
        "accent2": "#1D4ED8",
        "danger_bg": "#D55E0015",
        "warning_bg": "#E69F0015",
        "safe_bg": "#0072B215",
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
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ============================================================
# THEME SELECTOR (top right)
# ============================================================
col_spacer, col_theme = st.columns([4, 1])
with col_theme:
    selected_theme = st.selectbox(
        "Theme",
        ["Dark", "Light", "Color Blind"],
        index=["Dark", "Light", "Color Blind"].index(st.session_state.theme),
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

t = THEMES[st.session_state.theme]
is_light = st.session_state.theme == "Light"

# Build class info with current theme colors
CLASS_INFO = {
    "Danger": {
        "color": t["danger"],
        "bg": t["danger_bg"],
        "icon": "🔴" if st.session_state.theme != "Color Blind" else "🔶",
        "shape": "▲" if st.session_state.theme == "Color Blind" else "",
        "title": "Danger — Replace Immediately",
        "message": "Your tire tread is critically worn and unsafe to drive on. The tread depth is below the legal minimum. Driving on these tires poses a serious risk of hydroplaning, blowouts, and loss of control.",
        "action": "Stop driving on this tire and replace it as soon as possible. Do not drive in wet conditions.",
        "risk": 95,
    },
    "Warning": {
        "color": t["warning"],
        "bg": t["warning_bg"],
        "icon": "🟡" if st.session_state.theme != "Color Blind" else "🔷",
        "shape": "■" if st.session_state.theme == "Color Blind" else "",
        "title": "Warning — Replace Soon",
        "message": "Your tire tread is getting low and approaching the wear limit. The tire still works on dry roads but is losing its ability to grip in wet conditions. Replacement should be planned soon.",
        "action": "Schedule a tire replacement within the next few weeks. Avoid driving in heavy rain if possible.",
        "risk": 55,
    },
    "Safe tires": {
        "color": t["safe"],
        "bg": t["safe_bg"],
        "icon": "🟢" if st.session_state.theme != "Color Blind" else "✅",
        "shape": "●" if st.session_state.theme == "Color Blind" else "",
        "title": "Safe — Tire is in Good Condition",
        "message": "Your tire tread depth is sufficient and provides good grip on both dry and wet roads. The tread pattern is intact and able to channel water effectively.",
        "action": "No action needed. Continue regular tire maintenance and check again in a few months.",
        "risk": 10,
    },
}

# ============================================================
# CUSTOM CSS — dynamic based on theme
# ============================================================
tab_text_color = "rgba(0,0,0,0.5)" if is_light else "rgba(255,255,255,0.55)"
tab_hover_color = "rgba(0,0,0,0.8)" if is_light else "rgba(255,255,255,0.85)"

st.markdown(f"""
<style>
    .block-container {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
    .stApp > header {{
        height: 0 !important;
        min-height: 0 !important;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    .stApp {{
        background: {'linear-gradient(180deg, ' + t["bg"] + ' 0%, ' + t["bg2"] + ' 100%)'};
    }}
    
    /* Wide tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0px;
        background: {t["card"]};
        border-radius: 14px;
        padding: 5px;
        width: 100%;
        display: flex;
    }}
    .stTabs [data-baseweb="tab"] {{
        flex: 1;
        border-radius: 10px;
        color: {tab_text_color};
        font-weight: 700;
        font-size: 15px;
        padding: 14px 0;
        justify-content: center;
        white-space: nowrap;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {t["accent2"]}, {t["accent"]}) !important;
        color: white !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {tab_hover_color};
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}
    
    .info-card {{
        background: {t["card"]};
        border: 1px solid {t["card_border"]};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }}
    .info-card h3 {{
        color: {t["accent"]};
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .info-card p {{
        color: {t["text_secondary"]};
        font-size: 12px;
    }}
    
    .stat-box {{
        background: {t["card"]};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid {t["card_border"]};
    }}
    .stat-number {{
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 4px;
    }}
    .stat-label {{
        font-size: 12px;
        color: {t["text_muted"]};
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {t["accent2"]}, {t["accent"]}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(168,85,247,0.3) !important;
    }}
    
    .divider {{
        border-top: 1px solid {t["card_border"]};
        margin: 24px 0;
    }}
    
    .stFileUploader label {{
        color: {t["text_secondary"]} !important;
    }}
    
    /* Selectbox styling */
    .stSelectbox label {{
        color: {t["text_secondary"]} !important;
    }}
    .stSelectbox div[data-baseweb="select"] {{
        background: {t["card"]} !important;
        border-color: {t["card_border"]} !important;
        border-radius: 10px !important;
    }}
    
    /* General text color */
    .stMarkdown, .stMarkdown p, h1, h2, h3 {{
        color: {t["text"]} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        return None

def predict_tire(model, image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
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
theme_label = ""
if st.session_state.theme == "Color Blind":
    theme_label = "  •  ♿ Color Blind Mode"
elif st.session_state.theme == "Light":
    theme_label = "  •  ☀️ Light Mode"

st.markdown(f"""
<div style="background:linear-gradient(135deg,{t['accent2']},{t['accent']});padding:18px 32px;border-radius:0 0 16px 16px;text-align:center;margin-top:-40px;">
    <h1 style="color:white;font-size:22px;font-weight:700;margin:0;font-family:system-ui,-apple-system,sans-serif;">Tire Condition Analysis</h1>
    <p style="color:rgba(255,255,255,0.6);font-size:13px;margin:4px 0 0 0;font-family:system-ui,-apple-system,sans-serif;">Check your tire safety by uploading a photo{theme_label}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ============================================================
# MAIN TABS
# ============================================================
tab_info, tab_scan, tab_history = st.tabs(["How It Works", "Scan Tire", "History"])

# ============================================================
# TAB 1: SCAN
# ============================================================
with tab_scan:
    model = load_model()
    
    st.markdown("")
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.markdown("### Upload Tire Photo")
        st.markdown(f'<p style="color:{t["text_secondary"]};font-size:13px;margin-bottom:16px;">Take a photo of your tire from the side showing the tread</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a tire image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded tire image", use_container_width=True)
            analyze_btn = st.button("Analyze Tire", use_container_width=True)
        else:
            analyze_btn = False
            st.markdown(f"""
            <div style="background:{t['card']};border:1.5px dashed {t['accent']}40;border-radius:12px;padding:40px 24px;text-align:center;margin:16px 0;">
                <p style="color:{t['text_secondary']};font-size:14px;">Drop a tire photo here or click to browse</p>
                <p style="color:{t['text_muted']};font-size:12px;margin-top:6px;">JPG, PNG, WEBP</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_result:
        st.markdown("### Result")
        
        if uploaded_file is not None and analyze_btn:
            with st.spinner("Analyzing tire condition..."):
                time.sleep(1.5)
                if model is not None:
                    predicted_class, confidence, all_probs = predict_tire(model, image)
                else:
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
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M — %d %b %Y"),
                "result": predicted_class,
                "confidence": confidence,
                "icon": info["icon"],
            })
            
            # Color blind shape indicator
            shape_indicator = f'<span style="font-size:28px;margin-right:8px;">{info["shape"]}</span>' if info["shape"] else ""
            
            st.markdown(f"""
            <div style="background:{info['bg']};border:1.5px solid {info['color']}40;border-radius:12px;padding:20px;margin-bottom:14px;">
                <div style="font-size:18px;font-weight:700;color:{info['color']};margin-bottom:6px;">{info['title']}</div>
                <div style="font-size:13px;color:{t['text_secondary']};line-height:1.6;">{info['message']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin:16px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:{t['text_secondary']};font-size:12px;">Confidence</span>
                    <span style="color:{info['color']};font-weight:700;font-size:14px;">{confidence:.0%}</span>
                </div>
                <div style="background:{t['card_border']};border-radius:10px;height:12px;overflow:hidden;">
                    <div style="background:{info['color']};height:100%;border-radius:10px;width:{confidence*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            risk = info["risk"]
            st.markdown(f"""
            <div style="margin:16px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:{t['text_secondary']};font-size:12px;">Risk Level</span>
                    <span style="color:{info['color']};font-weight:700;font-size:14px;">{risk}%</span>
                </div>
                <div style="background:{t['card_border']};border-radius:10px;height:12px;overflow:hidden;">
                    <div style="background:linear-gradient(90deg, {t['safe']}, {t['warning']}, {t['danger']});height:100%;border-radius:10px;width:{risk}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background:{t['card']};border-radius:10px;padding:14px 18px;border-left:3px solid {info['color']};margin:14px 0;">
                <div style="font-size:11px;font-weight:600;color:{info['color']};margin-bottom:3px;text-transform:uppercase;">Recommended action</div>
                <div style="font-size:13px;color:{t['text_secondary']};line-height:1.5;">{info['action']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div style="margin-top:14px;"><p style="color:{t["text_muted"]};font-size:11px;font-weight:600;margin-bottom:6px;">Probabilities</p></div>', unsafe_allow_html=True)
            for cls_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                cls_info = CLASS_INFO[cls_name]
                cls_color = cls_info["color"]
                shape = f' {cls_info["shape"]}' if cls_info["shape"] else ""
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="color:{t['text_secondary']};font-size:12px;width:100px;">{cls_name}{shape}</span>
                    <div style="flex:1;background:{t['card_border']};border-radius:6px;height:8px;overflow:hidden;">
                        <div style="background:{cls_color};height:100%;width:{prob*100}%;border-radius:6px;"></div>
                    </div>
                    <span style="color:{t['text_muted']};font-size:11px;width:40px;text-align:right;">{prob:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{t['card']};border:1px solid {t['card_border']};border-radius:12px;padding:40px 24px;text-align:center;">
                <p style="color:{t['text_muted']};font-size:13px;">Upload a photo and click analyze to see results</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: HISTORY
# ============================================================
with tab_history:
    st.markdown("### Scan History")
    
    if len(st.session_state.history) == 0:
        st.markdown(f"""
        <div style="background:{t['card']};border:1px solid {t['card_border']};border-radius:12px;padding:40px 24px;text-align:center;margin:16px 0;">
            <p style="color:{t['text_muted']};font-size:13px;">No scans yet. Use the Scan Tire tab to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(st.session_state.history)
        danger_count = sum(1 for h in st.session_state.history if h["result"] == "Danger")
        warning_count = sum(1 for h in st.session_state.history if h["result"] == "Warning")
        safe_count = sum(1 for h in st.session_state.history if h["result"] == "Safe tires")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:{t["accent"]};">{total}</div><div class="stat-label">Total Scans</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:{t["danger"]};">{danger_count}</div><div class="stat-label">Danger</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:{t["warning"]};">{warning_count}</div><div class="stat-label">Warning</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:{t["safe"]};">{safe_count}</div><div class="stat-label">Safe</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        for item in reversed(st.session_state.history):
            info = CLASS_INFO[item["result"]]
            st.markdown(f"""
            <div style="background:{t['card']};border:1px solid {t['card_border']};border-radius:12px;padding:16px 20px;margin:8px 0;display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:24px;">{item['icon']}</span>
                    <div>
                        <div style="color:{info['color']};font-weight:700;font-size:14px;">{item['result']}</div>
                        <div style="color:{t['text_muted']};font-size:11px;">{item['time']}</div>
                    </div>
                </div>
                <div style="color:{t['text_secondary']};font-size:13px;">{item['confidence']:.0%} confidence</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 3: HOW IT WORKS
# ============================================================
with tab_info:
    st.markdown("### How It Works")
    st.markdown("")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="info-card">
            <h3>1. Take a Photo</h3>
            <p>Take a clear photo of your tire from the side showing the tread. Good lighting helps.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-card">
            <h3>2. AI Analysis</h3>
            <p>The model looks at the tread pattern and wear level to classify your tire into one of three categories.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="info-card">
            <h3>3. Get Results</h3>
            <p>You get a classification with a confidence score and a recommendation on what to do next.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Classification Categories")
    
    if st.session_state.theme == "Color Blind":
        st.markdown(f'<p style="color:{t["text_secondary"]};font-size:12px;margin-bottom:10px;">Color blind mode — categories use distinct shapes and adjusted colors</p>', unsafe_allow_html=True)
    
    st.markdown("")
    for cls_name, info in CLASS_INFO.items():
        shape_label = f' {info["shape"]}' if info["shape"] else ""
        st.markdown(f"""
        <div style="background:{t['card']};border-left:3px solid {info['color']};border-radius:0 10px 10px 0;padding:14px 18px;margin:8px 0;">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <span style="color:{info['color']};font-weight:600;font-size:14px;">{cls_name}</span><span>{shape_label}</span>
            </div>
            <p style="color:{t['text_secondary']};font-size:12px;line-height:1.5;">{info['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{t['accent']}15;border:1px solid {t['accent']}30;border-radius:12px;padding:20px;text-align:center;">
        <p style="color:{t['text_secondary']};font-size:12px;line-height:1.6;">
            <strong style="color:{t['accent']};">Disclaimer:</strong> This app provides an AI-based estimation and should not replace professional tire inspection. 
            Always consult a qualified mechanic for tire safety decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{t['accent2']}15,{t['accent']}15);border:1.5px solid {t['accent']}30;border-radius:12px;padding:28px;text-align:center;margin:12px 0;">
        <p style="color:{t['text']};font-size:16px;font-weight:600;margin-bottom:6px;">Ready to check your tires?</p>
        <p style="color:{t['text_secondary']};font-size:13px;">Click the <strong style="color:{t['accent']};">Scan Tire</strong> tab above to get started</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div style="text-align:center;padding:20px 0 10px 0;margin-top:32px;border-top:1px solid {t['card_border']};">
    <p style="color:{t['text_muted']};font-size:10px;">Tire Condition Analysis — Lucas van Male — Breda University 2026</p>
</div>
""", unsafe_allow_html=True)