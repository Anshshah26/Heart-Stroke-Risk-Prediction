import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; max-width: 720px !important; }

    /* ── Animated hero header ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1.5rem 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(255,80,120,0.18) 0%, transparent 70%);
        pointer-events: none;
    }

    /* Heartbeat SVG line */
    .heartbeat-line {
        display: block;
        margin: 0 auto 1.2rem;
        width: 260px;
        height: 44px;
    }
    .hb-path {
        stroke: #ff507a;
        stroke-width: 2.5;
        fill: none;
        stroke-dasharray: 600;
        stroke-dashoffset: 600;
        animation: draw 1.8s ease forwards, pulse 2.4s 1.8s ease-in-out infinite;
        filter: drop-shadow(0 0 6px rgba(255,80,122,0.7));
    }
    @keyframes draw {
        to { stroke-dashoffset: 0; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.45; }
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.4rem;
        letter-spacing: -0.5px;
        animation: fadeUp 0.7s 0.2s both;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.55);
        margin: 0;
        animation: fadeUp 0.7s 0.4s both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        animation: fadeUp 0.6s 0.5s both;
    }
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #ff507a;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .section-label::before {
        content: '';
        display: inline-block;
        width: 18px;
        height: 2px;
        background: #ff507a;
        border-radius: 2px;
    }

    /* ── Streamlit widget overrides ── */
    div[data-testid="stSlider"] > label,
    div[data-testid="stSelectbox"] > label {
        color: rgba(255,255,255,0.80) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1px;
    }

    /* Slider track */
    div[data-testid="stSlider"] .rc-slider-track { background: #ff507a !important; }
    div[data-testid="stSlider"] .rc-slider-handle {
        border-color: #ff507a !important;
        background: #ff507a !important;
        box-shadow: 0 0 0 4px rgba(255,80,122,0.25) !important;
    }
    div[data-testid="stSlider"] .rc-slider-rail  { background: rgba(255,255,255,0.15) !important; }

    /* Selectbox */
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    /* ── Predict button ── */
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #ff507a 0%, #c62a6a 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        padding: 0.75rem 2rem !important;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s !important;
        box-shadow: 0 6px 24px rgba(255,80,122,0.35) !important;
        margin-top: 0.5rem;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px rgba(255,80,122,0.5) !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
    }

    /* ── Result banners ── */
    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border: none !important;
        animation: resultBounce 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
        font-weight: 500;
    }
    @keyframes resultBounce {
        from { opacity: 0; transform: scale(0.92); }
        to   { opacity: 1; transform: scale(1); }
    }

    /* Error (high-risk) — deep red glow */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
        background: rgba(220, 38, 38, 0.18) !important;
        box-shadow: 0 0 30px rgba(220,38,38,0.3);
    }
    /* Success (low-risk) — emerald glow */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
        background: rgba(16, 185, 129, 0.15) !important;
        box-shadow: 0 0 30px rgba(16,185,129,0.25);
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.4rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <svg class="heartbeat-line" viewBox="0 0 260 44" xmlns="http://www.w3.org/2000/svg">
    <path class="hb-path"
      d="M0,22 L40,22 L52,6 L62,38 L72,6 L82,38 L92,22 L130,22
         L142,6 L152,38 L162,6 L172,38 L182,22 L260,22"/>
  </svg>
  <p class="hero-title">🫀 Heart Disease Risk Predictor</p>
  <p class="hero-subtitle">Fill in your vitals below — our model analyses your cardiovascular risk in seconds.</p>
</div>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load('heart_disease_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

model, scaler, expected_columns = load_artifacts()


# ── Section 1 — Personal info ─────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Personal information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=0, max_value=120, value=30)
with col2:
    sex = st.selectbox("Sex", options=["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)
st.markdown('</div>', unsafe_allow_html=True)


# ── Section 2 — Vitals ────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Vitals &amp; measurements</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    max_heart_rate         = st.slider("Max Heart Rate Achieved", 60, 220, 150)
with col4:
    serum_cholesterol  = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    st_depression      = st.slider("ST Depression (exercise vs rest)", 0.0, 10.0, 1.0, step=0.1)

fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
st.markdown('</div>', unsafe_allow_html=True)


# ── Section 3 — Clinical tests ───────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Clinical test results</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    rest_ecg = st.selectbox(
        "Resting ECG Results",
        options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    slope = st.selectbox(
        "Peak Exercise ST Slope",
        options=["Upsloping", "Flat", "Downsloping"]
    )
with col6:
    exercise_induced_angina = st.selectbox("Exercise-Induced Angina", options=["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)


# ── Predict button ────────────────────────────────────────────────────────────
if st.button("⚡  Analyse Risk"):
    with st.spinner("Running model…"):
        # Prepare input data with correct one-hot encoding
        raw_input = {
            "Age":                age,
            "RestingBP":          resting_blood_pressure,
            "Cholesterol":        serum_cholesterol,
            "FastingBS":          1 if fasting_blood_sugar == "Yes" else 0,
            "MaxHR":              max_heart_rate,
            "Oldpeak":            st_depression,
            "Sex_M":              1 if sex == "Male" else 0,
            "ChestPainType_ATA":  1 if chest_pain == "Atypical Angina" else 0,
            "ChestPainType_NAP":  1 if chest_pain == "Non-anginal Pain" else 0,
            "ChestPainType_TA":   1 if chest_pain == "Typical Angina" else 0,
            "RestingECG_Normal":  1 if rest_ecg == "Normal" else 0,
            "RestingECG_ST":      1 if rest_ecg == "ST-T Wave Abnormality" else 0,
            "ExerciseAngina_Y":   1 if exercise_induced_angina == "Yes" else 0,
            "ST_Slope_Flat":      1 if slope == "Flat" else 0,
            "ST_Slope_Up":        1 if slope == "Upsloping" else 0,
        }
        input_df = pd.DataFrame([raw_input])
        # Ensure all expected columns are present, filling missing ones with 0
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error(
            "⚠️ **High risk of heart disease detected.** "
            "Please consult a cardiologist or healthcare professional at your earliest convenience."
        )
    else:
        st.success(
            "✅ **Low risk of heart disease.** "
            "Great news — keep maintaining your healthy lifestyle and regular check-ups!"
        )