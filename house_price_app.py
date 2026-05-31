import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; margin-bottom: 1.5rem; }
.hero h1 {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 1rem; }
.card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #a78bfa; margin-bottom: 1rem;
}
.stNumberInput label, .stSelectbox label {
    color: #cbd5e1 !important; font-size: 0.875rem !important; font-weight: 500 !important;
}
.stNumberInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important; color: white !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important; font-size: 1.1rem !important;
    font-weight: 600 !important; padding: 0.8rem 2rem !important;
    border-radius: 12px !important; border: none !important;
    letter-spacing: 0.03em; margin-top: 0.5rem;
}
.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}
.result-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(37,99,235,0.2));
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.result-label { color: #94a3b8; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.5rem; }
.result-price {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.result-note { color: #64748b; font-size: 0.8rem; margin-top: 0.5rem; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("linear_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

TRAIN_COLUMNS = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_yes', 'guestroom_yes', 'basement_yes',
    'hotwaterheating_yes', 'airconditioning_yes', 'prefarea_yes',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

st.markdown("""
<div class="hero">
    <h1>🏠 House Price Predictor</h1>
    <p>Fill in the property details below to get an instant price estimate</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ `linear_model.pkl` not found. Place it in the same folder as this app.")
    st.stop()

st.markdown('<div class="card"><div class="card-title">📐 Property Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    area      = st.number_input("Area (sq ft)",  min_value=500,  max_value=20000, value=5000, step=100)
    bedrooms  = st.number_input("Bedrooms",       min_value=1,    max_value=10,    value=3)
    bathrooms = st.number_input("Bathrooms",      min_value=1,    max_value=10,    value=2)
with col2:
    stories   = st.number_input("Stories",        min_value=1,    max_value=10,    value=2)
    parking   = st.number_input("Parking Spots",  min_value=0,    max_value=10,    value=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">✨ Amenities & Features</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    mainroad         = st.selectbox("Main Road Access",  ["yes", "no"])
    guestroom        = st.selectbox("Guest Room",        ["yes", "no"])
    basement         = st.selectbox("Basement",          ["yes", "no"])
    hotwaterheating  = st.selectbox("Hot Water Heating", ["yes", "no"])
with col4:
    airconditioning  = st.selectbox("Air Conditioning",  ["yes", "no"])
    prefarea         = st.selectbox("Preferred Area",    ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔮 Predict House Price"):
    user_input = pd.DataFrame([{
        'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'stories': stories, 'parking': parking,
        'mainroad': mainroad, 'guestroom': guestroom, 'basement': basement,
        'hotwaterheating': hotwaterheating, 'airconditioning': airconditioning,
        'prefarea': prefarea, 'furnishingstatus': furnishingstatus
    }])

    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    predicted_price = model.predict(user_encoded)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Property Value</div>
        <div class="result-price">&#8360; {predicted_price:,.0f}</div>
        <div class="result-note">Based on Linear Regression model &middot; Results are estimates only</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 View Input Summary"):
        summary = {
            "Area (sq ft)": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
            "Stories": stories, "Parking": parking, "Main Road": mainroad,
            "Guest Room": guestroom, "Basement": basement,
            "Hot Water Heating": hotwaterheating, "Air Conditioning": airconditioning,
            "Preferred Area": prefarea, "Furnishing": furnishingstatus
        }
        st.dataframe(pd.DataFrame(summary.items(), columns=["Feature", "Value"]),
                     use_container_width=True, hide_index=True)