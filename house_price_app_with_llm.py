import pandas as pd
import pickle
import streamlit as st
import itertools
import re
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# ─── CSS ─────────────────────────────────────────────────────────────────────
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
.stNumberInput label, .stSelectbox label, .stTextInput label {
    color: #cbd5e1 !important; font-size: 0.875rem !important; font-weight: 500 !important;
}
.stNumberInput input, .stTextInput input {
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
.stButton > button:hover { box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important; }
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
.budget-over {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(239,68,68,0.1));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.budget-ok {
    background: linear-gradient(135deg, rgba(5,150,105,0.15), rgba(16,185,129,0.1));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.badge-over { color: #f87171; font-size: 2rem; font-weight: 700; }
.badge-ok   { color: #34d399; font-size: 2rem; font-weight: 700; }
.ai-box {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(167,139,250,0.25);
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-top: 1rem;
    color: #cbd5e1; font-size: 0.92rem; line-height: 1.75;
}
.ai-box ul { list-style: none; padding-left: 0; }
.ai-box li::before { content: "→ "; color: #a78bfa; font-weight: 600; }
.ai-box b { color: #e2e8f0; }
.ai-box p { margin: 0.3rem 0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
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

# ─── Helper: Encode & Predict ────────────────────────────────────────────────
def predict_price(params: dict) -> float:
    df_input = pd.DataFrame([params])
    encoded  = pd.get_dummies(df_input)
    encoded  = encoded.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return model.predict(encoded)[0]

# ─── Helper: Find cheapest combos under budget ───────────────────────────────
def find_budget_combos(base_params: dict, budget: float):
    yes_no_fields = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    furnishing_options = ['unfurnished', 'semi-furnished', 'furnished']
    cheaper_variants = []
    fields_currently_yes = [f for f in yes_no_fields if base_params.get(f) == 'yes']
    area_options = [base_params['area']]
    if base_params['area'] > 1000:
        area_options = [base_params['area'], base_params['area'] - 500, base_params['area'] - 1000]
        area_options = [a for a in area_options if a >= 500]
    furnish_idx = furnishing_options.index(base_params['furnishingstatus'])
    furnish_cheaper = furnishing_options[:furnish_idx + 1]
    for r in range(0, len(fields_currently_yes) + 1):
        for fields_to_flip in itertools.combinations(fields_currently_yes, r):
            for furnish in furnish_cheaper:
                for area_val in area_options:
                    variant = base_params.copy()
                    for f in fields_to_flip:
                        variant[f] = 'no'
                    variant['furnishingstatus'] = furnish
                    variant['area'] = area_val
                    price = predict_price(variant)
                    if price <= budget:
                        changes = []
                        for f in fields_to_flip:
                            changes.append(f"{f}: yes → no")
                        if furnish != base_params['furnishingstatus']:
                            changes.append(f"furnishing: {base_params['furnishingstatus']} → {furnish}")
                        if area_val != base_params['area']:
                            changes.append(f"area: {base_params['area']} → {area_val} sq ft")
                        cheaper_variants.append({"params": variant, "price": price,
                                                  "changes": changes, "n_changes": len(changes)})
            if len(cheaper_variants) >= 20:
                break
        if len(cheaper_variants) >= 20:
            break
    cheaper_variants.sort(key=lambda x: (x['n_changes'], x['price']))
    return cheaper_variants[:5]

# ─── Helper: Format Advice ───────────────────────────────────────────────────
def format_advice(text: str) -> str:
    lines = text.strip().split('\n')
    html_lines = []
    in_list = False
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<br>')
            continue
        if re.match(r'^[-*]\s+', line):
            if not in_list:
                html_lines.append('<ul style="margin:0.6rem 0 0.6rem 1.2rem; padding:0;">')
                in_list = True
            content = re.sub(r'^[-*]\s+', '', line)
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
            html_lines.append(f'<li style="margin-bottom:0.4rem;">{content}</li>')
            continue
        if in_list:
            html_lines.append('</ul>')
            in_list = False
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        if re.match(r'^\d+\.\s+', line):
            line = re.sub(r'^(\d+\.)\s+', r'<b style="color:#a78bfa">\1</b> ', line)
        html_lines.append(f'<p style="margin:0.4rem 0;">{line}</p>')
    if in_list:
        html_lines.append('</ul>')
    return '\n'.join(html_lines)

# ─── Helper: Get pydantic-ai GroqModel (same as 2nd file) ────────────────────
def get_groq_model():
    try:
        api_key = st.secrets["groq_clouds"].strip()
    except KeyError:
        st.error("Secret 'groq_clouds' not found. Add it in Streamlit Cloud → Settings → Secrets.")
        return None
    if not api_key:
        st.error("Groq API key is empty. Check your Streamlit secrets.")
        return None
    try:
        return GroqModel(
            'llama-3.3-70b-versatile',
            provider=GroqProvider(api_key=api_key)
        )
    except Exception as e:
        st.error(f"Failed to initialise GroqModel: {repr(e)}")
        return None

# ─── Helper: Ask Groq via pydantic-ai Agent ──────────────────────────────────
async def _ask_groq_async(prompt: str) -> str:
    llm = get_groq_model()
    if llm is None:
        return "Model initialisation failed."
    agent = Agent(
        model=llm,
        system_prompt="You are a helpful and friendly real estate advisor. Be concise, warm, and practical. Use bullet points where helpful."
    )
    result = await agent.run(prompt)
    return result.output if hasattr(result, 'output') else str(result)

def ask_groq(prompt: str) -> str:
    return asyncio.run(_ask_groq_async(prompt))

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Predictor</h1>
    <p>Get instant price estimates and AI-powered budget advice</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ `linear_model.pkl` not found. Place it in the same folder as this app.")
    st.stop()

# ─── Property Details ────────────────────────────────────────────────────────
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

# ─── Amenities ───────────────────────────────────────────────────────────────
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

# ─── Budget ───────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">💰 Your Budget (Optional)</div>', unsafe_allow_html=True)
budget = st.number_input("Budget ($)", min_value=0, max_value=50000000, value=0, step=50000,
                          help="Set to 0 to skip budget analysis")
st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔮 Predict & Analyse"):

    base_params = {
        'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'stories': stories, 'parking': parking,
        'mainroad': mainroad, 'guestroom': guestroom, 'basement': basement,
        'hotwaterheating': hotwaterheating, 'airconditioning': airconditioning,
        'prefarea': prefarea, 'furnishingstatus': furnishingstatus
    }

    predicted_price = predict_price(base_params)

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Property Value</div>
        <div class="result-price">$ {predicted_price:,.0f}</div>
        <div class="result-note">Based on Linear Regression model &middot; Estimates only</div>
    </div>
    """, unsafe_allow_html=True)

    if budget > 0:
        diff     = predicted_price - budget
        over_pct = (diff / budget) * 100

        if predicted_price <= budget:
            saving = budget - predicted_price
            st.markdown(f"""
            <div class="budget-ok">
                <div class="badge-ok">✅ Within Budget!</div>
                <p style="color:#94a3b8; margin-top:0.5rem;">
                    You're saving <b style="color:#34d399">$ {saving:,.0f}</b>
                    &nbsp;({abs(over_pct):.1f}% under budget)
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🤖 Getting AI advice..."):
                prompt = f"""
A buyer has a budget of ${budget:,.0f} and found a house predicted at ${predicted_price:,.0f}.
House: {area} sq ft, {bedrooms} beds, {bathrooms} baths, {stories} stories, {parking} parking.
Features: main road={mainroad}, guestroom={guestroom}, basement={basement},
hot water={hotwaterheating}, AC={airconditioning}, preferred area={prefarea}, furnishing={furnishingstatus}.
They are ${saving:,.0f} under budget ({abs(over_pct):.1f}%). Give 3-4 bullet points:
- Confirm this is a good deal
- What features make it valuable
- How to use the remaining budget (renovations, savings, etc.)
Keep it friendly and encouraging. Currency is Dollars.
"""
                try:
                    raw     = ask_groq(prompt)
                    formatted = format_advice(raw)
                    st.markdown(f'<div class="ai-box"><p style="color:#a78bfa;font-weight:600;margin-bottom:0.8rem;">🤖 Expert Suggestion</p>{formatted}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"AI advice unavailable: {e}")

        else:
            st.markdown(f"""
            <div class="budget-over">
                <div class="badge-over">⚠️ Over Budget</div>
                <p style="color:#94a3b8; margin-top:0.5rem;">
                    Exceeds budget by <b style="color:#f87171">$ {diff:,.0f}</b>
                    &nbsp;({over_pct:.1f}% over)
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🔍 Finding alternatives within your budget..."):
                alternatives = find_budget_combos(base_params, budget)

            if alternatives:
                st.markdown("### 💡 Cheaper Alternatives Within Your Budget")
                for i, alt in enumerate(alternatives[:3], 1):
                    change_text = ", ".join(alt['changes']) if alt['changes'] else "No changes needed"
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Option {i} &mdash; $ {alt['price']:,.0f}</div>
                        <p style="color:#94a3b8; font-size:0.9rem;">Changes: <span style="color:#a78bfa">{change_text}</span></p>
                        <p style="color:#34d399; font-size:0.85rem;">Saves $ {predicted_price - alt['price']:,.0f} from original estimate</p>
                    </div>
                    """, unsafe_allow_html=True)

                with st.spinner("🤖 Getting AI advice..."):
                    alt_summaries = "\n".join([
                        f"Option {i+1}: ${a['price']:,.0f} — changes: {', '.join(a['changes']) or 'minimal'}"
                        for i, a in enumerate(alternatives[:3])
                    ])
                    prompt = f"""
A buyer has a budget of ${budget:,.0f} but the house costs ${predicted_price:,.0f} (over by ${diff:,.0f}).
Original: {area} sq ft, {bedrooms} beds, {bathrooms} baths, furnishing={furnishingstatus},
AC={airconditioning}, preferred area={prefarea}, main road={mainroad}.
Cheaper alternatives:
{alt_summaries}
Give 4-5 bullet points: acknowledge the challenge, best trade-off option, features to keep vs drop, negotiation tips.
Be friendly and helpful. Currency is Dollars.
"""
                    try:
                        raw       = ask_groq(prompt)
                        formatted = format_advice(raw)
                        st.markdown(f'<div class="ai-box"><p style="color:#a78bfa;font-weight:600;margin-bottom:0.8rem;">🤖 Expert Suggestion</p>{formatted}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"AI advice unavailable: {e}")
            else:
                st.warning("No combinations found within your budget. Consider increasing budget or reducing area/stories.")

    with st.expander("📋 View Input Summary"):
        summary = {
            "Area (sq ft)": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
            "Stories": stories, "Parking": parking, "Main Road": mainroad,
            "Guest Room": guestroom, "Basement": basement,
            "Hot Water Heating": hotwaterheating, "Air Conditioning": airconditioning,
            "Preferred Area": prefarea, "Furnishing": furnishingstatus,
            "Budget": f"$ {budget:,.0f}" if budget > 0 else "Not set"
        }
        st.dataframe(pd.DataFrame(summary.items(), columns=["Feature", "Value"]),
                     use_container_width=True, hide_index=True)
