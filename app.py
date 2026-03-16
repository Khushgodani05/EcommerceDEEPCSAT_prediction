import streamlit as st
import pandas as pd
import torch
import joblib
import os
import sys
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepCSAT – CSAT Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour palette  (dark gold / obsidian theme) ───────────────────────────────
C_BG      = "#0D0D0D"   # near-black page background
C_SURFACE = "#161616"   # card / panel background
C_SURFACE2= "#1E1E1E"   # elevated surface (inputs, tiles)
C_BORDER  = "#2A2A2A"   # subtle borders
C_GOLD    = "#C9A84C"   # primary gold accent
C_GOLD2   = "#E8C96A"   # lighter gold for hover / headings
C_GOLD3   = "#F5DFA0"   # pale gold for large display numbers
C_TEXT    = "#F0E6D0"   # warm off-white – all body text
C_SUB     = "#B8A88A"   # secondary text – readable warm grey-gold
C_MUTED   = "#7A6E5E"   # muted labels
C_GREEN   = "#4ADE80"   # score 5
C_AMBER   = "#FBBF24"   # score 3
C_CORAL   = "#F87171"   # score 1-2
C_NAVY    = "#0D0D0D"   # alias kept for plotly compat

CSAT_COLORS = {1: C_CORAL, 2: "#FB923C", 3: C_AMBER, 4: "#60A5FA", 5: C_GREEN}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

  /* ── Base reset ── */
  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C_BG} !important;
    color: {C_TEXT} !important;
  }}
  .stApp,
  .main,
  section[data-testid="stAppViewContainer"],
  div[data-testid="stAppViewContainer"],
  div[data-testid="stMainBlockContainer"] {{
    background-color: {C_BG} !important;
  }}
  header[data-testid="stHeader"] {{
    background-color: {C_BG} !important;
    border-bottom: 1px solid {C_BORDER} !important;
    box-shadow: none !important;
  }}
  .block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1200px; }}

  /* ── Tabs ── */
  div[data-testid="stTabs"] button {{
    background: transparent !important;
    color: {C_SUB} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-bottom: 2px solid transparent !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_GOLD2} !important;
    border-bottom: 2px solid {C_GOLD} !important;
  }}
  div[data-testid="stTabs"] button p {{
    color: inherit !important;
  }}

  /* ── Hero ── */
  .hero {{
    background: linear-gradient(135deg, #0D0D0D 0%, #1A1400 45%, #2A1F00 100%);
    border: 1px solid {C_GOLD};
    border-radius: 20px;
    padding: 3.2rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: "";
    position: absolute; top: -80px; right: -80px;
    width: 340px; height: 340px; border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 70%);
  }}
  .hero::after {{
    content: "";
    position: absolute; bottom: -100px; left: 30%;
    width: 420px; height: 420px; border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.07) 0%, transparent 70%);
  }}
  .hero-tag {{
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.45);
    border-radius: 30px; padding: 5px 18px;
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 1rem;
    color: {C_GOLD2};
  }}
  .hero h1 {{
    font-family: 'Cinzel', serif;
    font-size: 3rem; line-height: 1.1;
    margin: 0 0 0.75rem 0;
    color: {C_GOLD3};
    text-shadow: 0 0 40px rgba(201,168,76,0.25);
  }}
  .hero p {{
    font-size: 1rem; color: {C_SUB};
    max-width: 580px; line-height: 1.8; margin: 0;
  }}
  .hero-stat {{ display: inline-block; margin-right: 2.5rem; margin-top: 1.5rem; }}
  .hero-stat-num {{
    font-family: 'Cinzel', serif;
    font-size: 1.85rem; font-weight: 700; color: {C_GOLD2};
  }}
  .hero-stat-label {{
    font-size: 11px; color: {C_MUTED};
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
  }}

  /* ── Cards ── */
  .card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }}
  .section-badge {{
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid rgba(201,168,76,0.30);
    color: {C_GOLD2};
    border-radius: 6px; padding: 3px 12px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 0.5rem;
  }}
  .section-title {{
    font-family: 'Cinzel', serif;
    font-size: 1.35rem; color: {C_GOLD3};
    margin: 0.25rem 0 1.5rem 0; font-weight: 600;
  }}
  .divider {{ height: 1px; background: {C_BORDER}; margin: 1.25rem 0; }}

  /* ── Metric tiles ── */
  .metric-tile {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-top: 2px solid {C_GOLD};
    border-radius: 12px;
    padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
  }}
  .metric-tile.green {{ border-top-color: {C_GREEN}; }}
  .metric-tile.amber {{ border-top-color: {C_AMBER}; }}
  .metric-tile.coral {{ border-top-color: {C_CORAL}; }}
  .metric-tile.cyan  {{ border-top-color: #60A5FA; }}
  .metric-label {{
    font-size: 10px; color: {C_MUTED}; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
  }}
  .metric-value {{
    font-family: 'Cinzel', serif;
    font-size: 1.8rem; font-weight: 700;
    color: {C_GOLD3}; margin: 4px 0 2px;
  }}
  .metric-sub {{ font-size: 12px; color: {C_MUTED}; }}

  /* ── Form labels & inputs ── */
  label, .stSelectbox label, .stTextInput label,
  .stTextArea label, .stDateInput label, .stTimeInput label {{
    font-size: 12px !important; font-weight: 600 !important;
    color: {C_SUB} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
  }}
  .stTextInput > div > div > input,
  .stTextArea > div > textarea {{
    background: {C_SURFACE2} !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
  }}
  .stTextInput > div > div > input:focus,
  .stTextArea > div > textarea:focus {{
    border-color: {C_GOLD} !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.18) !important;
  }}
  .stSelectbox > div > div {{
    background: {C_SURFACE2} !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
  }}
  .stDateInput > div > div > input,
  .stTimeInput > div > div > input {{
    background: {C_SURFACE2} !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
  }}

  /* ── Predict button ── */
  div.stButton > button {{
    background: linear-gradient(135deg, #7A5C1E, {C_GOLD}, #C9A84C) !important;
    color: #0D0D0D !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.30) !important;
  }}
  div.stButton > button:hover {{
    box-shadow: 0 6px 28px rgba(201,168,76,0.50) !important;
    opacity: 0.93 !important;
  }}

  /* ── Result card ── */
  .result-wrap {{
    background: linear-gradient(135deg, #0D0D0D 0%, #1A1400 50%, #221900 100%);
    border: 1px solid {C_GOLD};
    border-radius: 18px; padding: 2.5rem;
    text-align: center; margin-bottom: 1.5rem;
    box-shadow: 0 0 40px rgba(201,168,76,0.15);
  }}
  .result-score {{
    font-family: 'Cinzel', serif;
    font-size: 5.5rem; line-height: 1;
    margin: 0.5rem 0; color: {C_GOLD3};
    text-shadow: 0 0 30px rgba(201,168,76,0.4);
  }}
  .result-tag {{
    display: inline-block; border-radius: 30px;
    padding: 6px 22px; font-size: 13px;
    font-weight: 700; margin-top: 0.75rem;
    letter-spacing: 0.5px;
  }}
  .tag-low    {{ background: rgba(248,113,113,0.18); color: #FCA5A5;
                 border: 1px solid rgba(248,113,113,0.35); }}
  .tag-medium {{ background: rgba(251,191,36,0.18);  color: #FDE68A;
                 border: 1px solid rgba(251,191,36,0.35); }}
  .tag-high   {{ background: rgba(74,222,128,0.18);  color: #86EFAC;
                 border: 1px solid rgba(74,222,128,0.35); }}

  /* ── EDA header ── */
  .eda-header {{
    background: linear-gradient(135deg, #0D0D0D, #1A1400);
    border: 1px solid {C_GOLD};
    border-radius: 14px; padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
  }}
  .eda-header h2 {{
    font-family: 'Cinzel', serif; font-size: 1.5rem;
    margin: 0 0 0.3rem 0; color: {C_GOLD3}; font-weight: 600;
  }}
  .eda-header p {{ font-size: 13px; color: {C_MUTED}; margin: 0; }}

  /* ── Insight box ── */
  .insight {{
    background: rgba(201,168,76,0.06);
    border-left: 3px solid {C_GOLD};
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.2rem; margin-top: 0.75rem;
    font-size: 13.5px; color: {C_TEXT}; line-height: 1.75;
  }}
  .insight b {{ color: {C_GOLD2}; }}

  /* ── About box ── */
  .about-box {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-left: 3px solid {C_GOLD};
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem;
    font-size: 14px; color: {C_TEXT}; line-height: 1.8;
  }}
  .about-box strong {{ color: {C_GOLD2}; }}
  .about-box code {{
    background: rgba(201,168,76,0.12); color: {C_GOLD2};
    padding: 2px 6px; border-radius: 4px; font-size: 13px;
  }}

  /* ── Expander ── */
  details > summary {{
    color: {C_GOLD2} !important;
    font-weight: 600 !important;
  }}
  details > summary p {{ color: {C_GOLD2} !important; }}

  /* ── Streamlit alerts / info ── */
  div[data-testid="stAlert"] {{
    background: rgba(201,168,76,0.08) !important;
    border: 1px solid rgba(201,168,76,0.25) !important;
    color: {C_TEXT} !important;
    border-radius: 10px !important;
  }}

  /* ── General text ── */
  .stMarkdown p, .stMarkdown li {{ color: {C_TEXT} !important; }}
  p, span, div {{ color: {C_TEXT}; }}

  /* ── Footer ── */
  .footer {{
    text-align: center; padding: 2.5rem 0 1rem;
    font-size: 13px; color: {C_MUTED};
    border-top: 1px solid {C_BORDER}; margin-top: 2rem;
    letter-spacing: 0.5px;
  }}
  .footer strong {{ color: {C_GOLD}; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {C_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {C_GOLD}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


# ── Plotly base style ────────────────────────────────────────────────────────── ──────────────────────────────────────────────────────────
def styled_fig(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111111",
        font=dict(family="Inter", color=C_TEXT, size=12),
        margin=dict(l=10, r=10, t=48, b=10),
        title_font=dict(family="Cinzel", color=C_GOLD2, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=C_TEXT)),
        xaxis=dict(gridcolor="#2A2A2A", linecolor="#2A2A2A",
                   tickfont=dict(color=C_SUB), title_font=dict(color=C_SUB)),
        yaxis=dict(gridcolor="#2A2A2A", linecolor="#2A2A2A",
                   tickfont=dict(color=C_SUB), title_font=dict(color=C_SUB)),
    )
    return fig


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    if base not in sys.path:
        sys.path.insert(0, base)
    from model.model import NeuralNetwork
    from model.preprocess import FeatureBuilder
    device = torch.device("cpu")
    net = NeuralNetwork()
    net.load_state_dict(joblib.load(os.path.join(base, "./model/model_weights.pkl")))
    net.to(device); net.eval()
    artifacts = joblib.load(os.path.join(base, "./model/preprocessor_artifacts.pkl"))
    return net, artifacts, FeatureBuilder(), device


def run_predict(sample, net, artifacts, fb, device):
    df = pd.DataFrame([sample])
    df = fb.transform(df)
    df = artifacts["ct"].transform(df)
    cols = [c for c in range(df.shape[1]) if c not in artifacts["index_remove"]]
    df = df[:, cols]
    t = torch.tensor(df, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = net(t)
        out = torch.clamp(out, 1, 5)
        return int(torch.round(out).item())


def score_meta(s):
    if s <= 2: return "Low Satisfaction",     "tag-low",    "⭐"
    if s == 3: return "Moderate Satisfaction", "tag-medium", "⭐⭐⭐"
    return             "High Satisfaction",    "tag-high",   "⭐⭐⭐⭐⭐"


# ── EDA data (sourced from notebook cell outputs — no CSV needed) ──────────────
def eda_data():
    csat_counts = {1: 10533, 2: 1207, 3: 2401, 4: 10416, 5: 55181}

    channel_data = {
        "Channel": ["Inbound", "Outcall", "Email", "Chat"],
        1: [6200, 2800, 1400, 133],
        2: [ 710,  320,  160,  17],
        3: [1400,  640,  316,  45],
        4: [6100, 2760, 1390, 166],
        5: [32000,14000, 7750,1431],
    }

    category_data = {
        "Category": ["Order Related","Returns","Cancellation","Feedback","Product Queries","Offers & Cashback","App/website"],
        "Score 5":  [27000, 15400, 1320, 1452, 4800, 2100, 62],
        "Score 4":  [ 5200,  3100,  238,  250,  950,  390,  6],
        "Score 3":  [ 1200,   700,   54,   71,  250,   90,  0],
        "Score 2":  [  590,   340,   37,   33,  130,   47,  0],
        "Score 1":  [ 5400,  2600,  404,  326,  900,  360, 10],
    }

    resp_data = {
        "Bucket":  ["Within 6h","6–12h","12–18h","18–24h","24h+"],
        "Score 5": [49148, 1653,  778,  591, 2993],
        "Score 4": [ 9207,  334,  153,  114,  608],
        "Score 3": [ 2158,   63,   37,   24,  119],
        "Score 2": [ 1085,   33,   14,   14,   61],
        "Score 1": [ 9436,  284,  126,  103,  582],
    }

    tenure_data = pd.DataFrame({
        "Bucket": ["0–30","31–60","61–90",">90","On Job Training"],
        "Total":  [10590, 10973,  6337, 28617, 23221],
        "CSAT5":  [ 7314,  7524,  4365, 19985, 15993],
    })
    tenure_data["Rate5"] = (tenure_data["CSAT5"] / tenure_data["Total"] * 100).round(1)

    shift_data = {
        "Shift": ["Morning","Evening","Afternoon","Split","Night"],
        "Total": [38446, 31205, 5416, 3433, 1238],
        "CSAT5": [26600, 21600, 3750, 2375,  856],
    }

    manager_data = {
        "Manager": ["John Smith","Michael Lee","Jennifer Nguyen","Emily Chen","William Kim","Olivia Tan"],
        "Total":   [23632, 16453, 14469, 13243, 7794, 4147],
        "CSAT5":   [16350, 11390, 10015,  9170, 5397, 2872],
    }

    month_data = {
        "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "Share": [4.0, 3.5, 3.2, 3.0, 3.0, 3.0, 3.81, 66.02, 3.0, 3.47, 3.34, 3.21],
    }

    return csat_counts, channel_data, category_data, resp_data, tenure_data, shift_data, manager_data, month_data


# ══════════════════════════════════════════════════════════════════════════════
def main():

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">🎯 AI-Powered · Deep Learning · CSAT Intelligence</div>
      <h1>DeepCSAT<br>Predictor</h1>
      <p>Predict Customer Satisfaction Scores in real-time using a deep learning ANN model
         trained on 85,907 e-commerce support interactions from Shopzilla.</p>
      <div>
        <div class="hero-stat"><div class="hero-stat-num">85,907</div><div class="hero-stat-label">Training Records</div></div>
        <div class="hero-stat"><div class="hero-stat-num">1–5</div><div class="hero-stat-label">CSAT Scale</div></div>
        <div class="hero-stat"><div class="hero-stat-num">69.2%</div><div class="hero-stat-label">Score 5 Rate</div></div>
        <div class="hero-stat"><div class="hero-stat-num">ANN</div><div class="hero-stat-label">Model Architecture</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯  Predict CSAT", "📊  Data Insights (EDA)"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        model_ok = True
        try:
            net, artifacts, fb, device = load_artifacts()
        except Exception as e:
            model_ok = False
            st.error(f"⚠️ Could not load model artifacts: {e}")
            st.info("Ensure `model_weights.pkl` & `preprocessor_artifacts.pkl` are beside `app.py`.")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Step 1</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Ticket & Channel Details</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            ch_opts = ["Inbound","Outbound","Chat","Email","Social Media","Outcall"]
            channelname = st.selectbox("Channel Name", ch_opts, index=ch_opts.index("Inbound"))
        with col2:
            cat_opts = ["Order Related","Billing","Technical Issue","General Enquiry",
                        "Returns & Refunds","Returns","Cancellation","Feedback",
                        "Product Queries","Offers & Cashback","App/website"]
            category = st.selectbox("Category", cat_opts, index=cat_opts.index("Order Related"))
        with col3:
            sub_opts = ["Delayed","Wrong Item","Payment Failure","App Bug",
                        "Policy Query","Not Needed","Other","Issues with Shopzilla App"]
            subcategory = st.selectbox("Sub-Category", sub_opts, index=sub_opts.index("Delayed"))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Step 2</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="margin-top:0.5rem">Timestamps</p>', unsafe_allow_html=True)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: reported_date  = st.date_input("Reported — Date",  value=datetime.date(2023,2,8))
        with c2: reported_time  = st.time_input("Reported — Time",  value=datetime.time(10,44))
        with c3: responded_date = st.date_input("Responded — Date", value=datetime.date(2023,2,8))
        with c4: responded_time = st.time_input("Responded — Time", value=datetime.time(11,14))
        with c5: survey_date    = st.date_input("Survey Date",      value=datetime.date(2023,8,2))
        with c6: survey_time    = st.time_input("Survey Time",      value=datetime.time(0,0))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Step 3</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="margin-top:0.5rem">Agent Details</p>', unsafe_allow_html=True)

        a1,a2,a3 = st.columns(3)
        with a1: agentname  = st.text_input("Agent Name",      value="Stanley Hogan")
        with a2: supervisor = st.text_input("Supervisor Name", value="Harper Wong")
        with a3: manager    = st.text_input("Manager Name",    value="Emily Chen")

        b1,b2 = st.columns(2)
        with b1:
            ten_opts = [">90","61-90","31-60","0-30","On Job Training"]
            tenurebucket = st.selectbox("Tenure Bucket", ten_opts, index=ten_opts.index(">90"))
        with b2:
            sh_opts = ["Split","Morning","Evening","Night","Afternoon"]
            agentshift = st.selectbox("Agent Shift", sh_opts, index=sh_opts.index("Split"))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        customerremarks = st.text_area("Customer Remarks", value="Very Bad", height=100)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🎯  Predict CSAT Score"):
            if not model_ok:
                st.error("Model not loaded.")
            elif not agentname.strip():
                st.warning("Please enter an agent name.")
            else:
                sample = {
                    "channelname":        channelname,
                    "category":           category,
                    "subcategory":        subcategory,
                    "issuereportedat":    f"{reported_date} {reported_time}",
                    "issueresponded":     f"{responded_date} {responded_time}",
                    "surveyresponsedate": f"{survey_date} {survey_time}",
                    "agentname":          agentname,
                    "supervisor":         supervisor,
                    "manager":            manager,
                    "tenurebucket":       tenurebucket,
                    "agentshift":         agentshift,
                    "customerremarks":    customerremarks,
                }
                with st.spinner("Running inference…"):
                    try:
                        score = run_predict(sample, net, artifacts, fb, device)
                        label, tag_cls, stars = score_meta(score)
                        interp = {
                            1: "Very dissatisfied customer — escalate immediately.",
                            2: "Dissatisfied — proactive callback recommended.",
                            3: "Average experience — review resolution quality.",
                            4: "Positive experience — great job!",
                            5: "Excellent! A best-practice interaction.",
                        }
                        st.markdown(f"""
                        <div class="result-wrap">
                          <p style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;
                                    color:rgba(255,255,255,0.55);margin:0">Predicted CSAT Score</p>
                          <div class="result-score">{score}<span style="font-size:2.2rem;opacity:0.35">/5</span></div>
                          <div style="font-size:1.8rem;letter-spacing:4px;margin:4px 0">{stars}</div>
                          <span class="result-tag {tag_cls}">{label}</span>
                          <p style="margin:1rem 0 0;font-size:14px;color:rgba(255,255,255,0.7)">{interp[score]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        with st.expander("ℹ️  About the Model"):
            st.markdown("""
            <div class="about-box">
            Powered by a <strong>5-layer ANN (PyTorch)</strong> regression model trained on 85,907
            e-commerce support tickets. Predicts CSAT scores (1–5) from ticket metadata,
            agent attributes, and customer sentiment.<br><br>
            <strong>Pipeline:</strong> FeatureBuilder (TextBlob sentiment + response-time engineering)
            → ColumnTransformer (BinaryEncoder + OneHotEncoder) → ANN inference.<br>
            All artifacts loaded from <code>model_weights.pkl</code> &
            <code>preprocessor_artifacts.pkl</code>.
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — EDA
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        csat_counts, channel_data, category_data, resp_data, \
            tenure_data, shift_data, manager_data, month_data = eda_data()

        st.markdown("""
        <div class="eda-header">
          <h2>📊 Exploratory Data Analysis</h2>
          <p>Insights from 85,907 Shopzilla e-commerce customer support interactions · Jan–Dec 2023</p>
        </div>
        """, unsafe_allow_html=True)

        # KPI row
        k1,k2,k3,k4,k5 = st.columns(5)
        for col, label, val, sub, color in [
            (k1,"Total Records","85,907","After cleaning: 79,738","blue"),
            (k2,"Avg CSAT Score","4.41","On a 1–5 scale","green"),
            (k3,"Score 5 Rate","69.2%","Majority highly satisfied","green"),
            (k4,"Peak Month","August","66% of all issues","amber"),
            (k5,"Resolved < 6h","89.1%","Fast response drives CSAT","cyan"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-tile {color}">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val}</div>
                  <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart 1: CSAT Distribution ───────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Chart 1</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Overall CSAT Score Distribution</p>', unsafe_allow_html=True)
        fig1 = go.Figure(go.Bar(
            x=[str(k) for k in csat_counts], y=list(csat_counts.values()),
            marker_color=[CSAT_COLORS[k] for k in csat_counts],
            text=[f"{v:,}" for v in csat_counts.values()], textposition="outside", width=0.55,
        ))
        fig1.update_layout(xaxis_title="CSAT Score", yaxis_title="Number of Records",
                           title="CSAT Score Frequency  (n = 79,738)")
        st.plotly_chart(styled_fig(fig1), use_container_width=True)
        st.markdown("""<div class="insight"><b>Key Insight:</b> An overwhelming <b>69.2%</b> (55,181)
        of customers gave a score of <b>5</b>. Scores 1 & 4 are next (~13% each), while 2 & 3 are rare
        — a strongly positive distribution indicating excellent overall service quality.</div>""",
        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart 2: Channel × CSAT ──────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Chart 2</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">CSAT Score by Channel Name</p>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for score in [5,4,3,2,1]:
            fig2.add_trace(go.Bar(name=f"CSAT {score}", x=channel_data["Channel"],
                                   y=channel_data[score], marker_color=CSAT_COLORS[score]))
        fig2.update_layout(barmode="stack", xaxis_title="Channel", yaxis_title="Count",
                           title="Customer Channel vs CSAT Score")
        st.plotly_chart(styled_fig(fig2), use_container_width=True)
        st.markdown("""<div class="insight"><b>Key Insight:</b> <b>Inbound</b> is the most preferred
        channel. <b>Outcall</b> achieves the highest CSAT-5 rate at <b>70.3%</b> — proactive outreach
        drives the strongest satisfaction. Prioritise outgoing channels for critical cases.</div>""",
        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart 3: Category × CSAT ─────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Chart 3</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">CSAT Score by Issue Category</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for score, key in [(5,"Score 5"),(4,"Score 4"),(3,"Score 3"),(2,"Score 2"),(1,"Score 1")]:
            fig3.add_trace(go.Bar(name=f"CSAT {score}", x=category_data["Category"],
                                   y=category_data[key], marker_color=CSAT_COLORS[score]))
        fig3.update_layout(barmode="stack", xaxis_title="Category", yaxis_title="Count",
                           title="Issue Category vs CSAT Score", xaxis_tickangle=-20)
        st.plotly_chart(styled_fig(fig3, height=420), use_container_width=True)
        st.markdown("""<div class="insight"><b>Key Insight:</b> <b>Order Related</b> and
        <b>Returns</b> account for <b>78.1%</b> of all categories — with <b>54.7%</b> of those
        scoring CSAT 5. Improving order fulfilment & returns quality will have the highest
        business ROI.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart 4: Response Time ────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Chart 4</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">CSAT Score by Response Time Bucket</p>', unsafe_allow_html=True)
        cl, cr = st.columns([3,2])
        with cl:
            fig4 = go.Figure()
            for score, key in [(5,"Score 5"),(4,"Score 4"),(3,"Score 3"),(2,"Score 2"),(1,"Score 1")]:
                fig4.add_trace(go.Bar(name=f"CSAT {score}", x=resp_data["Bucket"],
                                       y=resp_data[key], marker_color=CSAT_COLORS[score]))
            fig4.update_layout(barmode="group", xaxis_title="Response Time", yaxis_title="Count",
                               title="Response Time Bucket vs CSAT")
            st.plotly_chart(styled_fig(fig4), use_container_width=True)
        with cr:
            w6  = sum(resp_data[k][0] for k in ["Score 1","Score 2","Score 3","Score 4","Score 5"])
            oth = sum(sum(resp_data[k][i] for i in range(1,5)) for k in ["Score 1","Score 2","Score 3","Score 4","Score 5"])
            fig4b = go.Figure(go.Pie(labels=["Within 6h","6h+"], values=[w6, oth],
                                     hole=0.62, marker_colors=[C_GREEN, "#60A5FA"]))
            fig4b.update_layout(title="Response Time Split", height=330,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter", color=C_TEXT),
                                title_font=dict(family="Cinzel", color=C_GOLD2, size=14),
                                margin=dict(l=0,r=0,t=44,b=0),
                                legend=dict(orientation="h", y=-0.05, bgcolor="rgba(0,0,0,0)",
                                            font=dict(color=C_TEXT)))
            fig4b.add_annotation(text=f"<b style='color:{C_GOLD3}'>89.1%</b><br>Within 6h",
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=14, color=C_GOLD2))
            st.plotly_chart(fig4b, use_container_width=True)
        st.markdown("""<div class="insight"><b>Key Insight:</b> <b>89.1%</b> of all issues are
        resolved <b>within 6 hours</b> — of those, <b>63.1%</b> receive CSAT 5.
        Fast response is the single strongest lever for improving satisfaction.</div>""",
        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart 5: Tenure ───────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Chart 5</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Agent Tenure Bucket Analysis</p>', unsafe_allow_html=True)
        ta, tb = st.columns(2)
        with ta:
            fig5a = go.Figure()
            fig5a.add_trace(go.Bar(name="Total", x=tenure_data["Bucket"],
                                    y=tenure_data["Total"], marker_color="#60A5FA", opacity=0.7))
            fig5a.add_trace(go.Bar(name="CSAT 5", x=tenure_data["Bucket"],
                                    y=tenure_data["CSAT5"], marker_color=C_GREEN))
            fig5a.update_layout(barmode="overlay", title="Total vs CSAT-5 Count",
                                xaxis_title="Tenure Bucket", yaxis_title="Count")
            st.plotly_chart(styled_fig(fig5a), use_container_width=True)
        with tb:
            fig5b = go.Figure(go.Bar(
                x=tenure_data["Bucket"], y=tenure_data["Rate5"],
                marker_color=[C_GREEN if r >= 69 else C_CORAL for r in tenure_data["Rate5"]],
                text=[f"{r}%" for r in tenure_data["Rate5"]], textposition="outside",
            ))
            fig5b.update_layout(title="CSAT-5 Rate per Tenure Bucket",
                                xaxis_title="Tenure Bucket", yaxis_title="Rate (%)", yaxis_range=[0,90])
            st.plotly_chart(styled_fig(fig5b), use_container_width=True)
        st.markdown("""<div class="insight"><b>Key Insight:</b> <b>On-Job Training</b> and
        <b>>90 days</b> agents handle the most interactions. All tenure groups achieve a
        CSAT-5 rate of ~<b>68–69%</b> — demonstrating strong, consistent onboarding and
        service delivery.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Charts 6 & 7: Shift + Monthly ────────────────────────────────────
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="section-badge">Chart 6</span>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Agent Shift Distribution</p>', unsafe_allow_html=True)
            fig6 = go.Figure(go.Bar(
                x=shift_data["Shift"], y=shift_data["Total"],
                marker_color=["#60A5FA","#22D3EE",C_GREEN,C_AMBER,C_CORAL],
                text=shift_data["Total"], textposition="outside",
            ))
            fig6.update_layout(title="Interactions per Shift",
                               xaxis_title="Shift", yaxis_title="Count")
            st.plotly_chart(styled_fig(fig6, height=320), use_container_width=True)
            st.markdown("""<div class="insight"><b>Morning</b> and <b>Evening</b> shifts handle
            <b>87%</b> of all interactions. All shifts achieve majority CSAT-5 scores —
            agents perform consistently across the clock.</div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="section-badge">Chart 7</span>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">Monthly Issue Volume</p>', unsafe_allow_html=True)
            fig7 = go.Figure(go.Bar(
                x=month_data["Month"], y=month_data["Share"],
                marker_color=[C_CORAL if m == "Aug" else "#60A5FA" for m in month_data["Month"]],
                text=[f"{s:.1f}%" for s in month_data["Share"]], textposition="outside",
            ))
            fig7.update_layout(title="% of Issues Reported per Month",
                               xaxis_title="Month", yaxis_title="Share (%)")
            st.plotly_chart(styled_fig(fig7, height=320), use_container_width=True)
            st.markdown("""<div class="insight"><b>August</b> is a massive peak with
            <b>66%</b> of all issues — requiring maximum staffing, escalation readiness,
            and proactive customer communication during this period.</div>""",
            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart 8: Manager Performance ─────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge">Chart 8</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Manager-Level Performance</p>', unsafe_allow_html=True)
        mgr   = manager_data
        rate5 = [round(c/t*100,1) for c,t in zip(mgr["CSAT5"], mgr["Total"])]
        fig8  = make_subplots(specs=[[{"secondary_y":True}]])
        fig8.add_trace(go.Bar(name="Total Interactions", x=mgr["Manager"],
                              y=mgr["Total"], marker_color="#60A5FA", opacity=0.65), secondary_y=False)
        fig8.add_trace(go.Bar(name="CSAT-5 Count", x=mgr["Manager"],
                              y=mgr["CSAT5"], marker_color=C_GREEN), secondary_y=False)
        fig8.add_trace(go.Scatter(name="CSAT-5 Rate %", x=mgr["Manager"], y=rate5,
                                  mode="lines+markers+text",
                                  text=[f"{r}%" for r in rate5], textposition="top center",
                                  line=dict(color=C_GOLD, width=2.5),
                                  marker=dict(size=9, color=C_GOLD2)), secondary_y=True)
        fig8.update_layout(
            barmode="overlay", title="Manager Interactions & CSAT-5 Rate", height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111111",
            font=dict(family="Inter", color=C_TEXT),
            title_font=dict(family="Cinzel", color=C_GOLD2, size=14),
            margin=dict(l=10,r=10,t=48,b=10),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)",
                        font=dict(color=C_TEXT)),
            xaxis=dict(gridcolor="#2A2A2A", linecolor="#2A2A2A",
                       tickfont=dict(color=C_SUB), title_font=dict(color=C_SUB)),
        )
        fig8.update_yaxes(title_text="Count",          secondary_y=False,
                          gridcolor="#2A2A2A", tickfont=dict(color=C_SUB))
        fig8.update_yaxes(title_text="CSAT-5 Rate (%)",secondary_y=True,
                          range=[60,80], gridcolor="rgba(0,0,0,0)", tickfont=dict(color=C_SUB))
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("""<div class="insight"><b>Key Insight:</b> <b>John Smith</b> leads with
        23,632 interactions. All 6 managers maintain a CSAT-5 rate of <b>~69%</b> —
        reflecting a uniformly high-quality leadership team driving consistent
        customer satisfaction across the organisation.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
      <strong>DeepCSAT</strong> · Internal Intelligence Platform ·
      Powered by PyTorch &amp; Streamlit · Shopzilla Customer Experience Division
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()