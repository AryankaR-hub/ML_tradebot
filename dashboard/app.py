import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH  = BASE_DIR / "data"

# ── Page config (must be FIRST st call) ───────────────────────
st.set_page_config(
    page_title="TradeBot",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ────────────────────────────────────────────────────
BG    = "#0C0C0C"
SURF  = "#141414"
SURF2 = "#1A1A1A"
LINE  = "#272727"
TEXT  = "#E8E3DC"
MUTED = "#5A5650"
DIM   = "#3A3733"
GOLD  = "#C9A84C"
GREEN = "#5DB87A"
RED   = "#C96B6B"
AMBER = "#C9943A"
WHITE = "#F5F0E8"

SENT_PALETTE = {
    "Extreme Fear":  RED,
    "Fear":          AMBER,
    "Neutral":       MUTED,
    "Greed":         GREEN,
    "Extreme Greed": GOLD,
}
SENT_ORDER   = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENT_VAL_MAP = {
    "Extreme Fear": 8, "Fear": 25, "Neutral": 50,
    "Greed": 75, "Extreme Greed": 92,
}

def rgba(hex6, a):
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ── CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background: {BG};
    color: {TEXT};
    -webkit-font-smoothing: antialiased;
}}
.stApp {{ background: {BG}; }}

[data-testid="stSidebar"] {{
    background: {SURF} !important;
    border-right: 1px solid {LINE} !important;
}}
[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {{
    background: {SURF2} !important;
    border-color: {LINE} !important;
    border-radius: 6px !important;
    color: {TEXT} !important;
}}
[data-testid="metric-container"] {{
    background: {SURF};
    border: 1px solid {LINE};
    border-radius: 8px;
    padding: 16px !important;
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.4rem !important;
    font-weight: 500 !important;
    color: {WHITE} !important;
}}
h1, h2, h3, h4 {{
    font-family: 'DM Sans', sans-serif !important;
    color: {WHITE} !important;
    font-weight: 500 !important;
}}
hr {{ border: none; border-top: 1px solid {LINE}; margin: 2rem 0; }}
.eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
}}
.section-title {{
    font-size: 1.1rem;
    font-weight: 500;
    color: {WHITE};
    margin-bottom: 18px;
}}
.ccard {{
    background: {SURF};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 4px 2px 0;
    margin-bottom: 6px;
}}
.finding {{
    display: grid;
    grid-template-columns: 28px 1fr;
    gap: 14px;
    padding: 13px 0;
    border-bottom: 1px solid {LINE};
    font-size: 0.875rem;
    line-height: 1.65;
    color: {TEXT};
}}
.finding:last-child {{ border-bottom: none; }}
.finding-n {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: {GOLD};
    padding-top: 4px;
}}
.finding b {{ color: {WHITE}; font-weight: 500; }}
.pred-win {{
    background: {rgba(GREEN, 0.07)};
    border: 1px solid {rgba(GREEN, 0.25)};
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 10px;
}}
.pred-loss {{
    background: {rgba(RED, 0.07)};
    border: 1px solid {rgba(RED, 0.25)};
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 10px;
}}
.pred-verdict {{ font-weight: 500; font-size: 1rem; margin-bottom: 2px; }}
.pred-pct {{
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: {MUTED};
    letter-spacing: 0.1em;
}}
.conf-track {{
    background: {LINE};
    border-radius: 2px;
    height: 3px;
    margin-top: 10px;
    overflow: hidden;
}}
.sb-label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
    display: block;
}}
.sb-stat {{
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: {MUTED};
    letter-spacing: 0.08em;
    padding: 5px 0;
    border-bottom: 1px solid {DIM};
}}
.sb-stat:last-child {{ border-bottom: none; }}
.sb-stat-val {{ color: {TEXT}; }}
.stButton > button {{
    width: 100%;
    background: {GOLD} !important;
    color: #0C0C0C !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 10px !important;
    transition: opacity .15s;
}}
.stButton > button:hover {{ opacity: 0.82 !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
::-webkit-scrollbar {{ width: 3px; }}
::-webkit-scrollbar-thumb {{ background: {LINE}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ──────────────────────────────────────────────
def apply_theme(fig, h=340, title=""):
    fig.update_layout(
        height=h,
        title=dict(
            text=title,
            font=dict(family="DM Mono", size=9, color=MUTED),
            x=0.015, y=0.985,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color=TEXT, size=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(family="DM Mono, monospace", size=9)),
        xaxis=dict(gridcolor=DIM, linecolor=LINE, zeroline=False,
                   tickfont=dict(family="DM Mono, monospace", size=9, color=MUTED)),
        yaxis=dict(gridcolor=DIM, linecolor=LINE, zeroline=False,
                   tickfont=dict(family="DM Mono, monospace", size=9, color=MUTED)),
        margin=dict(l=6, r=6, t=32, b=6),
        hoverlabel=dict(bgcolor=SURF2, bordercolor=LINE,
                        font=dict(family="DM Mono, monospace", size=10, color=TEXT)),
    )
    return fig

def chart_card(fig):
    st.markdown("<div class='ccard'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Load model safely ──────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        m = joblib.load(MODEL_PATH / "xgboost_model.pkl")
        e = joblib.load(MODEL_PATH / "encoders.pkl")
        return m, e
    except Exception:
        return None, None

model, encoders = load_model()


# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    trader_df    = pd.read_csv(DATA_PATH / "historical_data.csv")
    sentiment_df = pd.read_csv(DATA_PATH / "fear_greed.csv")
    trader_df["Timestamp"] = pd.to_datetime(trader_df["Timestamp"], unit="ms")
    trader_df["Date"]      = trader_df["Timestamp"].dt.date
    sentiment_df["date"]   = pd.to_datetime(sentiment_df["date"]).dt.date
    merged = pd.merge(
        trader_df, sentiment_df,
        left_on="Date", right_on="date", how="inner",
    )
    merged["Win"] = (merged["Closed PnL"] > 0).astype(int)
    return merged

df = load_data()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"""
    <div style="padding:20px 0 16px;">
        <div style="font-family:'DM Sans',sans-serif;font-weight:500;
                    font-size:1rem;color:{WHITE};">
            TradeBot
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                    color:{MUTED};letter-spacing:.14em;margin-top:4px;
                    text-transform:uppercase;">
            Hyperliquid · XGBoost
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<span class='sb-label'>Sentiment filter</span>", unsafe_allow_html=True)
    all_sent = sorted(df["classification"].dropna().unique())
    selected = st.multiselect("_sent", all_sent, default=all_sent,
                              label_visibility="collapsed")

    st.markdown(f"<span class='sb-label' style='margin-top:10px;display:block;'>Leaderboard top N</span>",
                unsafe_allow_html=True)
    top_n = st.slider("_topn", 5, 20, 10, label_visibility="collapsed")

    st.markdown("---")

    st.markdown(f"""
    <div style="font-weight:500;font-size:0.88rem;color:{WHITE};margin-bottom:12px;">
        Predict a Trade
    </div>
    """, unsafe_allow_html=True)

    coins    = sorted(df["Coin"].dropna().unique())
    p_coin   = st.selectbox("Asset", coins)
    p_side   = st.selectbox("Side", ["BUY", "SELL"])
    p_exec   = st.number_input("Exec price ($)", min_value=0.0, value=100.0, format="%.4f")
    p_sz_tok = st.number_input("Size (tokens)",  min_value=0.0, value=10.0,  format="%.4f")
    p_sz_usd = st.number_input("Size (USD)",     min_value=0.0, value=1000.0, format="%.2f")
    p_start  = st.number_input("Start position", value=100.0, format="%.2f")
    p_crossed = st.selectbox("Crossed?", [True, False])
    p_fee    = st.number_input("Fee ($)", min_value=0.0, value=1.0, format="%.4f")
    p_sent   = st.selectbox("Sentiment", SENT_ORDER)

    predict_btn = st.button("Run Prediction")

    if predict_btn:
        if model is None or encoders is None:
            st.error("Model not found. Run model.py first to generate the .pkl files.")
        else:
            try:
                coin_enc = encoders["Coin"].transform([p_coin])[0]
                side_enc = encoders["Side"].transform([p_side])[0]
                cls_key  = "classification" if "classification" in encoders else None
                cls_enc  = encoders[cls_key].transform([p_sent])[0] if cls_key else 0

                inp = pd.DataFrame({
                    "Coin":            [coin_enc],
                    "Execution Price": [p_exec],
                    "Size Tokens":     [p_sz_tok],
                    "Size USD":        [p_sz_usd],
                    "Side":            [side_enc],
                    "Start Position":  [p_start],
                    "Crossed":         [int(p_crossed)],
                    "Fee":             [p_fee],
                    "timestamp":       [1730007000],
                    "value":           [SENT_VAL_MAP[p_sent]],
                    "classification":  [cls_enc],
                })

                pred  = model.predict(inp)[0]
                proba = model.predict_proba(inp)[0]
                win_p = proba[1] * 100
                los_p = proba[0] * 100

                if pred == 1:
                    box_cls  = "pred-win"
                    color    = GREEN
                    verdict  = "Likely Profitable"
                    conf_txt = f"WIN PROB  {win_p:.1f}%"
                    bar_w    = win_p
                else:
                    box_cls  = "pred-loss"
                    color    = RED
                    verdict  = "Likely Loss"
                    conf_txt = f"LOSS PROB  {los_p:.1f}%"
                    bar_w    = los_p

                st.markdown(f"""
                <div class="{box_cls}">
                    <div class="pred-verdict" style="color:{color};">{verdict}</div>
                    <div class="pred-pct">{conf_txt}</div>
                    <div class="conf-track">
                        <div style="background:{color};height:3px;border-radius:2px;
                                    width:{bar_w:.0f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")

    for k, v in [("Model","XGBoost"), ("Accuracy","96%"),
                 ("Features","11"), ("Records","184k+")]:
        st.markdown(
            f"<div class='sb-stat'><span>{k}</span>"
            f"<span class='sb-stat-val'>{v}</span></div>",
            unsafe_allow_html=True,
        )


# ── Filter data ────────────────────────────────────────────────
fdf = df[df["classification"].isin(selected)] if selected else df.copy()


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding:32px 0 4px;">
    <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                letter-spacing:.2em;text-transform:uppercase;
                color:{MUTED};margin-bottom:10px;">
        Hyperliquid · Fear &amp; Greed Overlay
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-weight:300;
                font-size:2.2rem;line-height:1.15;
                color:{WHITE};letter-spacing:-0.03em;">
        Market Intelligence<br>
        <span style="color:{GOLD};">Dashboard</span>
    </div>
    <div style="margin-top:8px;font-size:0.82rem;color:{MUTED};font-weight:300;">
        184k+ trades · XGBoost 96% · sentiment-segmented analytics
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  KPIs
# ══════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trades",    f"{len(fdf):,}")
c2.metric("Total PnL", f"${fdf['Closed PnL'].sum():,.0f}")
c3.metric("Avg PnL",   f"${fdf['Closed PnL'].mean():.2f}")
c4.metric("Win Rate",  f"{fdf['Win'].mean()*100:.1f}%")
c5.metric("Avg Size",  f"${fdf['Size USD'].mean():,.0f}")
c6.metric("Volume",    f"${fdf['Size USD'].sum()/1e6:.1f}M")

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SENTIMENT
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='eyebrow'>Sentiment</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>How Market Mood Shapes Outcomes</div>",
            unsafe_allow_html=True)

present = [s for s in SENT_ORDER if s in fdf["classification"].unique()]
left, right = st.columns(2)

pnl_s = (
    fdf.groupby("classification")["Closed PnL"]
    .mean().reindex(present).reset_index()
)
fig_pnl = go.Figure(go.Bar(
    x=pnl_s["classification"],
    y=pnl_s["Closed PnL"],
    marker=dict(
        color=[SENT_PALETTE.get(s, MUTED) for s in pnl_s["classification"]],
        line=dict(width=0),
    ),
    text=[f"${v:.0f}" for v in pnl_s["Closed PnL"]],
    textposition="outside",
    textfont=dict(family="DM Mono, monospace", size=9, color=MUTED),
    hovertemplate="<b>%{x}</b><br>Avg PnL: $%{y:.2f}<extra></extra>",
))
fig_pnl.update_layout(showlegend=False, bargap=0.45)
apply_theme(fig_pnl, h=300, title="avg pnl per trade")
with left:
    chart_card(fig_pnl)

wr_s = (
    fdf.groupby("classification")["Win"]
    .mean().mul(100).reindex(present).reset_index()
)
fig_wr = go.Figure(go.Bar(
    x=wr_s["classification"],
    y=wr_s["Win"],
    marker=dict(
        color=[SENT_PALETTE.get(s, MUTED) for s in wr_s["classification"]],
        line=dict(width=0),
    ),
    text=[f"{v:.1f}%" for v in wr_s["Win"]],
    textposition="outside",
    textfont=dict(family="DM Mono, monospace", size=9, color=MUTED),
    hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
))
fig_wr.update_layout(showlegend=False, bargap=0.45)
fig_wr.update_yaxes(ticksuffix="%")
apply_theme(fig_wr, h=300, title="win rate (%)")
with right:
    chart_card(fig_wr)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PnL DISTRIBUTION + AVG TRADE SIZE
# ══════════════════════════════════════════════════════════════
left2, right2 = st.columns([1.2, 0.8])

pnl_clip = fdf[fdf["Closed PnL"].between(
    fdf["Closed PnL"].quantile(0.01),
    fdf["Closed PnL"].quantile(0.99),
)].copy()

fig_box = go.Figure()
for s in SENT_ORDER:
    grp = pnl_clip[pnl_clip["classification"] == s]
    if grp.empty:
        continue
    col = SENT_PALETTE.get(s, MUTED)
    fig_box.add_trace(go.Box(
        y=grp["Closed PnL"],
        name=s,
        marker_color=col,
        line_color=col,
        fillcolor=rgba(col, 0.10),
        boxmean="sd",
        hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
    ))
apply_theme(fig_box, h=360, title="pnl spread by sentiment (p1-p99)")
with left2:
    chart_card(fig_box)

sz_s = fdf.groupby("classification")["Size USD"].mean().reindex(present).reset_index()
fig_sz = go.Figure(go.Bar(
    x=sz_s["classification"],
    y=sz_s["Size USD"],
    marker=dict(
        color=[SENT_PALETTE.get(s, MUTED) for s in sz_s["classification"]],
        line=dict(width=0),
    ),
    text=[f"${v:,.0f}" for v in sz_s["Size USD"]],
    textposition="outside",
    textfont=dict(family="DM Mono, monospace", size=9, color=MUTED),
    hovertemplate="<b>%{x}</b><br>Avg size: $%{y:,.0f}<extra></extra>",
))
fig_sz.update_layout(showlegend=False, bargap=0.4)
fig_sz.update_yaxes(tickprefix="$")
apply_theme(fig_sz, h=360, title="avg trade size by sentiment")
with right2:
    chart_card(fig_sz)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LEADERBOARD
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='eyebrow'>Leaderboard</div>", unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>Top {top_n} Wallets by Realised PnL</div>",
            unsafe_allow_html=True)

top_wallets = (
    fdf.groupby("Account")
    .agg(PnL=("Closed PnL","sum"), Trades=("Closed PnL","count"), WR=("Win","mean"))
    .reset_index()
    .sort_values("PnL", ascending=False)
    .head(top_n)
)
top_wallets["WR"]    = (top_wallets["WR"] * 100).round(1)
top_wallets["Label"] = (top_wallets["Account"].str[:6]
                        + "..." + top_wallets["Account"].str[-4:])

fig_lb = go.Figure(go.Bar(
    x=top_wallets["PnL"],
    y=top_wallets["Label"],
    orientation="h",
    marker=dict(
        color=[GOLD if v >= 0 else RED for v in top_wallets["PnL"]],
        line=dict(width=0),
    ),
    text=[f"  ${v/1e3:.1f}k  |  {w}% WR"
          for v, w in zip(top_wallets["PnL"], top_wallets["WR"])],
    textposition="outside",
    textfont=dict(family="DM Mono, monospace", size=9, color=MUTED),
    customdata=top_wallets[["Trades","WR"]].values,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "PnL: $%{x:,.0f}<br>"
        "Trades: %{customdata[0]}<br>"
        "Win Rate: %{customdata[1]}%<extra></extra>"
    ),
))
fig_lb.update_xaxes(tickprefix="$")
apply_theme(fig_lb, h=max(300, top_n * 34), title="total realised pnl")
chart_card(fig_lb)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='eyebrow'>Model</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>What the Model Relies On</div>",
            unsafe_allow_html=True)

feat_data = [
    ("Start Position",  0.4111),
    ("Side",            0.3798),
    ("Execution Price", 0.0357),
    ("Crossed",         0.0345),
    ("value",           0.0343),
    ("Coin",            0.0291),
    ("timestamp",       0.0288),
    ("classification",  0.0216),
    ("Fee",             0.0094),
    ("Size Tokens",     0.0082),
    ("Size USD",        0.0074),
]
fi_df = (pd.DataFrame(feat_data, columns=["Feature", "Importance"])
         .sort_values("Importance"))

bar_colors = [
    GOLD if v >= 0.15 else (TEXT if v >= 0.03 else MUTED)
    for v in fi_df["Importance"]
]
fig_fi = go.Figure(go.Bar(
    x=fi_df["Importance"],
    y=fi_df["Feature"],
    orientation="h",
    marker=dict(color=bar_colors, line=dict(width=0)),
    text=[f"{v:.3f}" for v in fi_df["Importance"]],
    textposition="outside",
    textfont=dict(family="DM Mono, monospace", size=9, color=MUTED),
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
))
fig_fi.update_layout(showlegend=False)
apply_theme(fig_fi, h=380, title="xgboost feature importance")
chart_card(fig_fi)

st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  KEY FINDINGS
# ══════════════════════════════════════════════════════════════
st.markdown("<div class='eyebrow'>Findings</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>What the Data Says</div>",
            unsafe_allow_html=True)

findings = [
    ("01", "<b>Greed conditions</b> produce the highest average PnL. Extreme Greed "
           "has the most consistent win rate -- momentum strategies thrive when the "
           "crowd is already bullish."),
    ("02", "<b>Direction was dropped</b> from the model with zero accuracy loss. "
           "96% holds across train, test, and out-of-sample -- clean generalisation, "
           "no target leakage."),
    ("03", "<b>Fear and Extreme Greed</b> both correlate with larger average position "
           "sizes. Risk-appetite spikes at sentiment extremes, not just in bull conditions."),
    ("04", "PnL follows a <b>Pareto distribution</b> -- the top 10 wallets account "
           "for a disproportionate share. Most participants are small and breakeven "
           "or negative."),
    ("05", "<b>Start Position (41%) and Side (38%)</b> together explain ~79% of model "
           "signal. Entry context and trade direction are the only things that really matter."),
]
for n, body in findings:
    st.markdown(
        f"<div class='finding'>"
        f"<div class='finding-n'>{n}</div>"
        f"<div>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Footer ─────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;font-family:'DM Mono',monospace;font-size:0.57rem;
            color:{MUTED};letter-spacing:.14em;padding:4px 0 20px;
            text-transform:uppercase;">
    TradeBot · Streamlit + XGBoost · 96% accuracy · 184k+ trades
</div>
""", unsafe_allow_html=True)