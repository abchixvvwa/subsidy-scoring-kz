"""
КазнаЛинза — Streamlit-дашборд
==============================
Run:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="КазнаЛинза · KZ-2025",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT           = Path(__file__).resolve().parent.parent
ML_RESULTS_PATH = _ROOT / "data" / "processed" / "ml_results.csv"
FEATURES_PATH   = _ROOT / "data" / "processed" / "features.csv"
MODEL_PATH      = _ROOT / "data" / "processed" / "lgbm_model.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Design-system global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: #F5F5F7;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E5EA !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

[data-testid="metric-container"] {
    background: #FFFFFF;
    border: none;
    border-radius: 18px;
    padding: 24px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

[data-testid="metric-container"] label {
    color: #86868B !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    letter-spacing: -0.1px !important;
    text-transform: none !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMetricValue"] {
    color: #1D1D1F !important;
    font-size: 34px !important;
    font-weight: 600 !important;
    letter-spacing: -1px !important;
    font-family: 'Inter', sans-serif !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E5E5EA !important;
    gap: 0 !important;
    padding: 0 32px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #86868B !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 14px 20px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}

.stTabs [aria-selected="true"] {
    color: #1D1D1F !important;
    border-bottom-color: #1D1D1F !important;
    background: transparent !important;
    font-weight: 500 !important;
}

[data-testid="stDataFrame"] {
    border: none !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 12px !important;
    color: #1D1D1F !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #0071E3 !important;
    box-shadow: 0 0 0 4px rgba(0,113,227,0.15) !important;
}

.stButton button {
    background: #0071E3 !important;
    color: white !important;
    border: none !important;
    border-radius: 980px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
}

.stButton button:hover {
    background: #0077ED !important;
    box-shadow: 0 4px 16px rgba(0,113,227,0.3) !important;
}

[data-testid="stSlider"] > div > div > div {
    background: #0071E3 !important;
}

section[data-testid="stSidebar"] .stMultiSelect > div,
section[data-testid="stSidebar"] .stSelectbox > div {
    background: #F5F5F7 !important;
    border: 1px solid #E5E5EA !important;
    border-radius: 10px !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { 
    background: rgba(0,0,0,0.15); 
    border-radius: 3px; 
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

_CLUSTER_COLOR_MAP = {
    "Крупные эффективные": "#0071E3",
    "Малые активные":      "#34C759",
    "Средние стабильные":  "#FF9F0A",
    "Малые рискованные":   "#FF3B30",
}
_FALLBACK_COLORS = ["#0071E3", "#34C759", "#FF9F0A", "#FF3B30", "#AF52DE"]

_PLOTLY_BASE = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#1D1D1F", size=12),
    margin=dict(l=10, r=20, t=24, b=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_ml_results() -> pd.DataFrame | None:
    if not ML_RESULTS_PATH.exists():
        return None
    return pd.read_csv(ML_RESULTS_PATH)


@st.cache_data(ttl=120)
def load_features() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        return None
    return pd.read_csv(FEATURES_PATH)


@st.cache_data(ttl=120)
def load_lgbm_importances() -> pd.DataFrame | None:
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        sys.path.insert(0, str(_ROOT))
        from src.ml_model import LGBM_FEATURES  # type: ignore
        return (
            pd.DataFrame({"Feature": LGBM_FEATURES, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=True)
            .reset_index(drop=True)
        )
    except Exception:
        return None


ml_df = load_ml_results()

# ─────────────────────────────────────────────────────────────────────────────
# No-data state
# ─────────────────────────────────────────────────────────────────────────────

if ml_df is None:
    st.markdown(
        """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
min-height:80vh;gap:20px;padding:40px;">
  <div style="width:64px;height:64px;background:linear-gradient(135deg,#0071E3 0%,#AF52DE 100%);
  border-radius:16px;display:flex;align-items:center;justify-content:center;
  font-size:28px;color:white;font-weight:700;font-family:'Inter',sans-serif;">S</div>
  <div style="font-family:'Inter',sans-serif;font-size:20px;font-weight:600;
  color:#1D1D1F;text-align:center;">Данные не загружены</div>
  <div style="font-family:'Inter',sans-serif;font-size:13px;
  color:#86868B;text-align:center;">Запустите ML-пайплайн для генерации результатов</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Запустить ML Pipeline"):
        with st.spinner("Запуск пайплайна..."):
            result = subprocess.run(
                [sys.executable, str(_ROOT / "src" / "ml_model.py")],
                capture_output=True, text=True, cwd=str(_ROOT),
            )
        if result.returncode == 0:
            st.success("Пайплайн завершён. Обновите страницу.")
            st.code(result.stdout[-3000:])
        else:
            st.error("Пайплайн завершился с ошибкой.")
            st.code(result.stderr[-2000:])
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Ensure required columns
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(_ROOT))

if "ml_probability" not in ml_df.columns:
    ml_df["ml_probability"] = 0.5
if "ml_rank" not in ml_df.columns:
    ml_df["ml_rank"] = (
        ml_df["ml_probability"].rank(ascending=False, method="first").astype(int)
    )

# Load composite scores
try:
    from src.scoring import compute_scores, explain_score  # type: ignore
    features_df = load_features()
    scored_df   = compute_scores(features_df) if features_df is not None else None
except Exception:
    scored_df   = None
    features_df = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo block
    st.markdown(
        """
<div style="padding:24px 20px 20px;border-bottom:1px solid #E5E5EA;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <div style="width:32px;height:32px;background:linear-gradient(135deg,#0071E3 0%,#AF52DE 100%);
    border-radius:8px;display:flex;align-items:center;justify-content:center;
    font-size:14px;color:white;font-weight:700;font-family:'Inter',sans-serif;flex-shrink:0;">К</div>
    <span style="font-family:'Inter',sans-serif;font-size:15px;font-weight:600;
    color:#1D1D1F;letter-spacing:-0.3px;">КазнаЛинза</span>
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:11px;
  color:#86868B;letter-spacing:0.3px;">DECENTRATHON 5.0 · KZ-2025</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Filters section header
    st.markdown(
        """
<div style="padding:20px 20px 12px;">
  <div style="font-family:'Inter',sans-serif;font-size:11px;
  color:#86868B;letter-spacing:1px;font-weight:500;">ФИЛЬТРЫ</div>
</div>
""",
        unsafe_allow_html=True,
    )

    all_oblasts    = sorted(ml_df["oblast"].dropna().unique().tolist())
    all_directions = sorted(ml_df["primary_direction"].dropna().unique().tolist())

    sel_oblasts = st.multiselect(
        "Регион",
        options=all_oblasts,
        default=[],
        placeholder="Все регионы",
    )
    sel_directions = st.multiselect(
        "Направление",
        options=all_directions,
        default=[],
        placeholder="Все направления",
    )
    min_score = st.slider("Мин. ML скор", 0.0, 1.0, 0.0, step=0.05)

    # System status — pinned at bottom via padding spacer
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div style="padding:16px 20px;border-top:1px solid #E5E5EA;
background:#FFFFFF;margin:0 -1px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
    <div style="width:6px;height:6px;border-radius:50%;background:#34C759;
    box-shadow:0 0 8px rgba(52,199,89,0.4);flex-shrink:0;"></div>
    <span style="font-family:'Inter',sans-serif;font-size:11px;
    color:#34C759;font-weight:500;">СИСТЕМА АКТИВНА</span>
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:11px;color:#86868B;line-height:1.8;">
    LightGBM · KMeans · IsoForest<br>
    36 651 записей · 457 заявителей
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sel_oblasts:
        out = out[out["oblast"].isin(sel_oblasts)]
    if sel_directions:
        out = out[out["primary_direction"].isin(sel_directions)]
    out = out[out["ml_probability"] >= min_score]
    return out


fdf = apply_filters(ml_df)

# ─────────────────────────────────────────────────────────────────────────────
# Header — metrics + tabs
# ─────────────────────────────────────────────────────────────────────────────

# Compute metric values
total_count = len(fdf)
avg_ml      = fdf["ml_probability"].mean() if total_count else 0.0
anom_count  = int(fdf["is_anomaly"].sum()) if "is_anomaly" in fdf.columns else 0

top_region = "—"
if total_count and "oblast" in fdf.columns and "total_amount_received" in fdf.columns:
    try:
        top_region = fdf.groupby("oblast")["total_amount_received"].sum().idxmax()
    except Exception:
        pass

st.markdown(
    """
<div style="padding:24px 32px 20px;background:#F5F5F7;">
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Заявителей", f"{total_count:,}")
with m2:
    st.metric("Ср. ML скор", f"{avg_ml:.2f}")
with m3:
    st.metric("Аномалий", anom_count, delta=f"из {total_count}")
with m4:
    lbl = str(top_region)
    st.metric("Топ регион", lbl[:22] + "…" if len(lbl) > 22 else lbl)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "01 / РЕЙТИНГ",
    "02 / ЗАЯВИТЕЛЬ",
    "03 / СЕГМЕНТЫ",
    "04 / АНОМАЛИИ",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — РЕЙТИНГ
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(
        """
<div style="padding:24px 32px 16px;">
  <div style="font-family:'Inter',sans-serif;font-size:11px;
  color:#86868B;letter-spacing:1px;margin-bottom:4px;font-weight:500;">
    AI-РАНЖИРОВАНИЕ · LIGHTGBM + COMPOSITE</div>
  <div style="font-family:'Inter',sans-serif;font-size:22px;
  font-weight:600;color:#1D1D1F;letter-spacing:-0.5px;">
    Рейтинг заявителей</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)
        top_n = st.slider("Top N заявителей", 10, 50, 20, 5, key="tab1_topn")
        st.markdown("</div>", unsafe_allow_html=True)

    # Build ranked dataframe
    rank_df = fdf.copy()
    if scored_df is not None:
        rank_df = rank_df.merge(
            scored_df[["applicant_id", "final_score"]],
            on="applicant_id", how="left",
        )
    else:
        rank_df["final_score"] = np.nan

    rank_df = rank_df.sort_values("ml_probability", ascending=False).head(top_n).reset_index(drop=True)

    # Select & configure columns
    col_order = ["applicant_id", "ml_probability", "final_score", "approval_rate",
                 "cluster_label", "is_anomaly", "oblast"]
    avail_cols = [c for c in col_order if c in rank_df.columns]
    tbl1 = rank_df[avail_cols].copy()

    col_cfg1: dict = {}
    if "ml_probability" in tbl1.columns:
        col_cfg1["ml_probability"] = st.column_config.ProgressColumn(
            "ML Score", format="%.2f", min_value=0, max_value=1
        )
    if "final_score" in tbl1.columns:
        col_cfg1["final_score"] = st.column_config.ProgressColumn(
            "Composite", format="%.2f", min_value=0, max_value=1
        )
    if "approval_rate" in tbl1.columns:
        col_cfg1["approval_rate"] = st.column_config.ProgressColumn(
            "Одобрений", format="%.2f", min_value=0, max_value=1
        )
    if "cluster_label" in tbl1.columns:
        col_cfg1["cluster_label"] = st.column_config.TextColumn("Кластер")
    if "is_anomaly" in tbl1.columns:
        col_cfg1["is_anomaly"] = st.column_config.CheckboxColumn("⚠")
    if "oblast" in tbl1.columns:
        col_cfg1["oblast"] = st.column_config.TextColumn("Регион")

    st.markdown("<div style='padding:0 32px 20px'>", unsafe_allow_html=True)
    st.dataframe(tbl1, column_config=col_cfg1, hide_index=True,
                 use_container_width=True, height=400)
    st.markdown("</div>", unsafe_allow_html=True)

    # Horizontal bar chart
    chart_df = rank_df.sort_values("ml_probability", ascending=True).copy()
    chart_df["applicant_id"] = chart_df["applicant_id"].astype(str)

    # Build cluster color sequence matching values present
    cluster_col = "cluster_label" if "cluster_label" in chart_df.columns else None
    unique_clusters = chart_df[cluster_col].unique().tolist() if cluster_col else []
    color_map = {
        c: _CLUSTER_COLOR_MAP.get(c, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        for i, c in enumerate(unique_clusters)
    }

    fig1 = px.bar(
        chart_df,
        x="ml_probability",
        y="applicant_id",
        orientation="h",
        color=cluster_col,
        color_discrete_map=color_map if cluster_col else None,
        color_discrete_sequence=_FALLBACK_COLORS if not cluster_col else None,
    )
    fig1.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(family='Inter, sans-serif', color='#1D1D1F', size=12),
        height=420,
        xaxis=dict(
            gridcolor='#F5F5F7',
            linecolor='#E5E5EA',
            title="ML вероятность одобрения",
            title_font=dict(color='#86868B', size=11),
            tickfont=dict(color='#86868B', size=11),
            showgrid=True,
            zeroline=False,
            range=[0, 1.05],
        ),
        yaxis=dict(
            gridcolor='#F5F5F7',
            linecolor='#E5E5EA',
            title="",
            title_font=dict(color='#86868B', size=11),
            tickfont=dict(color='#86868B', size=11),
            showgrid=False,
            zeroline=False,
            autorange="reversed",
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(color='#86868B', size=11),
        ),
        margin=dict(l=10, r=20, t=24, b=10),
        bargap=0.35,
    )
    fig1.update_traces(marker_line_width=0)

    st.markdown("<div style='padding:0 32px 32px'>", unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ЗАЯВИТЕЛЬ
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        """
<div style="padding:24px 32px 16px;">
  <div style="font-family:'Inter',sans-serif;font-size:11px;
  color:#86868B;letter-spacing:1px;margin-bottom:4px;font-weight:500;">ПРОФИЛЬ · МОДЕЛИ · АНОМАЛИЯ</div>
  <div style="font-family:'Inter',sans-serif;font-size:22px;font-weight:600;
  color:#1D1D1F;letter-spacing:-0.5px;">Анализ заявителя</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='padding:0 32px 16px'>", unsafe_allow_html=True)
    aid_raw = st.text_input(
        "ID заявителя",
        placeholder="Введите ID заявителя...",
        label_visibility="collapsed",
        key="applicant_id_input",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if aid_raw.strip():
        aid      = aid_raw.strip()
        ml_mask  = ml_df["applicant_id"].astype(str) == aid

        if not ml_mask.any():
            st.markdown(
                f"""
<div style="margin:0 32px;background:#FFFFFF;border:1px solid #F5F5F7;
border-radius:18px;padding:20px 24px;font-family:'Inter',sans-serif;
font-size:13px;color:#86868B;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
  Заявитель <span style="color:#1D1D1F;font-weight:500;">{aid}</span> не найден.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            ml_row   = ml_df[ml_mask].iloc[0]
            ml_prob  = float(ml_row.get("ml_probability", 0) or 0)
            ml_rnk   = int(ml_row.get("ml_rank", 0) or 0)
            cl_label = str(ml_row.get("cluster_label", "—"))
            is_anom  = bool(ml_row.get("is_anomaly", False))
            anom_txt = str(ml_row.get("anomaly_reason", ""))

            comp_score = None
            comp_rank  = None
            factors    = None
            total_rows = len(ml_df)

            if scored_df is not None and features_df is not None:
                sc_mask = scored_df["applicant_id"].astype(str) == aid
                if sc_mask.any():
                    sr         = scored_df[sc_mask].iloc[0]
                    comp_score = float(sr["final_score"])
                    comp_rank  = int(sr["rank"]) if "rank" in sr.index else None
                try:
                    expl    = explain_score(aid, features_df)
                    factors = expl.get("factors", {})
                except Exception:
                    pass

            discrepancy = abs(ml_prob - comp_score) if comp_score is not None else 0.0

            st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)
            left_col, mid_col, right_col = st.columns([2, 1.5, 1.5], gap="large")
            st.markdown("</div>", unsafe_allow_html=True)

            # ── LEFT: score cards + factors ───────────────────────────────
            with left_col:
                # Score cards grid
                comp_html = ""
                if comp_score is not None:
                    comp_html = f"""
  <div style="background:#FFFFFF;border:1px solid #F5F5F7;
  border-radius:18px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-family:'Inter',sans-serif;font-size:11px;
    color:#86868B;font-weight:500;margin-bottom:8px;">COMPOSITE</div>
    <div style="font-family:'Inter',sans-serif;font-size:42px;font-weight:600;
    color:#FF9F0A;letter-spacing:-2px;line-height:1;">{comp_score:.0%}</div>
    <div style="font-family:'Inter',sans-serif;font-size:11px;
    color:#86868B;margin-top:4px;">Ранг #{comp_rank if comp_rank else "?"} из {total_rows}</div>
  </div>"""
                else:
                    comp_html = """
  <div style="background:#FFFFFF;border:1px solid #F5F5F7;
  border-radius:18px;padding:20px;display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-family:'Inter',sans-serif;font-size:11px;color:#86868B;">
      Composite N/A</div>
  </div>"""

                st.markdown(
                    f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;">
  <div style="background:#FFFFFF;border:1px solid #F5F5F7;
  border-radius:18px;padding:20px;position:relative;overflow:hidden;
  box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#0071E3,#AF52DE);"></div>
    <div style="font-family:'Inter',sans-serif;font-size:11px;
    color:#86868B;font-weight:500;margin-bottom:8px;">LIGHTGBM</div>
    <div style="font-family:'Inter',sans-serif;font-size:42px;font-weight:600;
    color:#0071E3;letter-spacing:-2px;line-height:1;">{ml_prob:.0%}</div>
    <div style="font-family:'Inter',sans-serif;font-size:11px;
    color:#86868B;margin-top:4px;">Ранг #{ml_rnk} из {total_rows}</div>
  </div>
  {comp_html}
</div>
""",
                    unsafe_allow_html=True,
                )

                # Discrepancy badge
                if discrepancy > 0.3:
                    st.markdown(
                        f"""
<div style="background:#FFFBEB;border:1px solid #FDE68A;
border-radius:12px;padding:8px 12px;display:inline-flex;align-items:center;gap:6px;
margin-bottom:16px;">
  <div style="width:6px;height:6px;border-radius:50%;background:#FF9F0A;flex-shrink:0;"></div>
  <span style="font-family:'Inter',sans-serif;font-size:12px;color:#FF9F0A;font-weight:500;">
    Расхождение моделей: {discrepancy:.2f}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # Factor contribution bars
                if factors:
                    st.markdown(
                        """<div style="font-family:'Inter',sans-serif;font-size:11px;
color:#86868B;font-weight:500;margin-bottom:12px;">ФАКТОРЫ СКОРА</div>""",
                        unsafe_allow_html=True,
                    )
                    fdf_factors = pd.DataFrame([
                        {"Factor": v["label"], "Contribution": v["contribution"]}
                        for v in factors.values()
                    ]).sort_values("Contribution", ascending=True)

                    # Normalize opacity 0.3-1.0
                    max_c = fdf_factors["Contribution"].max() or 1.0
                    fdf_factors["opacity"] = (
                        fdf_factors["Contribution"] / max_c * 0.7 + 0.3
                    ).clip(0.3, 1.0)

                    fig_f = go.Figure(go.Bar(
                        x=fdf_factors["Contribution"],
                        y=fdf_factors["Factor"],
                        orientation="h",
                        marker=dict(
                            color="#0071E3",
                            opacity=fdf_factors["opacity"].tolist(),
                            line=dict(width=0),
                        ),
                    ))
                    fig_f.update_layout(
                        plot_bgcolor='#FFFFFF',
                        paper_bgcolor='#FFFFFF',
                        font=dict(family='Inter, sans-serif', color='#1D1D1F', size=12),
                        height=200,
                        xaxis=dict(
                            gridcolor='#F5F5F7',
                            linecolor='#E5E5EA',
                            title="Вклад",
                            title_font=dict(color='#86868B', size=11),
                            tickfont=dict(color='#86868B', size=11),
                            showgrid=True,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            gridcolor='#F5F5F7',
                            linecolor='#E5E5EA',
                            title_font=dict(color='#86868B', size=11),
                            tickfont=dict(color='#86868B', size=11),
                            showgrid=False,
                            zeroline=False,
                        ),
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            bordercolor='rgba(0,0,0,0)',
                            font=dict(color='#86868B', size=11),
                        ),
                        margin=dict(l=10, r=20, t=24, b=10),
                    )
                    st.plotly_chart(fig_f, use_container_width=True)

            # ── MID: profile + anomaly ────────────────────────────────────
            with mid_col:
                ob  = str(ml_row.get("oblast", "—"))
                prd = str(ml_row.get("primary_direction", "—"))
                tot = int(ml_row.get("total_applications", 0) or 0)
                apr = float(ml_row.get("approval_rate", 0) or 0)

                def _field(label: str, value: str) -> str:
                    return f"""
<div style="margin-bottom:12px;">
  <div style="font-size:9px;color:#4A5168;letter-spacing:1px;
  text-transform:uppercase;font-family:'JetBrains Mono',monospace;">{label}</div>
  <div style="font-size:14px;color:#F1F3F9;font-weight:500;
  font-family:'Syne',sans-serif;margin-top:2px;">{value}</div>
</div>"""

                profile_html = f"""
<div style="background:#0E1118;border:1px solid rgba(255,255,255,0.07);
border-radius:12px;padding:20px;margin-bottom:12px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
  color:#4A5168;letter-spacing:2px;margin-bottom:16px;">ПРОФИЛЬ</div>
  {_field("Регион", ob)}
  {_field("Направление", prd[:40] + "…" if len(prd) > 40 else prd)}
  {_field("Заявок", f"{tot:,}")}
  {_field("Одобрений", f"{apr:.0%}")}
  {_field("Кластер", cl_label)}
</div>"""

                st.markdown(profile_html, unsafe_allow_html=True)

                if is_anom:
                    st.markdown(
                        f"""
<div style="background:#FFF5F5;
border:1px solid #FFD7D5;border-radius:18px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#FF3B30;
    box-shadow:0 0 10px rgba(255,59,48,0.4);flex-shrink:0;"></div>
    <span style="font-family:'Inter',sans-serif;font-size:11px;
    color:#FF3B30;font-weight:600;letter-spacing:0.5px;">АНОМАЛИЯ ОБНАРУЖЕНА</span>
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:13px;
  color:#86868B;line-height:1.6;">{anom_txt}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
<div style="background:#F0FFF4;
border:1px solid #C6F6D5;border-radius:18px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#34C759;
    box-shadow:0 0 10px rgba(52,199,89,0.4);flex-shrink:0;"></div>
    <span style="font-family:'Inter',sans-serif;font-size:11px;
    color:#34C759;font-weight:600;">В НОРМЕ</span>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

            # ── RIGHT: LightGBM feature importance ────────────────────────
            with right_col:
                imp_df = load_lgbm_importances()
                if imp_df is not None:
                    st.markdown(
                        """<div style="font-family:'Inter',sans-serif;font-size:11px;
color:#86868B;font-weight:500;margin-bottom:12px;">FEATURE IMPORTANCE</div>""",
                        unsafe_allow_html=True,
                    )
                    fig_imp = px.bar(
                        imp_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale=["#E8F0FE", "#0071E3", "#003B8E"],
                    )
                    fig_imp.update_traces(marker_line_width=0)
                    fig_imp.update_layout(
                        plot_bgcolor='#FFFFFF',
                        paper_bgcolor='#FFFFFF',
                        font=dict(family='Inter, sans-serif', color='#1D1D1F', size=12),
                        height=380,
                        coloraxis_showscale=False,
                        xaxis=dict(
                            gridcolor='#F5F5F7',
                            linecolor='#E5E5EA',
                            title="Importance",
                            title_font=dict(color='#86868B', size=11),
                            tickfont=dict(color='#86868B', size=11),
                            showgrid=True,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            gridcolor='#F5F5F7',
                            linecolor='#E5E5EA',
                            title_font=dict(color='#86868B', size=11),
                            tickfont=dict(color='#86868B', size=11),
                            showgrid=False,
                            zeroline=False,
                        ),
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            bordercolor='rgba(0,0,0,0)',
                            font=dict(color='#86868B', size=11),
                        ),
                        margin=dict(l=10, r=20, t=24, b=10),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.markdown(
                        """<div style="background:#FFFFFF;border:1px solid #F5F5F7;
border-radius:18px;padding:20px;font-family:'Inter',sans-serif;font-size:12px;
color:#86868B;box-shadow:0 2px 8px rgba(0,0,0,0.06);">Модель не загружена</div>""",
                        unsafe_allow_html=True,
                    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — СЕГМЕНТЫ
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        """
<div style="padding:24px 32px 16px;">
  <div style="font-family:'Inter',sans-serif;font-size:11px;
  color:#86868B;letter-spacing:1px;margin-bottom:4px;font-weight:500;">КЛАСТЕРИЗАЦИЯ · KMEANS</div>
  <div style="font-family:'Inter',sans-serif;font-size:22px;font-weight:600;
  color:#1D1D1F;letter-spacing:-0.5px;">Сегменты заявителей</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if "cluster_label" not in fdf.columns:
        st.markdown(
            """<div style="margin:0 32px;background:#FFFFFF;border:1px solid #F5F5F7;
border-radius:18px;padding:20px;font-family:'Inter',sans-serif;font-size:13px;
color:#86868B;box-shadow:0 2px 8px rgba(0,0,0,0.06);">Данные по кластерам отсутствуют.</div>""",
            unsafe_allow_html=True,
        )
    else:
        sc_df = fdf.dropna(subset=["cluster_label"]).copy()

        # Build cluster color map from data
        unique_cls = sc_df["cluster_label"].unique().tolist()
        cmap3 = {
            c: _CLUSTER_COLOR_MAP.get(c, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
            for i, c in enumerate(unique_cls)
        }

        fig_sc = px.scatter(
            sc_df,
            x="approval_rate",
            y="total_amount_received",
            color="cluster_label",
            size="total_applications" if "total_applications" in sc_df.columns else None,
            size_max=40,
            hover_data=["applicant_id", "oblast", "ml_probability"],
            color_discrete_map=cmap3,
            labels={
                "approval_rate":         "Доля одобрений",
                "total_amount_received": "Сумма субсидий (тг)",
                "cluster_label":         "Кластер",
            },
        )
        fig_sc.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family='Inter, sans-serif', color='#1D1D1F', size=12),
            height=500,
            xaxis=dict(
                gridcolor='#F5F5F7',
                linecolor='#E5E5EA',
                title="Доля одобрений",
                title_font=dict(color='#86868B', size=11),
                tickfont=dict(color='#86868B', size=11),
                tickformat=".0%",
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor='#F5F5F7',
                linecolor='#E5E5EA',
                title="Сумма субсидий (тг)",
                title_font=dict(color='#86868B', size=11),
                tickfont=dict(color='#86868B', size=11),
                showgrid=False,
                zeroline=False,
            ),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                font=dict(color='#86868B', size=11),
                title_text="Кластер",
                title_font=dict(color='#86868B', size=11),
            ),
            margin=dict(l=10, r=20, t=24, b=10),
        )
        fig_sc.update_traces(
            marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.1)"))
        )

        st.markdown("<div style='padding:0 32px'>", unsafe_allow_html=True)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Bottom row: donut + stats table
        st.markdown("<div style='padding:0 32px 32px'>", unsafe_allow_html=True)
        col_pie, col_tbl = st.columns([1, 2], gap="large")
        st.markdown("</div>", unsafe_allow_html=True)

        with col_pie:
            cl_counts = sc_df.groupby("cluster_label").size().reset_index(name="count")
            colors_pie = [cmap3.get(c, "#4F7EFF") for c in cl_counts["cluster_label"]]

            fig_pie = go.Figure(go.Pie(
                labels=cl_counts["cluster_label"],
                values=cl_counts["count"],
                hole=0.65,
                marker=dict(
                    colors=colors_pie,
                    line=dict(color="#F5F5F7", width=2),
                ),
                textfont=dict(family="Inter, sans-serif", size=11, color="#86868B"),
            ))
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='#1D1D1F', size=12),
                height=320,
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    font=dict(color='#86868B', size=11),
                ),
                margin=dict(l=0, r=0, t=24, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_tbl:
            cl_stats = (
                sc_df.groupby("cluster_label")
                .agg(
                    Количество=("applicant_id", "count"),
                    Avg_ML=("ml_probability", "mean"),
                    Avg_Approval=("approval_rate", "mean"),
                    Avg_Subsidy=("total_amount_received", "mean"),
                )
                .reset_index()
                .rename(columns={"cluster_label": "Кластер"})
            )
            cl_stats["Avg_Approval"] = (cl_stats["Avg_Approval"] * 100).round(1).astype(str) + "%"
            cl_stats["Avg_Subsidy"]  = (cl_stats["Avg_Subsidy"] / 1_000_000).round(2)
            cl_stats["Avg_ML"]       = cl_stats["Avg_ML"].round(3)

            st.dataframe(
                cl_stats,
                use_container_width=True,
                hide_index=True,
                height=320,
                column_config={
                    "Avg_ML": st.column_config.ProgressColumn(
                        "Avg ML", format="%.3f", min_value=0, max_value=1
                    ),
                    "Avg_Subsidy": st.column_config.NumberColumn(
                        "Avg Субсидия, млн тг", format="%.2f"
                    ),
                },
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — АНОМАЛИИ
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        """
<div style="padding:24px 32px 16px;">
  <div style="font-family:'Inter',sans-serif;font-size:11px;
  color:#86868B;letter-spacing:1px;margin-bottom:4px;font-weight:500;">ISOLATION FOREST · ДЕТЕКЦИЯ</div>
  <div style="font-family:'Inter',sans-serif;font-size:22px;font-weight:600;
  color:#1D1D1F;letter-spacing:-0.5px;">Аномальные заявители</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if "is_anomaly" not in fdf.columns:
        st.markdown(
            """<div style="margin:0 32px;background:#FFFFFF;border:1px solid #F5F5F7;
border-radius:18px;padding:20px;font-family:'Inter',sans-serif;font-size:13px;
color:#86868B;box-shadow:0 2px 8px rgba(0,0,0,0.06);">Данные по аномалиям отсутствуют.</div>""",
            unsafe_allow_html=True,
        )
    else:
        anom_df = (
            fdf[fdf["is_anomaly"] == True]
            .sort_values("anomaly_score")
            .reset_index(drop=True)
        )
        n_anom  = len(anom_df)
        n_total = len(fdf)
        pct_str = f"{n_anom / n_total * 100:.1f}%" if n_total else "—"

        susp_region = "—"
        if n_anom and "oblast" in anom_df.columns:
            try:
                susp_region = anom_df["oblast"].value_counts().idxmax()
            except Exception:
                pass

        # Top 3 metrics
        am1, am2, am3 = st.columns(3)
        with am1:
            st.metric("Аномалий", n_anom)
        with am2:
            st.metric("Доля", pct_str)
        with am3:
            lbl = str(susp_region)
            st.metric("Топ-регион", lbl[:22] + "…" if len(lbl) > 22 else lbl)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Alert block
        st.markdown(
            f"""
<div style="background:#FFF5F5;
border:1px solid #FFD7D5;border-radius:18px;
padding:16px 20px;margin:0 32px 20px;
display:flex;align-items:center;gap:16px;">
  <div style="width:40px;height:40px;border-radius:10px;
  background:#FFE5E5;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;">
    <span style="font-size:18px;">&#9888;</span>
  </div>
  <div>
    <div style="font-family:'Inter',sans-serif;font-weight:600;
    color:#FF3B30;font-size:14px;">{n_anom} аномальных заявителей обнаружено</div>
    <div style="font-family:'Inter',sans-serif;color:#86868B;
    font-size:13px;margin-top:2px;">Рекомендуется ручная проверка</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if anom_df.empty:
            st.markdown(
                """<div style="margin:0 32px;background:#F0FFF4;
border:1px solid #C6F6D5;border-radius:18px;padding:20px;
font-family:'Inter',sans-serif;font-size:13px;color:#34C759;
box-shadow:0 2px 8px rgba(0,0,0,0.04);">
Аномалии не обнаружены при текущих фильтрах.</div>""",
                unsafe_allow_html=True,
            )
        else:
            acol_order = [
                "applicant_id", "ml_probability", "anomaly_score",
                "anomaly_reason", "oblast", "primary_direction",
            ]
            avail_a = [c for c in acol_order if c in anom_df.columns]
            tbl_a = anom_df[avail_a].copy()

            def style_anomalies(row):
                val = row.get("anomaly_score", 0)
                if pd.notna(val) and val < -0.2:
                    return (
                        ["background-color:rgba(239,68,68,0.06);color:#EF4444"] * len(row)
                    )
                return [""] * len(row)

            a_cfg: dict = {}
            if "ml_probability" in tbl_a.columns:
                a_cfg["ml_probability"] = st.column_config.ProgressColumn(
                    "ML Score", format="%.2f", min_value=0, max_value=1
                )
            if "anomaly_score" in tbl_a.columns:
                a_cfg["anomaly_score"] = st.column_config.NumberColumn(
                    "Anomaly Score", format="%.4f"
                )
            if "anomaly_reason" in tbl_a.columns:
                a_cfg["anomaly_reason"] = st.column_config.TextColumn("Причина", width="large")
            if "oblast" in tbl_a.columns:
                a_cfg["oblast"] = st.column_config.TextColumn("Регион")
            if "primary_direction" in tbl_a.columns:
                a_cfg["primary_direction"] = st.column_config.TextColumn("Направление")

            st.markdown("<div style='padding:0 32px 32px'>", unsafe_allow_html=True)
            st.dataframe(
                tbl_a.style.apply(style_anomalies, axis=1),
                column_config=a_cfg,
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
