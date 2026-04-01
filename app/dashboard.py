"""
Subsidy Scoring AI — Dashboard
================================
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
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Subsidy AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT           = Path(__file__).resolve().parent.parent
FEATURES_PATH   = _ROOT / "data" / "processed" / "features.csv"
ML_RESULTS_PATH = _ROOT / "data" / "processed" / "ml_results.csv"
MODEL_PATH      = _ROOT / "data" / "processed" / "lgbm_model.pkl"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    #MainMenu, footer, header { visibility: hidden; }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
    }

    .stApp { background: #0A0A0B; }

    section[data-testid="stSidebar"] {
        background: #0D0D0F;
        border-right: 1px solid #1E1E21;
    }

    /* Typography */
    h1 {
        font-size: 28px !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
        color: #F5F5F7 !important;
    }
    h2 {
        font-size: 18px !important;
        font-weight: 500 !important;
        color: #A1A1AA !important;
        letter-spacing: -0.2px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #111113;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #1E1E21;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #71717A !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        transition: all 0.15s;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1E1E21 !important;
        color: #F5F5F7 !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: #111113;
        border: 1px solid #1E1E21;
        border-radius: 16px;
        padding: 20px 24px !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.03);
    }
    [data-testid="metric-container"] label {
        color: #71717A !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #F5F5F7 !important;
        font-size: 30px !important;
        font-weight: 600 !important;
        letter-spacing: -1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        color: #52525B !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E1E21 !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* Buttons */
    .stButton button {
        background: #5B8EF0 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        padding: 8px 20px !important;
        transition: all 0.15s !important;
        letter-spacing: 0.1px;
    }
    .stButton button:hover {
        background: #4A7DE0 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(91,142,240,0.3) !important;
    }

    /* Slider */
    [data-testid="stSlider"] > div > div > div {
        background: #5B8EF0 !important;
    }

    /* Text inputs */
    [data-testid="stTextInput"] input {
        background: #111113 !important;
        border: 1px solid #1E1E21 !important;
        border-radius: 10px !important;
        color: #F5F5F7 !important;
        font-size: 14px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #5B8EF0 !important;
        box-shadow: 0 0 0 3px rgba(91,142,240,0.15) !important;
    }

    /* Multiselect */
    [data-testid="stMultiSelect"] > div > div {
        background: #111113 !important;
        border: 1px solid #1E1E21 !important;
        border-radius: 10px !important;
        color: #F5F5F7 !important;
    }

    /* Sidebar labels */
    section[data-testid="stSidebar"] label {
        color: #52525B !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Dividers */
    hr { border-color: #1E1E21 !important; margin: 24px 0 !important; }

    /* Streamlit default padding overrides */
    .block-container { padding-top: 24px !important; padding-bottom: 48px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0 32px;">
      <div style="width:36px;height:36px;background:linear-gradient(135deg,#5B8EF0,#8B5CF6);
                  border-radius:10px;display:flex;align-items:center;justify-content:center;
                  font-size:16px;color:white;font-weight:700;flex-shrink:0;">S</div>
      <div>
        <div style="font-size:20px;font-weight:600;color:#F5F5F7;letter-spacing:-0.3px;line-height:1.2;">
          Subsidy AI</div>
        <div style="font-size:12px;color:#52525B;margin-top:1px;">
          Kazakhstan · 2025 · 457 applicants</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
        from src.ml_model import LGBM_FEATURES
        return pd.DataFrame({
            "Feature": LGBM_FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True).reset_index(drop=True)
    except Exception:
        return None


ml_df = load_ml_results()

# ---------------------------------------------------------------------------
# No data state
# ---------------------------------------------------------------------------

if ml_df is None:
    st.markdown(
        """
        <div style="text-align:center;padding:80px 0;">
          <div style="width:64px;height:64px;background:linear-gradient(135deg,#5B8EF0,#8B5CF6);
                      border-radius:16px;margin:0 auto 20px;display:flex;align-items:center;
                      justify-content:center;font-size:28px;color:white;font-weight:700;">S</div>
          <div style="font-size:18px;font-weight:500;color:#F5F5F7;margin-bottom:8px;">
            Data not loaded</div>
          <div style="font-size:14px;color:#52525B;margin-bottom:28px;">
            Run the ML pipeline to generate results</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Run ML Pipeline"):
        with st.spinner("Running pipeline..."):
            result = subprocess.run(
                [sys.executable, str(_ROOT / "src" / "ml_model.py")],
                capture_output=True, text=True, cwd=str(_ROOT),
            )
        if result.returncode == 0:
            st.success("Pipeline completed. Reload the page.")
            st.code(result.stdout[-3000:])
        else:
            st.error("Pipeline failed.")
            st.code(result.stderr[-2000:])
    st.stop()

# ---------------------------------------------------------------------------
# Ensure required columns exist
# ---------------------------------------------------------------------------

sys.path.insert(0, str(_ROOT))

if "ml_probability" not in ml_df.columns:
    ml_df["ml_probability"] = 0.5
if "ml_rank" not in ml_df.columns:
    ml_df["ml_rank"] = ml_df["ml_probability"].rank(ascending=False, method="first").astype(int)

# Load composite scores
try:
    from src.scoring import compute_scores, explain_score
    features_df = load_features()
    scored_df   = compute_scores(features_df) if features_df is not None else None
except Exception:
    scored_df = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#52525B;'
        'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:16px;">Filters</div>',
        unsafe_allow_html=True,
    )

    all_oblasts    = sorted(ml_df["oblast"].dropna().unique())
    all_directions = sorted(ml_df["primary_direction"].dropna().unique())

    sel_oblasts = st.multiselect(
        "Region",
        options=all_oblasts,
        default=[],
        placeholder="All regions",
    )
    sel_directions = st.multiselect(
        "Direction",
        options=all_directions,
        default=[],
        placeholder="All directions",
    )
    min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background:#0D1F0D;border:1px solid #14532D;border-radius:12px;
                    padding:14px 16px;">
          <div style="font-size:11px;color:#4ADE80;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
            System active</div>
          <div style="font-size:12px;color:#52525B;line-height:1.8;">
            Model: LightGBM + KMeans<br>
            Data: 36 651 applications</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if sel_oblasts:
        out = out[out["oblast"].isin(sel_oblasts)]
    if sel_directions:
        out = out[out["primary_direction"].isin(sel_directions)]
    out = out[out["ml_probability"] >= min_score]
    return out


fdf = apply_filters(ml_df)

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

total_count = len(fdf)
avg_ml      = fdf["ml_probability"].mean() if total_count else 0.0
anom_count  = int(fdf["is_anomaly"].sum()) if "is_anomaly" in fdf.columns else 0

top_region = "—"
if total_count and "oblast" in fdf.columns and "total_amount_received" in fdf.columns:
    try:
        top_region = (
            fdf.groupby("oblast")["total_amount_received"].sum().idxmax()
        )
    except Exception:
        pass

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Applicants", f"{total_count:,}")
with c2:
    st.metric("Avg ML Score", f"{avg_ml:.2f}")
with c3:
    st.metric("Anomalies", anom_count, delta=f"of {total_count}")
with c4:
    lbl = str(top_region)
    st.metric("Top Region", lbl[:22] + "…" if len(lbl) > 22 else lbl)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["Rating", "Applicant", "Segments", "Anomalies"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Rating
# ═══════════════════════════════════════════════════════════════════════════

_PLOTLY_LAYOUT = dict(
    plot_bgcolor="#111113",
    paper_bgcolor="#111113",
    font_color="#A1A1AA",
    font_size=12,
)

_CLUSTER_COLORS = ["#5B8EF0", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]

with tab1:
    st.markdown(
        '<div style="font-size:13px;font-weight:500;color:#71717A;'
        'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:20px;">'
        'AI applicant ranking</div>',
        unsafe_allow_html=True,
    )

    top_n = st.slider("Top N", 10, 50, 20, 5, key="tab1_topn")

    # Merge with composite score
    rank_df = fdf.copy()
    if scored_df is not None:
        rank_df = rank_df.merge(
            scored_df[["applicant_id", "final_score"]],
            on="applicant_id", how="left",
        )
    else:
        rank_df["final_score"] = np.nan

    rank_df = rank_df.sort_values("ml_probability", ascending=False).head(top_n)

    display_cols = {
        "applicant_id":       "ID",
        "ml_probability":     "ML Score",
        "final_score":        "Composite",
        "approval_rate":      "Approval",
        "cluster_label":      "Cluster",
        "is_anomaly":         "Anomaly",
        "oblast":             "Region",
    }
    avail = {k: v for k, v in display_cols.items() if k in rank_df.columns}
    tbl = rank_df[list(avail.keys())].rename(columns=avail).reset_index(drop=True)

    col_cfg: dict = {}
    if "ML Score" in tbl.columns:
        col_cfg["ML Score"] = st.column_config.ProgressColumn(
            "ML Score", format="%.2f", min_value=0, max_value=1
        )
    if "Composite" in tbl.columns:
        col_cfg["Composite"] = st.column_config.ProgressColumn(
            "Composite", format="%.2f", min_value=0, max_value=1
        )
    if "Approval" in tbl.columns:
        col_cfg["Approval"] = st.column_config.ProgressColumn(
            "Одобрений %", format="%.2f", min_value=0, max_value=1
        )
    if "Cluster" in tbl.columns:
        col_cfg["Cluster"] = st.column_config.TextColumn("Cluster")
    if "Anomaly" in tbl.columns:
        col_cfg["Anomaly"] = st.column_config.CheckboxColumn("Anomaly")

    st.dataframe(tbl, column_config=col_cfg, hide_index=True, use_container_width=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Bar chart
    top10 = fdf.nlargest(10, "ml_probability").copy()
    top10["applicant_id"] = top10["applicant_id"].astype(str)
    top10 = top10.sort_values("ml_probability", ascending=True)

    fig = px.bar(
        top10,
        x="ml_probability",
        y="applicant_id",
        orientation="h",
        color="cluster_label" if "cluster_label" in top10.columns else None,
        color_discrete_sequence=_CLUSTER_COLORS,
        labels={"ml_probability": "ML probability", "applicant_id": ""},
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(gridcolor="#1E1E21", title="ML probability", range=[0, 1.05]),
        yaxis=dict(gridcolor="#1E1E21", title="", type="category"),
        legend=dict(bgcolor="#111113", bordercolor="#1E1E21", borderwidth=1),
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Applicant analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div style="font-size:13px;font-weight:500;color:#71717A;'
        'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:20px;">'
        'Applicant analysis</div>',
        unsafe_allow_html=True,
    )

    aid_input = st.text_input(
        "",
        placeholder="Enter applicant ID...",
        label_visibility="collapsed",
        key="applicant_id_input",
    )

    if aid_input.strip():
        aid = aid_input.strip()
        ml_mask = ml_df["applicant_id"].astype(str) == aid

        if not ml_mask.any():
            st.markdown(
                f'<div style="background:#111113;border:1px solid #1E1E21;'
                f'border-radius:12px;padding:20px 24px;color:#71717A;font-size:14px;">'
                f'Applicant <code style="color:#F5F5F7">{aid}</code> not found.</div>',
                unsafe_allow_html=True,
            )
        else:
            ml_row    = ml_df[ml_mask].iloc[0]
            ml_prob   = float(ml_row.get("ml_probability", 0) or 0)
            ml_rnk    = int(ml_row.get("ml_rank", 0) or 0)
            cl_label  = str(ml_row.get("cluster_label", "—"))
            is_anom   = bool(ml_row.get("is_anomaly", False))
            anom_txt  = str(ml_row.get("anomaly_reason", ""))

            comp_score = None
            comp_rank  = None
            factors    = None

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

            left_col, right_col = st.columns([6, 4], gap="large")

            with left_col:
                # Score cards
                score_row = st.columns(2)
                with score_row[0]:
                    prob_pct   = f"{ml_prob:.0%}"
                    score_clr  = "#10B981" if ml_prob >= 0.7 else ("#F59E0B" if ml_prob >= 0.4 else "#EF4444")
                    st.markdown(
                        f'<div style="background:#111113;border:1px solid #1E1E21;'
                        f'border-radius:16px;padding:20px 24px;margin-bottom:12px;">'
                        f'<div style="font-size:11px;color:#71717A;text-transform:uppercase;'
                        f'letter-spacing:0.5px;margin-bottom:8px;">LightGBM probability</div>'
                        f'<div style="font-size:40px;font-weight:600;color:{score_clr};'
                        f'letter-spacing:-1px;line-height:1;">{prob_pct}</div>'
                        f'<div style="font-size:12px;color:#52525B;margin-top:6px;">'
                        f'Rank #{ml_rnk} of {len(ml_df)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with score_row[1]:
                    if comp_score is not None:
                        c_pct = f"{comp_score:.0%}"
                        c_clr = "#10B981" if comp_score >= 0.6 else ("#F59E0B" if comp_score >= 0.3 else "#EF4444")
                        diff  = abs(ml_prob - comp_score)
                        disc_block = ""
                        if diff > 0.3:
                            disc_block = (
                                f'<div style="margin-top:8px;font-size:11px;color:#F59E0B;'
                                f'background:rgba(245,158,11,0.08);border-radius:6px;'
                                f'padding:4px 8px;display:inline-block;">'
                                f'Discrepancy {diff:.2f}</div>'
                            )
                        cr = f"Rank #{comp_rank} of {len(ml_df)}" if comp_rank else ""
                        st.markdown(
                            f'<div style="background:#111113;border:1px solid #1E1E21;'
                            f'border-radius:16px;padding:20px 24px;margin-bottom:12px;">'
                            f'<div style="font-size:11px;color:#71717A;text-transform:uppercase;'
                            f'letter-spacing:0.5px;margin-bottom:8px;">Composite score</div>'
                            f'<div style="font-size:40px;font-weight:600;color:{c_clr};'
                            f'letter-spacing:-1px;line-height:1;">{c_pct}</div>'
                            f'<div style="font-size:12px;color:#52525B;margin-top:6px;">{cr}</div>'
                            f'{disc_block}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Factor bars
                if factors:
                    st.markdown(
                        '<div style="font-size:11px;font-weight:600;color:#52525B;'
                        'text-transform:uppercase;letter-spacing:0.8px;margin:20px 0 12px;">'
                        'Score factors</div>',
                        unsafe_allow_html=True,
                    )
                    fdf_factors = pd.DataFrame([
                        {"Factor": v["label"], "Contribution": v["contribution"],
                         "Weight": v["weight"], "Value": v["raw_value"]}
                        for v in factors.values()
                    ]).sort_values("Contribution", ascending=True)

                    fig_f = px.bar(
                        fdf_factors,
                        x="Contribution", y="Factor", orientation="h",
                        color="Contribution",
                        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                        range_color=[0, 0.35],
                    )
                    fig_f.update_traces(marker_line_width=0)
                    fig_f.update_layout(
                        **_PLOTLY_LAYOUT,
                        height=240,
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor="#1E1E21", title="Contribution"),
                        yaxis=dict(gridcolor="#1E1E21", title=""),
                    )
                    st.plotly_chart(fig_f, use_container_width=True)

                # Feature importance
                imp_df = load_lgbm_importances()
                if imp_df is not None:
                    st.markdown(
                        '<div style="font-size:11px;font-weight:600;color:#52525B;'
                        'text-transform:uppercase;letter-spacing:0.8px;margin:20px 0 12px;">'
                        'LightGBM feature importance</div>',
                        unsafe_allow_html=True,
                    )
                    fig_imp = px.bar(
                        imp_df,
                        x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=["#1E3A5F", "#5B8EF0", "#8B5CF6"],
                    )
                    fig_imp.update_traces(marker_line_width=0)
                    fig_imp.update_layout(
                        **_PLOTLY_LAYOUT,
                        height=280,
                        coloraxis_showscale=False,
                        xaxis=dict(gridcolor="#1E1E21", title="Importance (splits)"),
                        yaxis=dict(gridcolor="#1E1E21", title=""),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

            with right_col:
                # Cluster card
                st.markdown(
                    f'<div style="background:#111113;border:1px solid #1E1E21;'
                    f'border-radius:16px;padding:20px 24px;margin-bottom:12px;">'
                    f'<div style="font-size:11px;color:#71717A;text-transform:uppercase;'
                    f'letter-spacing:0.5px;margin-bottom:10px;">Cluster</div>'
                    f'<div style="font-size:16px;font-weight:500;color:#F5F5F7;">'
                    f'{cl_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Region / direction
                ob  = str(ml_row.get("oblast", "—"))
                prd = str(ml_row.get("primary_direction", "—"))
                tot = int(ml_row.get("total_applications", 0) or 0)
                apr = float(ml_row.get("approval_rate", 0) or 0)
                st.markdown(
                    f'<div style="background:#111113;border:1px solid #1E1E21;'
                    f'border-radius:16px;padding:20px 24px;margin-bottom:12px;">'
                    f'<div style="font-size:11px;color:#71717A;text-transform:uppercase;'
                    f'letter-spacing:0.5px;margin-bottom:12px;">Profile</div>'
                    f'<div style="display:grid;gap:10px;">'
                    f'<div><div style="font-size:11px;color:#52525B;">Region</div>'
                    f'<div style="font-size:13px;color:#F5F5F7;margin-top:2px;">{ob}</div></div>'
                    f'<div><div style="font-size:11px;color:#52525B;">Direction</div>'
                    f'<div style="font-size:13px;color:#F5F5F7;margin-top:2px;">{prd}</div></div>'
                    f'<div><div style="font-size:11px;color:#52525B;">Applications</div>'
                    f'<div style="font-size:13px;color:#F5F5F7;margin-top:2px;">{tot:,}</div></div>'
                    f'<div><div style="font-size:11px;color:#52525B;">Approval rate</div>'
                    f'<div style="font-size:13px;color:#F5F5F7;margin-top:2px;">{apr:.0%}</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Anomaly block
                if is_anom:
                    st.markdown(
                        f'<div style="background:#1C0A0A;border:1px solid #7F1D1D;'
                        f'border-radius:12px;padding:16px 20px;">'
                        f'<div style="color:#EF4444;font-weight:600;font-size:13px;'
                        f'margin-bottom:10px;">Anomaly detected</div>'
                        f'<div style="color:#A1A1AA;font-size:13px;line-height:1.6;">'
                        f'{anom_txt}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background:#0D1F0D;border:1px solid #14532D;'
                        'border-radius:12px;padding:16px 20px;">'
                        '<div style="color:#4ADE80;font-weight:600;font-size:13px;">'
                        'No anomaly</div>'
                        '<div style="color:#52525B;font-size:12px;margin-top:4px;">'
                        'Applicant is within normal range</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Segments
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        '<div style="font-size:13px;font-weight:500;color:#71717A;'
        'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:20px;">'
        'Applicant segmentation</div>',
        unsafe_allow_html=True,
    )

    if "cluster_label" not in fdf.columns:
        st.markdown(
            '<div style="color:#71717A;font-size:14px;">Cluster data not available.</div>',
            unsafe_allow_html=True,
        )
    else:
        sc_df = fdf.dropna(subset=["cluster_label"]).copy()
        sc_df["subsidy_mln"] = sc_df["total_amount_received"] / 1_000_000

        fig_sc = px.scatter(
            sc_df,
            x="approval_rate",
            y="subsidy_mln",
            color="cluster_label",
            size="total_applications",
            size_max=28,
            hover_data=["applicant_id", "oblast", "ml_probability"],
            color_discrete_sequence=_CLUSTER_COLORS,
            labels={
                "approval_rate": "Approval rate",
                "subsidy_mln":   "Total subsidy, mln KZT",
                "cluster_label": "Cluster",
            },
        )
        fig_sc.update_layout(
            **_PLOTLY_LAYOUT,
            height=440,
            xaxis=dict(gridcolor="#1E1E21", title="Approval rate", tickformat=".0%"),
            yaxis=dict(gridcolor="#1E1E21", title="Total subsidy, mln KZT"),
            legend=dict(bgcolor="#111113", bordercolor="#1E1E21", borderwidth=1,
                        title_text="Cluster"),
        )
        fig_sc.update_traces(marker_line_width=0)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_pie, col_tbl = st.columns([1, 1], gap="large")

        with col_pie:
            cl_counts = sc_df.groupby("cluster_label").size().reset_index(name="count")
            fig_pie = px.pie(
                cl_counts,
                values="count",
                names="cluster_label",
                hole=0.55,
                color_discrete_sequence=_CLUSTER_COLORS,
            )
            fig_pie.update_traces(
                textinfo="percent+label",
                textfont_size=12,
                marker=dict(line=dict(color="#0A0A0B", width=2)),
            )
            fig_pie.update_layout(
                **_PLOTLY_LAYOUT,
                height=320,
                showlegend=False,
                margin=dict(l=0, r=0, t=16, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_tbl:
            cl_stats = (
                sc_df.groupby("cluster_label")
                .agg(
                    Count=("applicant_id", "count"),
                    Avg_ML=("ml_probability", "mean"),
                    Avg_Approval=("approval_rate", "mean"),
                    Avg_Subsidy=("total_amount_received", "mean"),
                )
                .reset_index()
                .rename(columns={
                    "cluster_label": "Cluster",
                    "Count":         "Count",
                    "Avg_ML":        "Avg ML",
                    "Avg_Approval":  "Avg Approval",
                    "Avg_Subsidy":   "Avg Subsidy, mln",
                })
            )
            cl_stats["Avg Approval"]    = (cl_stats["Avg Approval"] * 100).round(1).astype(str) + "%"
            cl_stats["Avg ML"]          = cl_stats["Avg ML"].round(3)
            cl_stats["Avg Subsidy, mln"] = (cl_stats["Avg Subsidy, mln"] / 1_000_000).round(1)

            st.dataframe(
                cl_stats,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg ML": st.column_config.ProgressColumn(
                        "Avg ML", format="%.3f", min_value=0, max_value=1
                    ),
                },
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Anomalies
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    if "is_anomaly" not in fdf.columns:
        st.markdown(
            '<div style="color:#71717A;font-size:14px;">Anomaly data not available.</div>',
            unsafe_allow_html=True,
        )
    else:
        anom_df = fdf[fdf["is_anomaly"] == True].sort_values("anomaly_score").reset_index(drop=True)
        n_anom  = len(anom_df)

        if n_anom > 20:
            st.markdown(
                f'<div style="background:#1C0A0A;border:1px solid #7F1D1D;border-radius:12px;'
                f'padding:14px 20px;margin-bottom:24px;display:flex;align-items:center;gap:14px;">'
                f'<div style="font-size:20px;flex-shrink:0;">!</div>'
                f'<div>'
                f'<div style="color:#EF4444;font-weight:600;font-size:14px;">'
                f'{n_anom} anomalous applicants detected</div>'
                f'<div style="color:#71717A;font-size:13px;margin-top:2px;">'
                f'Manual review recommended</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="font-size:13px;font-weight:500;color:#71717A;'
            'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:20px;">'
            'Anomaly report</div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Anomalies", n_anom)
        with m2:
            pct = f"{n_anom / len(fdf) * 100:.1f}%" if len(fdf) else "—"
            st.metric("Share", pct)
        with m3:
            susp = "—"
            if n_anom and "oblast" in anom_df.columns:
                try:
                    susp = anom_df["oblast"].value_counts().idxmax()
                except Exception:
                    pass
            lbl = str(susp)
            st.metric("Top suspect region", lbl[:20] + "…" if len(lbl) > 20 else lbl)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        if anom_df.empty:
            st.markdown(
                '<div style="background:#0D1F0D;border:1px solid #14532D;border-radius:12px;'
                'padding:20px 24px;color:#4ADE80;font-size:14px;">No anomalies after filtering.</div>',
                unsafe_allow_html=True,
            )
        else:
            disp_cols = {
                "applicant_id":      "ID",
                "ml_probability":    "ML Score",
                "anomaly_score":     "Anomaly Score",
                "anomaly_reason":    "Reason",
                "oblast":            "Region",
                "primary_direction": "Direction",
            }
            avail = {k: v for k, v in disp_cols.items() if k in anom_df.columns}
            tbl_a = anom_df[list(avail.keys())].rename(columns=avail).copy()

            def highlight_critical(row):
                val = row.get("Anomaly Score", 0)
                if pd.notna(val) and val < -0.2:
                    return ["background-color: rgba(239,68,68,0.08); color: #EF4444"] * len(row)
                return [""] * len(row)

            styled = tbl_a.style.apply(highlight_critical, axis=1)

            a_cfg: dict = {}
            if "ML Score" in tbl_a.columns:
                a_cfg["ML Score"] = st.column_config.ProgressColumn(
                    "ML Score", format="%.2f", min_value=0, max_value=1
                )
            if "Anomaly Score" in tbl_a.columns:
                a_cfg["Anomaly Score"] = st.column_config.NumberColumn(
                    "Anomaly Score", format="%.4f", help="Lower = more suspicious"
                )
            if "Reason" in tbl_a.columns:
                a_cfg["Reason"] = st.column_config.TextColumn("Reason", width="large")

            st.dataframe(styled, use_container_width=True, hide_index=True, column_config=a_cfg)
