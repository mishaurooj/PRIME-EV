# ============================================================
# PRIME-EV Interactive Decision Dashboard
# Professional Streamlit GUI for station-level EV infrastructure
# prioritization, risk assessment, downstream planning analysis,
# and baseline comparison.
#
# Run:
#   conda activate prime-ev
#   pip install streamlit pandas numpy scikit-learn plotly
#   cd /d D:\other\prime-ev
#   streamlit run prime_ev_dashboard.py
# ============================================================

import os
import re
import math
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PRIME-EV Decision Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #F7F9FC;}
    .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
    h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    .prime-card {
        background: white;
        border: 1px solid #E7EAF0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(30, 55, 90, 0.06);
        margin-bottom: 12px;
    }
    .small-muted {
        color: #5B667A;
        font-size: 0.86rem;
    }
    .status-good {
        background-color: #EAF7EF;
        color: #0B6B3A;
        border-radius: 10px;
        padding: 6px 10px;
        font-weight: 600;
        display: inline-block;
    }
    .status-mid {
        background-color: #FFF4D6;
        color: #8A5A00;
        border-radius: 10px;
        padding: 6px 10px;
        font-weight: 600;
        display: inline-block;
    }
    .status-risk {
        background-color: #FCE8E8;
        color: #A61B1B;
        border-radius: 10px;
        padding: 6px 10px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Dataset paths
# -----------------------------
CANDIDATE_DATA_PATHS = [
    r"D:\other\prime-ev\Dataset\ev_charging_stations-dataset.csv",
    r"D:\other\prime-ev\Code\ev_charging_stations-dataset.csv",
    "/mnt/data/ev_charging_stations-dataset.csv",
    "ev_charging_stations-dataset.csv",
]

NUMERIC_COLS = [
    "Cost (USD/kWh)",
    "Distance to City (km)",
    "Usage Stats (avg users/day)",
    "Charging Capacity (kW)",
    "Installation Year",
    "Reviews (Rating)",
    "Parking Spots",
]

CAT_COLS = [
    "Charger Type",
    "Station Operator",
    "Connector Types",
    "Renewable Energy Source",
    "Maintenance Frequency",
]

REQUIRED_COLS = ["Station ID", "Latitude", "Longitude", "Address", "Availability"] + NUMERIC_COLS + CAT_COLS


# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "Uploaded CSV"
    else:
        existing = next((p for p in CANDIDATE_DATA_PATHS if os.path.exists(p)), None)
        if existing is None:
            raise FileNotFoundError(
                "Dataset not found. Put ev_charging_stations-dataset.csv in "
                r"D:\other\prime-ev\Dataset or upload it from the sidebar."
            )
        df = pd.read_csv(existing)
        source = existing

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df.copy()
    df["Station ID"] = df["Station ID"].astype(str).str.strip().str.upper()

    for col in NUMERIC_COLS + ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CAT_COLS + ["Address", "Availability"]:
        df[col] = df[col].astype(str).fillna("Unknown")

    return df, source


def normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    norm = pd.DataFrame(
        scaler.fit_transform(df[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=df.index,
    )

    b = pd.DataFrame(index=df.index)
    b["low_cost"] = 1.0 - norm["Cost (USD/kWh)"]
    b["low_distance"] = 1.0 - norm["Distance to City (km)"]
    b["high_usage"] = norm["Usage Stats (avg users/day)"]
    b["high_capacity"] = norm["Charging Capacity (kW)"]
    b["newer_station"] = norm["Installation Year"]
    b["high_rating"] = norm["Reviews (Rating)"]
    b["parking"] = norm["Parking Spots"]
    b["renewable"] = df["Renewable Energy Source"].str.lower().str.contains(
        "yes|solar|wind|hydro|renew|green", regex=True
    ).astype(float)
    b["demand_capacity_fit"] = 1.0 - np.abs(b["high_usage"] - b["high_capacity"])
    b["cost_access_fit"] = np.sqrt(np.clip(b["low_cost"] * b["low_distance"], 0, 1))
    return norm, b


def build_prime_ev_scores(df: pd.DataFrame) -> pd.DataFrame:
    norm, b = normalize_features(df)

    demand_capacity = np.sqrt(np.clip(b["high_usage"] * b["high_capacity"], 0, 1))
    access_cost = np.sqrt(np.clip(b["low_distance"] * b["low_cost"], 0, 1))

    risk_score = (
        0.30 * norm["Cost (USD/kWh)"]
        + 0.25 * norm["Distance to City (km)"]
        + 0.18 * (1.0 - b["high_rating"])
        + 0.15 * np.abs(b["high_usage"] - b["high_capacity"])
        + 0.07 * (1.0 - b["renewable"])
        + 0.05 * (1.0 - b["newer_station"])
    )

    uncertainty = b[
        ["high_usage", "high_capacity", "high_rating", "low_distance", "low_cost", "renewable"]
    ].std(axis=1)

    deployment_impact = (
        0.26 * b["demand_capacity_fit"]
        + 0.22 * b["high_usage"]
        + 0.18 * b["high_capacity"]
        + 0.14 * b["high_rating"]
        + 0.10 * b["parking"]
        + 0.10 * b["renewable"]
    )

    utility = (
        0.30 * demand_capacity
        + 0.16 * access_cost
        + 0.15 * b["demand_capacity_fit"]
        + 0.13 * b["high_rating"]
        + 0.10 * b["renewable"]
        + 0.07 * b["parking"]
        + 0.04 * b["newer_station"]
        - 0.06 * risk_score
        - 0.03 * uncertainty
    )

    # Convert utility to a clean 0-100 score.
    utility_100 = 100 * (utility - utility.min()) / (utility.max() - utility.min() + 1e-12)
    risk_100 = 100 * risk_score
    impact_100 = 100 * deployment_impact
    uncertainty_100 = 100 * uncertainty

    out = df.copy()
    out["PRIME_RiskScore_100"] = risk_100
    out["PRIME_Uncertainty_100"] = uncertainty_100
    out["PRIME_DeploymentImpact_100"] = impact_100
    out["PRIME_UtilityScore_100"] = utility_100
    out["DemandCapacityFit_100"] = 100 * b["demand_capacity_fit"]
    out["CostAccessFit_100"] = 100 * b["cost_access_fit"]
    out["RenewableFlag"] = b["renewable"].astype(int)
    out["NetworkRank"] = out["PRIME_UtilityScore_100"].rank(ascending=False, method="min").astype(int)
    out["RiskRank"] = out["PRIME_RiskScore_100"].rank(ascending=True, method="min").astype(int)
    return out


def baseline_scores(df_scored: pd.DataFrame) -> pd.DataFrame:
    _, b = normalize_features(df_scored)

    weights = {
        "high_usage": 0.18,
        "high_capacity": 0.16,
        "high_rating": 0.14,
        "low_distance": 0.14,
        "low_cost": 0.12,
        "demand_capacity_fit": 0.12,
        "renewable": 0.08,
        "parking": 0.04,
        "newer_station": 0.02,
    }

    def weighted_sum(w):
        return sum(w[k] * b[k] for k in w)

    ahp_w = {
        "high_usage": 9 / 37,
        "high_capacity": 7 / 37,
        "high_rating": 6 / 37,
        "low_distance": 5 / 37,
        "low_cost": 4 / 37,
        "renewable": 3 / 37,
        "parking": 2 / 37,
        "newer_station": 1 / 37,
    }

    # MCDM and heuristic baselines.
    scores = {
        "PRIME-EV": df_scored["PRIME_UtilityScore_100"] / 100,
        "AHP": weighted_sum(ahp_w),
        "MultiObjective": weighted_sum(weights),
        "CostOnly": b["low_cost"],
        "DistanceOnly": b["low_distance"],
        "DemandCapacity": 0.5 * b["high_usage"] + 0.5 * b["high_capacity"],
        "QualityGreen": 0.55 * b["high_rating"] + 0.30 * b["renewable"] + 0.15 * b["newer_station"],
    }

    # ML baselines trained on the internal deployment proxy.
    X = pd.concat(
        [
            df_scored[NUMERIC_COLS].reset_index(drop=True),
            pd.get_dummies(df_scored[CAT_COLS], drop_first=False).reset_index(drop=True),
        ],
        axis=1,
    )
    y = weighted_sum(weights).to_numpy()

    try:
        gbr = GradientBoostingRegressor(random_state=42, n_estimators=60, max_depth=3)
        gbr.fit(X, y)
        scores["GBR"] = gbr.predict(X)
    except Exception:
        pass

    try:
        rf = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=7, n_jobs=-1)
        rf.fit(X, y)
        scores["RF"] = rf.predict(X)
    except Exception:
        pass

    base = pd.DataFrame(scores, index=df_scored.index)
    for c in base.columns:
        base[c] = 100 * (base[c] - base[c].min()) / (base[c].max() - base[c].min() + 1e-12)

    return base


def get_station_id_from_text(text: str) -> str:
    text = (text or "").strip().upper()
    match = re.search(r"EVS\d{5}", text)
    return match.group(0) if match else text


def status_label(value: float, reverse: bool = False) -> Tuple[str, str]:
    if reverse:
        good = value <= 35
        mid = 35 < value <= 65
    else:
        good = value >= 70
        mid = 45 <= value < 70

    if good:
        return "Favorable", "status-good"
    if mid:
        return "Moderate", "status-mid"
    return "Needs Attention", "status-risk"


def nearest_stations(df_scored: pd.DataFrame, station_idx: int, k: int = 6) -> pd.DataFrame:
    features = [
        "Latitude",
        "Longitude",
        "Cost (USD/kWh)",
        "Distance to City (km)",
        "Usage Stats (avg users/day)",
        "Charging Capacity (kW)",
        "Reviews (Rating)",
        "Parking Spots",
        "PRIME_UtilityScore_100",
        "PRIME_RiskScore_100",
    ]
    X = MinMaxScaler().fit_transform(df_scored[features])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df_scored)), metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors([X[station_idx]])
    idx = indices[0][1:]
    sim = df_scored.iloc[idx].copy()
    sim["SimilarityScore_100"] = 100 * (1 - distances[0][1:] / (distances[0][1:].max() + 1e-12))
    return sim


def make_gauge(title: str, value: float, reverse: bool = False) -> go.Figure:
    if reverse:
        steps = [
            {"range": [0, 35], "color": "#DCEFE4"},
            {"range": [35, 65], "color": "#FFF1C7"},
            {"range": [65, 100], "color": "#F9D6D5"},
        ]
    else:
        steps = [
            {"range": [0, 45], "color": "#F9D6D5"},
            {"range": [45, 70], "color": "#FFF1C7"},
            {"range": [70, 100], "color": "#DCEFE4"},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": "/100", "font": {"size": 30}},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#193B68"},
                "steps": steps,
                "threshold": {
                    "line": {"color": "#111111", "width": 3},
                    "thickness": 0.75,
                    "value": float(value),
                },
            },
        )
    )
    fig.update_layout(height=245, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor="white")
    return fig


def make_radar(row: pd.Series) -> go.Figure:
    labels = [
        "Utility",
        "Low Risk",
        "Impact",
        "Demand-Capacity Fit",
        "Cost-Access Fit",
        "Rating",
        "Renewable",
    ]
    values = [
        row["PRIME_UtilityScore_100"],
        100 - row["PRIME_RiskScore_100"],
        row["PRIME_DeploymentImpact_100"],
        row["DemandCapacityFit_100"],
        row["CostAccessFit_100"],
        100 * row["Reviews (Rating)"] / 5,
        100 * row["RenewableFlag"],
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=row["Station ID"],
            line=dict(color="#193B68", width=3),
            fillcolor="rgba(25,59,104,0.18)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=360,
        margin=dict(l=35, r=35, t=35, b=25),
        paper_bgcolor="white",
    )
    return fig


def make_baseline_chart(base_scores: pd.DataFrame, station_idx: int) -> go.Figure:
    row = base_scores.iloc[station_idx].sort_values(ascending=True)
    fig = px.bar(
        x=row.values,
        y=row.index,
        orientation="h",
        labels={"x": "Score / 100", "y": "Method"},
        text=[f"{v:.1f}" for v in row.values],
    )
    fig.update_traces(marker_color="#285C89", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=30, t=20, b=20),
        xaxis_range=[0, 105],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def make_network_scatter(df_scored: pd.DataFrame, row: pd.Series) -> go.Figure:
    fig = px.scatter(
        df_scored,
        x="PRIME_RiskScore_100",
        y="PRIME_UtilityScore_100",
        color="Charger Type",
        size="Charging Capacity (kW)",
        hover_name="Station ID",
        hover_data=["Address", "Usage Stats (avg users/day)", "Cost (USD/kWh)", "Reviews (Rating)"],
        labels={
            "PRIME_RiskScore_100": "Risk Score / 100",
            "PRIME_UtilityScore_100": "Utility Score / 100",
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[row["PRIME_RiskScore_100"]],
            y=[row["PRIME_UtilityScore_100"]],
            mode="markers+text",
            marker=dict(size=18, color="#C1121F", line=dict(color="white", width=2)),
            text=[row["Station ID"]],
            textposition="top center",
            name="Selected station",
        )
    )
    fig.update_layout(height=430, paper_bgcolor="white", plot_bgcolor="white", legend_title_text="")
    return fig


def make_map(df_scored: pd.DataFrame, row: pd.Series) -> go.Figure:
    sample = df_scored.copy()
    if len(sample) > 1200:
        sample = sample.sample(1200, random_state=42)

    fig = px.scatter_mapbox(
        sample,
        lat="Latitude",
        lon="Longitude",
        color="PRIME_UtilityScore_100",
        size="Charging Capacity (kW)",
        hover_name="Station ID",
        hover_data=["Address", "PRIME_UtilityScore_100", "PRIME_RiskScore_100"],
        color_continuous_scale="Viridis",
        zoom=1,
        height=460,
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=[row["Latitude"]],
            lon=[row["Longitude"]],
            mode="markers",
            marker=dict(size=18, color="#D62828"),
            text=[row["Station ID"]],
            name="Selected station",
        )
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
    )
    return fig


def recommendation_text(row: pd.Series) -> Tuple[str, str]:
    utility = row["PRIME_UtilityScore_100"]
    risk = row["PRIME_RiskScore_100"]
    fit = row["DemandCapacityFit_100"]
    renewable = row["RenewableFlag"]

    if utility >= 75 and risk <= 45:
        decision = "High-priority deployment candidate"
        action = "Keep this station in the top planning list. It offers strong utility with manageable risk."
    elif utility >= 60 and risk <= 65:
        decision = "Conditional priority candidate"
        action = "Review cost, distance, and maintenance before final approval. It can be prioritized if budget or access targets match your planning constraints."
    elif risk > 65:
        decision = "Risk-sensitive candidate"
        action = "Do not prioritize immediately. Check maintenance history, operator reliability, and cost exposure first."
    else:
        decision = "Lower-priority candidate"
        action = "Use this station as a backup option unless regional coverage requires it."

    details = []
    if fit < 45:
        details.append("Demand-capacity fit is weak, so utilization may not align with available charging capacity.")
    if renewable == 0:
        details.append("Renewable energy support is absent, which may reduce sustainability score.")
    if row["Reviews (Rating)"] < 3.5:
        details.append("Customer rating is below 3.5, so service quality may need inspection.")
    if row["Distance to City (km)"] > 12:
        details.append("Distance to city center is high, which may reduce accessibility.")

    if not details:
        details.append("No major station-level concern appears from the available dataset fields.")

    return decision, action + " " + " ".join(details)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("PRIME-EV")
st.sidebar.caption("Priority Ranking and Infrastructure Management Engine for EV Networks")

uploaded = st.sidebar.file_uploader("Optional: upload EV charging station CSV", type=["csv"])

try:
    raw_df, data_source = load_dataset(uploaded)
except Exception as exc:
    st.error(str(exc))
    st.stop()

df_scored = build_prime_ev_scores(raw_df)
base_scores = baseline_scores(df_scored)

st.sidebar.success(f"Loaded {len(df_scored):,} stations")
st.sidebar.caption(f"Source: {data_source}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Station search")
default_id = "EVS00001"
station_input = st.sidebar.text_input("Enter station ID", value=default_id, help="Example: EVS00001")
station_id = get_station_id_from_text(station_input)

if station_id not in set(df_scored["Station ID"]):
    st.sidebar.error("Station ID not found.")
    st.sidebar.info("Use a valid ID such as EVS00001 to EVS05000.")
    station_id = default_id

station_idx = int(df_scored.index[df_scored["Station ID"] == station_id][0])
row = df_scored.loc[station_idx]

top_k = st.sidebar.slider("Top-K stations to display", min_value=5, max_value=50, value=10, step=5)
show_map = st.sidebar.checkbox("Show station map", value=True)
show_similar = st.sidebar.checkbox("Show similar stations", value=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class="prime-card">
        <h1 style="margin-bottom: 0.15rem;">PRIME-EV Interactive Decision Dashboard</h1>
        <div class="small-muted">
            Enter a station ID such as <b>EVS00001</b>. The dashboard runs station-level risk assessment, utility ranking,
            deployment impact analysis, baseline comparison, similarity search, and visual planning diagnostics.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Station overview
# -----------------------------
c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])

with c1:
    st.markdown(
        f"""
        <div class="prime-card">
            <h3 style="margin-top: 0;">Selected Station</h3>
            <b>{row['Station ID']}</b><br>
            <span class="small-muted">{row['Address']}</span><br><br>
            <b>Operator:</b> {row['Station Operator']}<br>
            <b>Charger:</b> {row['Charger Type']}<br>
            <b>Connector:</b> {row['Connector Types']}<br>
            <b>Availability:</b> {row['Availability']}
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.metric("Network Rank", f"#{int(row['NetworkRank']):,}", f"out of {len(df_scored):,}")
with c3:
    st.metric("Utility Score", f"{row['PRIME_UtilityScore_100']:.1f}/100")
with c4:
    st.metric("Risk Score", f"{row['PRIME_RiskScore_100']:.1f}/100")

decision, action = recommendation_text(row)
status, status_class = status_label(row["PRIME_UtilityScore_100"])

st.markdown(
    f"""
    <div class="prime-card">
        <span class="{status_class}">{decision}</span>
        <p style="margin-top: 0.75rem; margin-bottom: 0;">{action}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Gauges
# -----------------------------
g1, g2, g3, g4 = st.columns(4)
with g1:
    st.plotly_chart(make_gauge("Utility", row["PRIME_UtilityScore_100"]), use_container_width=True)
with g2:
    st.plotly_chart(make_gauge("Risk", row["PRIME_RiskScore_100"], reverse=True), use_container_width=True)
with g3:
    st.plotly_chart(make_gauge("Deployment Impact", row["PRIME_DeploymentImpact_100"]), use_container_width=True)
with g4:
    st.plotly_chart(make_gauge("Demand-Capacity Fit", row["DemandCapacityFit_100"]), use_container_width=True)

# -----------------------------
# Main tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Station Analysis",
        "Model & Baselines",
        "Planning Visuals",
        "Top Stations",
        "Raw Data",
    ]
)

with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### PRIME-EV station profile")
        profile_df = pd.DataFrame(
            {
                "Metric": [
                    "Cost (USD/kWh)",
                    "Distance to City (km)",
                    "Usage (avg users/day)",
                    "Charging Capacity (kW)",
                    "Installation Year",
                    "Customer Rating",
                    "Parking Spots",
                    "Renewable Energy",
                    "Maintenance Frequency",
                ],
                "Value": [
                    row["Cost (USD/kWh)"],
                    row["Distance to City (km)"],
                    row["Usage Stats (avg users/day)"],
                    row["Charging Capacity (kW)"],
                    row["Installation Year"],
                    row["Reviews (Rating)"],
                    row["Parking Spots"],
                    row["Renewable Energy Source"],
                    row["Maintenance Frequency"],
                ],
            }
        )
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

        st.markdown("#### Decision interpretation")
        interpretation = pd.DataFrame(
            {
                "Output": [
                    "Utility score",
                    "Risk score",
                    "Deployment impact",
                    "Demand-capacity fit",
                    "Cost-access fit",
                    "Network rank",
                ],
                "Value": [
                    f"{row['PRIME_UtilityScore_100']:.2f}/100",
                    f"{row['PRIME_RiskScore_100']:.2f}/100",
                    f"{row['PRIME_DeploymentImpact_100']:.2f}/100",
                    f"{row['DemandCapacityFit_100']:.2f}/100",
                    f"{row['CostAccessFit_100']:.2f}/100",
                    f"{int(row['NetworkRank'])} of {len(df_scored)}",
                ],
                "Meaning": [
                    "Higher value means stronger priority for planning.",
                    "Lower value means lower operational concern.",
                    "Captures demand, capacity, rating, parking, and renewable support.",
                    "Checks whether station demand aligns with installed capacity.",
                    "Captures low cost and accessibility jointly.",
                    "Position of the station in the full network ranking.",
                ],
            }
        )
        st.dataframe(interpretation, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Multi-factor diagnostic radar")
        st.plotly_chart(make_radar(row), use_container_width=True)

with tab2:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Baseline comparison for selected station")
        st.plotly_chart(make_baseline_chart(base_scores, station_idx), use_container_width=True)

    with right:
        st.markdown("#### Method scores")
        method_table = base_scores.iloc[station_idx].sort_values(ascending=False).reset_index()
        method_table.columns = ["Method", "Score / 100"]
        method_table["Score / 100"] = method_table["Score / 100"].round(2)
        method_table["Rank"] = range(1, len(method_table) + 1)
        method_table = method_table[["Rank", "Method", "Score / 100"]]
        st.dataframe(method_table, use_container_width=True, hide_index=True)

    st.markdown("#### Model logic used in this dashboard")
    st.info(
        "The GUI uses a transparent PRIME-EV inference proxy when no trained checkpoint is available. "
        "It combines infrastructure representation inputs, risk score, uncertainty score, deployment impact, "
        "and utility ranking. This keeps the dashboard runnable directly from the CSV while matching the paper workflow: "
        "IRE → IRAM → DIM → PUN."
    )

with tab3:
    st.markdown("#### Network position: risk vs utility")
    st.plotly_chart(make_network_scatter(df_scored, row), use_container_width=True)

    if show_map:
        st.markdown("#### Geographic station view")
        st.plotly_chart(make_map(df_scored, row), use_container_width=True)

    if show_similar:
        st.markdown("#### Similar stations")
        sim = nearest_stations(df_scored, station_idx, k=6)
        sim_cols = [
            "Station ID",
            "Address",
            "Charger Type",
            "Cost (USD/kWh)",
            "Distance to City (km)",
            "Usage Stats (avg users/day)",
            "Charging Capacity (kW)",
            "PRIME_UtilityScore_100",
            "PRIME_RiskScore_100",
            "NetworkRank",
            "SimilarityScore_100",
        ]
        st.dataframe(
            sim[sim_cols].round(3),
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.markdown(f"#### Top {top_k} PRIME-EV priority stations")
    top = df_scored.sort_values("PRIME_UtilityScore_100", ascending=False).head(top_k)
    top_cols = [
        "NetworkRank",
        "Station ID",
        "Address",
        "Charger Type",
        "Station Operator",
        "PRIME_UtilityScore_100",
        "PRIME_RiskScore_100",
        "PRIME_DeploymentImpact_100",
        "Usage Stats (avg users/day)",
        "Charging Capacity (kW)",
        "Cost (USD/kWh)",
        "Distance to City (km)",
        "Renewable Energy Source",
    ]
    st.dataframe(top[top_cols].round(3), use_container_width=True, hide_index=True)

    fig_top = px.bar(
        top.sort_values("PRIME_UtilityScore_100"),
        x="PRIME_UtilityScore_100",
        y="Station ID",
        orientation="h",
        color="PRIME_RiskScore_100",
        color_continuous_scale="RdYlGn_r",
        labels={"PRIME_UtilityScore_100": "Utility Score / 100", "PRIME_RiskScore_100": "Risk / 100"},
        hover_data=["Address", "Charging Capacity (kW)", "Usage Stats (avg users/day)"],
    )
    fig_top.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_top, use_container_width=True)

with tab5:
    st.markdown("#### Filtered dataset with PRIME-EV outputs")
    search_text = st.text_input("Search address, operator, charger type, or station ID", value="")
    filtered = df_scored.copy()
    if search_text.strip():
        q = search_text.strip().lower()
        mask = (
            filtered["Station ID"].str.lower().str.contains(q)
            | filtered["Address"].str.lower().str.contains(q)
            | filtered["Station Operator"].str.lower().str.contains(q)
            | filtered["Charger Type"].str.lower().str.contains(q)
        )
        filtered = filtered[mask]

    cols = [
        "Station ID",
        "Address",
        "Charger Type",
        "Station Operator",
        "Cost (USD/kWh)",
        "Distance to City (km)",
        "Usage Stats (avg users/day)",
        "Charging Capacity (kW)",
        "Reviews (Rating)",
        "Renewable Energy Source",
        "PRIME_UtilityScore_100",
        "PRIME_RiskScore_100",
        "PRIME_DeploymentImpact_100",
        "NetworkRank",
    ]
    st.dataframe(filtered[cols].round(3), use_container_width=True, hide_index=True)

    csv = filtered[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered station results",
        data=csv,
        file_name="prime_ev_filtered_station_results.csv",
        mime="text/csv",
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="small-muted" style="padding-top: 1rem;">
    Suggested workflow: start with EVS00001, compare its utility and risk scores, inspect similar stations,
    then use the Top Stations tab to prepare a planning shortlist.
    </div>
    """,
    unsafe_allow_html=True,
)
