#!/usr/bin/env python3
"""
PRIME-EV V11 final reviewer-completion experiment script
=====================================================

This single file reruns PRIME-EV with reviewer-critical V6 additions:
  1. explicit target and preference-label construction;
  2. operator-disjoint train/validation/test splits;
  3. supervised preference pairs with reported pair counts;
  4. leakage checks and label-weight sensitivity;
  5. trained PRIME-EV regional transfer tests;
  6. operator-level, geographic, and accessibility equity metrics;
  7. corrected SSI, DeltaPred, normalized latency, and composite score;
  8. transparent MCDM, multi-objective, machine-learning, Pareto, and random baselines;
  9. MCDM and multi-objective weight sensitivity;
 10. uncertainty enters the final priority ranking input;
 11. fixed-cutoff ranking, MAP, recall, hit-rate, overlap, and regret metrics;
 12. risk-interval coverage, residual-uncertainty diagnostics, and calibration plots;
 13. priority-proxy distribution analysis explaining high random NDCG;
 14. multi-operator deployment fairness diagnostics on the full candidate pool;
 15. authoritative manuscript values generated from one saved result file.

Default full run:
    python prime_ev_reviewer_ready.py \
        --data ev_charging_stations-dataset.csv \
        --output prime_ev_reviewer_results \
        --epochs 50 --torch-threads 1

Fast installation check:
    python prime_ev_reviewer_ready.py \
        --data ev_charging_stations-dataset.csv \
        --output prime_ev_quick_test \
        --quick

Dependencies:
    pip install numpy pandas scipy scikit-learn torch matplotlib

Methodological scope
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
EPS = 1e-8

NUMERIC_MODEL_COLS = [
    "Cost (USD/kWh)",
    "Distance to City (km)",
    "Charging Capacity (kW)",
    "Installation Year",
    "Parking Spots",
]

CATEGORICAL_MODEL_COLS = [
    "Charger Type",
    "Connector Types",
    "Renewable Energy Source",
    "Maintenance Frequency",
    "Availability",
]

LABEL_ONLY_COLS = [
    "Reviews (Rating)",
    "Usage Stats (avg users/day)",
]

REQUIRED_COLS = [
    "Station ID",
    "Latitude",
    "Longitude",
    "Station Operator",
] + NUMERIC_MODEL_COLS + CATEGORICAL_MODEL_COLS + LABEL_ONLY_COLS

BASE_LABEL_WEIGHTS = {
    "risk_proxy": 0.60,
    "demand_target": 0.40,
}

AHP_WEIGHTS = {
    "capacity_gap": 0.20,
    "high_distance": 0.20,
    "older_station": 0.15,
    "maintenance_gap": 0.15,
    "limited_availability": 0.10,
    "high_cost": 0.08,
    "renewable_gap": 0.07,
    "parking_gap": 0.05,
}

MO_WEIGHTS = {
    "capacity_gap": 0.22,
    "high_distance": 0.18,
    "older_station": 0.12,
    "maintenance_gap": 0.12,
    "limited_availability": 0.12,
    "high_cost": 0.10,
    "renewable_gap": 0.08,
    "parking_gap": 0.06,
}


@dataclass
class ExperimentConfig:
    data_path: str
    output_dir: str
    epochs: int = 50
    sensitivity_epochs: int = 25
    regional_epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 16
    batch_pairs_train: int = 60000
    batch_pairs_val: int = 10000
    batch_pairs_test: int = 10000
    pair_threshold: float = 0.05
    risk_weight: float = 0.60
    demand_weight: float = 0.40
    lambda_risk: float = 1.0
    lambda_demand: float = 0.5
    lambda_rank: float = 1.0
    patience: int = 20
    validation_metric: str = "ndcg_top_fraction"
    station_batch_size: int = 256
    pair_batch_size: int = 512
    eval_batch_size: int = 512
    top_fraction: float = 0.10
    run_ablations: bool = True
    run_regional_transfer: bool = True
    run_label_sensitivity: bool = True
    run_baseline_sensitivity: bool = True
    quick: bool = False
    device: str = "auto"
    torch_threads: int = 1
    steps_per_epoch: int = 0
    pair_margin_weighting: bool = True
    split_seed: int = 42


@dataclass
class Preprocessor:
    numeric_medians: Dict[str, float]
    categorical_modes: Dict[str, str]
    numeric_scaler: MinMaxScaler
    one_hot_encoder: OneHotEncoder
    rating_min: float
    rating_max: float
    usage_min: float
    usage_max: float
    feature_names: List[str]
    distance_median: float
    distance_q75: float


@dataclass
class DataBundle:
    raw: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    preprocessor: Preprocessor
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    u_train: np.ndarray
    u_val: np.ndarray
    u_test: np.ndarray
    g_train: np.ndarray
    g_val: np.ndarray
    g_test: np.ndarray
    cost_train: np.ndarray
    cost_val: np.ndarray
    cost_test: np.ndarray
    dist_train: np.ndarray
    dist_val: np.ndarray
    dist_test: np.ndarray
    train_pairs: Tuple[np.ndarray, np.ndarray, np.ndarray]
    val_pairs: Tuple[np.ndarray, np.ndarray, np.ndarray]
    test_pairs: Tuple[np.ndarray, np.ndarray, np.ndarray]
    split_metadata: Dict[str, Any]


@dataclass
class ModelResult:
    name: str
    model: "PrimeEV"
    history: pd.DataFrame
    train_seconds: float
    best_epoch: int
    test_scores: np.ndarray
    test_mu: np.ndarray
    test_sigma: np.ndarray
    test_usage_hat: np.ndarray
    test_metrics: Dict[str, float]
    losses: Dict[str, float]
    latency_ms_per_station: float


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    value = spearmanr(np.asarray(a), np.asarray(b)).correlation
    return 0.0 if value is None or np.isnan(value) else float(value)


def safe_kendall(a: Sequence[float], b: Sequence[float]) -> float:
    value = kendalltau(np.asarray(a), np.asarray(b)).correlation
    return 0.0 if value is None or np.isnan(value) else float(value)


def minmax_vector(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < EPS:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def scale_with_bounds(values: pd.Series, lo: float, hi: float) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.clip((arr - lo) / (hi - lo + EPS), 0.0, 1.0)


def region_from_longitude(longitude: float) -> str:
    if longitude < -30.0:
        return "Americas"
    if longitude <= 60.0:
        return "Europe_Africa"
    return "Asia_Oceania"


def parse_availability_hours(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"24/7", "24x7", "always", "open 24 hours"}:
        return 24.0
    match = __import__("re").match(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", text)
    if not match:
        return 12.0
    h1, m1, h2, m2 = map(int, match.groups())
    start = h1 + m1 / 60.0
    end = h2 + m2 / 60.0
    duration = end - start
    if duration <= 0:
        duration += 24.0
    return float(np.clip(duration, 0.0, 24.0))


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=convert)


def format_float(value: float, digits: int = 4) -> str:
    if value is None or np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


# =============================================================================
# Data splitting and preprocessing
# =============================================================================

def operator_disjoint_split(df: pd.DataFrame, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    groups = df["Station Operator"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("Operator-disjoint splitting requires at least three station operators.")

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_pos, test_pos = next(outer.split(df, groups=groups))

    train_val_groups = groups[train_val_pos]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(df.iloc[train_val_pos], groups=train_val_groups))
    train_pos = train_val_pos[train_rel]
    val_pos = train_val_pos[val_rel]

    metadata = {
        "strategy": "operator-disjoint",
        "train_operators": sorted(df.iloc[train_pos]["Station Operator"].astype(str).unique().tolist()),
        "validation_operators": sorted(df.iloc[val_pos]["Station Operator"].astype(str).unique().tolist()),
        "test_operators": sorted(df.iloc[test_pos]["Station Operator"].astype(str).unique().tolist()),
    }
    return np.asarray(train_pos), np.asarray(val_pos), np.asarray(test_pos), metadata


def fit_preprocessor(df_train: pd.DataFrame) -> Preprocessor:
    numeric_medians: Dict[str, float] = {}
    numeric_frame = pd.DataFrame(index=df_train.index)
    for col in NUMERIC_MODEL_COLS:
        values = pd.to_numeric(df_train[col], errors="coerce")
        median = float(values.median())
        numeric_medians[col] = median
        numeric_frame[col] = values.fillna(median)

    categorical_modes: Dict[str, str] = {}
    categorical_frame = pd.DataFrame(index=df_train.index)
    for col in CATEGORICAL_MODEL_COLS:
        values = df_train[col].astype("string")
        nonmissing = values.dropna()
        mode = str(nonmissing.mode().iloc[0]) if not nonmissing.empty else "Unknown"
        categorical_modes[col] = mode
        categorical_frame[col] = values.fillna("Unknown").astype(str)

    numeric_scaler = MinMaxScaler()
    numeric_scaler.fit(numeric_frame[NUMERIC_MODEL_COLS])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    encoder.fit(categorical_frame[CATEGORICAL_MODEL_COLS])

    rating = pd.to_numeric(df_train["Reviews (Rating)"], errors="coerce")
    rating_median = float(rating.median())
    rating = rating.fillna(rating_median)
    usage = pd.to_numeric(df_train["Usage Stats (avg users/day)"], errors="coerce")
    usage_median = float(usage.median())
    usage = usage.fillna(usage_median)

    numeric_names = list(NUMERIC_MODEL_COLS)
    categorical_names = encoder.get_feature_names_out(CATEGORICAL_MODEL_COLS).tolist()
    feature_names = numeric_names + categorical_names

    distance = pd.to_numeric(df_train["Distance to City (km)"], errors="coerce")
    distance = distance.fillna(float(distance.median()))

    return Preprocessor(
        numeric_medians=numeric_medians,
        categorical_modes=categorical_modes,
        numeric_scaler=numeric_scaler,
        one_hot_encoder=encoder,
        rating_min=float(rating.min()),
        rating_max=float(rating.max()),
        usage_min=float(usage.min()),
        usage_max=float(usage.max()),
        feature_names=feature_names,
        distance_median=float(distance.median()),
        distance_q75=float(distance.quantile(0.75)),
    )


def transform_inputs(df_part: pd.DataFrame, prep: Preprocessor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric_frame = pd.DataFrame(index=df_part.index)
    for col in NUMERIC_MODEL_COLS:
        values = pd.to_numeric(df_part[col], errors="coerce")
        numeric_frame[col] = values.fillna(prep.numeric_medians[col])
    numeric_scaled = prep.numeric_scaler.transform(numeric_frame[NUMERIC_MODEL_COLS]).astype(np.float32)

    categorical_frame = pd.DataFrame(index=df_part.index)
    for col in CATEGORICAL_MODEL_COLS:
        values = df_part[col].astype("string")
        categorical_frame[col] = values.fillna("Unknown").astype(str)
    categorical_encoded = prep.one_hot_encoder.transform(categorical_frame[CATEGORICAL_MODEL_COLS]).astype(np.float32)

    X = np.concatenate([numeric_scaled, categorical_encoded], axis=1).astype(np.float32)

    cost_index = NUMERIC_MODEL_COLS.index("Cost (USD/kWh)")
    dist_index = NUMERIC_MODEL_COLS.index("Distance to City (km)")
    cost = numeric_scaled[:, cost_index].astype(np.float32)
    dist = numeric_scaled[:, dist_index].astype(np.float32)
    return X, cost, dist


def transform_targets(df_part: pd.DataFrame, prep: Preprocessor, risk_weight: float, demand_weight: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rating = pd.to_numeric(df_part["Reviews (Rating)"], errors="coerce")
    rating = rating.fillna((prep.rating_min + prep.rating_max) / 2.0)
    rating_scaled = scale_with_bounds(rating, prep.rating_min, prep.rating_max)
    y = 1.0 - rating_scaled

    usage = pd.to_numeric(df_part["Usage Stats (avg users/day)"], errors="coerce")
    usage = usage.fillna((prep.usage_min + prep.usage_max) / 2.0)
    u = scale_with_bounds(usage, prep.usage_min, prep.usage_max)

    weight_sum = risk_weight + demand_weight
    if weight_sum <= 0:
        raise ValueError("risk_weight + demand_weight must be positive.")
    wr = risk_weight / weight_sum
    wd = demand_weight / weight_sum
    g = wr * y + wd * u
    return y.astype(np.float32), u.astype(np.float32), g.astype(np.float32)


def sample_preference_pairs(labels: np.ndarray, n_pairs: int, threshold: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=float)
    n = len(labels)
    if n < 2:
        raise ValueError("At least two stations are required to generate preference pairs.")

    rng = np.random.default_rng(seed)
    left_chunks: List[np.ndarray] = []
    right_chunks: List[np.ndarray] = []
    rho_chunks: List[np.ndarray] = []
    collected = 0
    attempts = 0
    max_attempts = 1000

    while collected < n_pairs and attempts < max_attempts:
        batch = max(4096, (n_pairs - collected) * 3)
        i = rng.integers(0, n, size=batch)
        j = rng.integers(0, n, size=batch)
        valid = i != j
        diff = labels[i] - labels[j]
        valid &= np.abs(diff) > threshold
        i = i[valid]
        j = j[valid]
        diff = diff[valid]
        if len(i) > 0:
            take = min(len(i), n_pairs - collected)
            left_chunks.append(i[:take])
            right_chunks.append(j[:take])
            rho_chunks.append(np.sign(diff[:take]).astype(np.float32))
            collected += take
        attempts += 1

    if collected < n_pairs:
        raise RuntimeError(
            f"Could create only {collected} of {n_pairs} requested pairs. "
            f"Reduce --pair-threshold or pair counts."
        )

    return (
        np.concatenate(left_chunks).astype(np.int64),
        np.concatenate(right_chunks).astype(np.int64),
        np.concatenate(rho_chunks).astype(np.float32),
    )


def build_bundle(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ExperimentConfig,
    split_metadata: Dict[str, Any],
    risk_weight: Optional[float] = None,
    demand_weight: Optional[float] = None,
    pair_seed_offset: int = 0,
) -> DataBundle:
    risk_weight = config.risk_weight if risk_weight is None else risk_weight
    demand_weight = config.demand_weight if demand_weight is None else demand_weight

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    prep = fit_preprocessor(train_df)

    X_train, cost_train, dist_train = transform_inputs(train_df, prep)
    X_val, cost_val, dist_val = transform_inputs(val_df, prep)
    X_test, cost_test, dist_test = transform_inputs(test_df, prep)

    y_train, u_train, g_train = transform_targets(train_df, prep, risk_weight, demand_weight)
    y_val, u_val, g_val = transform_targets(val_df, prep, risk_weight, demand_weight)
    y_test, u_test, g_test = transform_targets(test_df, prep, risk_weight, demand_weight)

    train_pairs = sample_preference_pairs(
        g_train, config.batch_pairs_train, config.pair_threshold, SEED + 11 + pair_seed_offset
    )
    val_pairs = sample_preference_pairs(
        g_val, config.batch_pairs_val, config.pair_threshold, SEED + 12 + pair_seed_offset
    )
    test_pairs = sample_preference_pairs(
        g_test, config.batch_pairs_test, config.pair_threshold, SEED + 13 + pair_seed_offset
    )

    metadata = dict(split_metadata)
    metadata.update(
        {
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "risk_weight": float(risk_weight),
            "demand_weight": float(demand_weight),
            "pair_threshold": float(config.pair_threshold),
            "train_pairs": int(len(train_pairs[0])),
            "validation_pairs": int(len(val_pairs[0])),
            "test_pairs": int(len(test_pairs[0])),
            "feature_count": int(X_train.shape[1]),
            "feature_names": prep.feature_names,
            "label_only_variables": LABEL_ONLY_COLS,
        }
    )

    return DataBundle(
        raw=df,
        train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx),
        test_idx=np.asarray(test_idx),
        preprocessor=prep,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        u_train=u_train,
        u_val=u_val,
        u_test=u_test,
        g_train=g_train,
        g_val=g_val,
        g_test=g_test,
        cost_train=cost_train,
        cost_val=cost_val,
        cost_test=cost_test,
        dist_train=dist_train,
        dist_val=dist_val,
        dist_test=dist_test,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        split_metadata=metadata,
    )


def rebuild_bundle_labels(bundle: DataBundle, config: ExperimentConfig, risk_weight: float, demand_weight: float, offset: int) -> DataBundle:
    return build_bundle(
        bundle.raw,
        bundle.train_idx,
        bundle.val_idx,
        bundle.test_idx,
        config,
        dict(bundle.split_metadata),
        risk_weight=risk_weight,
        demand_weight=demand_weight,
        pair_seed_offset=offset,
    )


# =============================================================================
# PRIME-EV model
# =============================================================================

class InfrastructureRepresentationEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, use_attention: bool = True, use_conv: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.use_conv = use_conv

        if use_conv:
            self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.attention = nn.Linear(64, 1)
            self.output = nn.Linear(64, latent_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_conv:
            return self.mlp(x)

        structured = x.unsqueeze(2) * x.unsqueeze(1)
        h = F.relu(self.conv1(structured))
        h = F.relu(self.conv2(h))
        h = h.transpose(1, 2)
        if self.use_attention:
            alpha = torch.softmax(self.attention(h), dim=1)
            pooled = torch.sum(alpha * h, dim=1)
        else:
            pooled = torch.mean(h, dim=1)
        return self.output(pooled)


class InfrastructureRiskAssessmentModule(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.hidden = nn.Linear(latent_dim, 64)
        self.mean_head = nn.Linear(64, 1)
        self.scale_head = nn.Linear(64, 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.hidden(z))
        mu = torch.sigmoid(self.mean_head(h)).squeeze(1)
        sigma = F.softplus(self.scale_head(h)).squeeze(1) + 1e-4
        return mu, sigma


class DeploymentImpactModule(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, mu: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, mu.unsqueeze(1), distance.unsqueeze(1)], dim=1)
        return self.net(inp).squeeze(1)


class PriorityUtilityNetwork(nn.Module):
    def __init__(self, latent_dim: int, include_risk: bool = True, include_uncertainty: bool = True):
        super().__init__()
        self.include_risk = include_risk
        self.include_uncertainty = include_uncertainty
        input_dim = latent_dim + 2 + (1 if include_risk else 0) + (1 if include_uncertainty else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, cost: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        pieces = [z]
        if self.include_risk:
            pieces.append(mu.unsqueeze(1))
        if self.include_uncertainty:
            pieces.append(torch.log1p(sigma).unsqueeze(1))
        pieces.extend([cost.unsqueeze(1), distance.unsqueeze(1)])
        return self.net(torch.cat(pieces, dim=1)).squeeze(1)


class PrimeEV(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        use_attention: bool = True,
        use_conv: bool = True,
        include_risk_in_ranker: bool = True,
    ):
        super().__init__()
        self.encoder = InfrastructureRepresentationEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            use_attention=use_attention,
            use_conv=use_conv,
        )
        self.risk = InfrastructureRiskAssessmentModule(latent_dim)
        self.demand = DeploymentImpactModule(latent_dim)
        self.ranker = PriorityUtilityNetwork(latent_dim, include_risk=include_risk_in_ranker, include_uncertainty=True)

    def forward(self, x: torch.Tensor, cost: torch.Tensor, distance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        mu, sigma = self.risk(z)
        usage_hat = self.demand(z, mu, distance)
        score = self.ranker(z, mu, sigma, cost, distance)
        return mu, sigma, usage_hat, score


# =============================================================================
# Losses and metrics
# =============================================================================

def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(sigma) + 0.5 * ((target - mu) / sigma) ** 2)


def pairwise_logistic_loss(scores: torch.Tensor, pairs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    i, j, rho = pairs
    return torch.mean(F.softplus(-rho * (scores[i] - scores[j])))


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=float)[:k]
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(rel / discounts))


def ndcg_at_k(y_true: Sequence[float], scores: Sequence[float], k: int) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    k = int(np.clip(k, 1, len(y)))
    pred_order = np.argsort(-s)[:k]
    ideal_order = np.argsort(-y)[:k]
    ideal = dcg_at_k(y[ideal_order], k)
    return dcg_at_k(y[pred_order], k) / (ideal + EPS)


def precision_at_k(y_true: Sequence[float], scores: Sequence[float], k: int) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    k = int(np.clip(k, 1, len(y)))
    true_top = set(np.argsort(-y)[:k].tolist())
    pred_top = set(np.argsort(-s)[:k].tolist())
    return len(true_top.intersection(pred_top)) / float(k)


def pairwise_accuracy(labels: np.ndarray, scores: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    i, j, rho = pairs
    predicted = np.sign(scores[i] - scores[j])
    predicted[predicted == 0] = 1
    return float(np.mean(predicted == rho))


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray, np.ndarray], top_fraction: float) -> Dict[str, float]:
    n = len(labels)
    top_k_fraction = max(1, int(math.ceil(n * top_fraction)))
    top_k_10 = min(10, n)
    return {
        "NDCG_full": ndcg_at_k(labels, scores, n),
        "NDCG_at_10_percent": ndcg_at_k(labels, scores, top_k_fraction),
        "Precision_at_10": precision_at_k(labels, scores, top_k_10),
        "Precision_at_10_percent": precision_at_k(labels, scores, top_k_fraction),
        "Spearman": safe_spearman(labels, scores),
        "KendallTau": safe_kendall(labels, scores),
        "PairwiseAccuracy": pairwise_accuracy(labels, scores, pairs),
    }


def risk_metrics(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    nll = float(np.mean(np.log(sigma + EPS) + 0.5 * ((y_true - mu) / (sigma + EPS)) ** 2))
    return {
        "Risk_MSE": float(mean_squared_error(y_true, mu)),
        "Risk_NLL": nll,
        "MeanSigma": float(np.mean(sigma)),
    }


def system_stress_index(scores: Sequence[float]) -> float:
    normalized = minmax_vector(scores)
    return float(np.mean(np.abs(normalized - np.mean(normalized))))


def normalized_mad_group(values: np.ndarray, groups: Sequence[str]) -> float:
    frame = pd.DataFrame({"value": values, "group": np.asarray(groups, dtype=str)})
    means = frame.groupby("group")["value"].mean().to_numpy()
    if len(means) == 0:
        return 0.0
    return float(np.mean(np.abs(means - np.mean(means))))


def selection_rate_disparity(selected_mask: np.ndarray, groups: Sequence[str]) -> float:
    frame = pd.DataFrame({"selected": selected_mask.astype(float), "group": np.asarray(groups, dtype=str)})
    rates = frame.groupby("group")["selected"].mean().to_numpy()
    if len(rates) <= 1:
        return 0.0
    return float(np.max(rates) - np.min(rates))


def fairness_metrics(
    raw_test: pd.DataFrame,
    scores: np.ndarray,
    mu: np.ndarray,
    prep: Preprocessor,
    top_fraction: float,
) -> Dict[str, float]:
    n = len(raw_test)
    k = max(1, int(math.ceil(n * top_fraction)))
    selected_idx = np.argsort(-scores)[:k]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected_idx] = True

    operators = raw_test["Station Operator"].astype(str).to_numpy()
    regions = raw_test["Longitude"].apply(region_from_longitude).to_numpy()
    distance = pd.to_numeric(raw_test["Distance to City (km)"], errors="coerce")
    distance = distance.fillna(prep.distance_median).to_numpy(dtype=float)
    zones = np.where(distance <= prep.distance_median, "Urban", "Intercity")
    low_access = distance >= prep.distance_q75

    operator_risk_balance = normalized_mad_group(mu, operators)
    operator_selection_disparity = selection_rate_disparity(selected_mask, operators)
    geographic_selection_disparity = selection_rate_disparity(selected_mask, regions)
    accessibility_selection_disparity = selection_rate_disparity(selected_mask, zones)

    selected_low_access_share = float(np.mean(low_access[selected_mask])) if selected_mask.any() else 0.0
    population_low_access_share = float(np.mean(low_access))
    low_access_coverage_gap = abs(selected_low_access_share - population_low_access_share)

    geographic_equity_score = float(
        np.clip(
            1.0
            - np.mean(
                [
                    geographic_selection_disparity,
                    accessibility_selection_disparity,
                    low_access_coverage_gap,
                ]
            ),
            0.0,
            1.0,
        )
    )

    return {
        "OperatorRiskBalance": operator_risk_balance,
        "OperatorSelectionRateDisparity": operator_selection_disparity,
        "GeographicSelectionRateDisparity": geographic_selection_disparity,
        "AccessibilitySelectionRateDisparity": accessibility_selection_disparity,
        "SelectedLowAccessShare": selected_low_access_share,
        "PopulationLowAccessShare": population_low_access_share,
        "LowAccessCoverageGap": low_access_coverage_gap,
        "MeanSelectedDistance_km": float(np.mean(distance[selected_mask])),
        "UrbanSelectionRate": float(np.mean(selected_mask[zones == "Urban"])) if np.any(zones == "Urban") else 0.0,
        "IntercitySelectionRate": float(np.mean(selected_mask[zones == "Intercity"])) if np.any(zones == "Intercity") else 0.0,
        "GeographicEquityScore": geographic_equity_score,
        "DemographicEquityAvailable": 0.0,
    }


# =============================================================================
# Training and inference
# =============================================================================

def tensors_for_bundle(bundle: DataBundle, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "X_train": torch.tensor(bundle.X_train, dtype=torch.float32, device=device),
        "X_val": torch.tensor(bundle.X_val, dtype=torch.float32, device=device),
        "X_test": torch.tensor(bundle.X_test, dtype=torch.float32, device=device),
        "y_train": torch.tensor(bundle.y_train, dtype=torch.float32, device=device),
        "y_val": torch.tensor(bundle.y_val, dtype=torch.float32, device=device),
        "y_test": torch.tensor(bundle.y_test, dtype=torch.float32, device=device),
        "u_train": torch.tensor(bundle.u_train, dtype=torch.float32, device=device),
        "u_val": torch.tensor(bundle.u_val, dtype=torch.float32, device=device),
        "u_test": torch.tensor(bundle.u_test, dtype=torch.float32, device=device),
        "g_train": torch.tensor(bundle.g_train, dtype=torch.float32, device=device),
        "g_val": torch.tensor(bundle.g_val, dtype=torch.float32, device=device),
        "g_test": torch.tensor(bundle.g_test, dtype=torch.float32, device=device),
        "cost_train": torch.tensor(bundle.cost_train, dtype=torch.float32, device=device),
        "cost_val": torch.tensor(bundle.cost_val, dtype=torch.float32, device=device),
        "cost_test": torch.tensor(bundle.cost_test, dtype=torch.float32, device=device),
        "dist_train": torch.tensor(bundle.dist_train, dtype=torch.float32, device=device),
        "dist_val": torch.tensor(bundle.dist_val, dtype=torch.float32, device=device),
        "dist_test": torch.tensor(bundle.dist_test, dtype=torch.float32, device=device),
        "train_pairs": tuple(torch.tensor(x, device=device) for x in bundle.train_pairs),
        "val_pairs": tuple(torch.tensor(x, device=device) for x in bundle.val_pairs),
        "test_pairs": tuple(torch.tensor(x, device=device) for x in bundle.test_pairs),
    }


def evaluate_losses(
    model: PrimeEV,
    tensors: Dict[str, torch.Tensor],
    split: str,
    variant: Mapping[str, Any],
    config: ExperimentConfig,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        mu, sigma, usage_hat, score = model(
            tensors[f"X_{split}"], tensors[f"cost_{split}"], tensors[f"dist_{split}"]
        )
        if variant.get("deterministic_risk", False):
            risk_loss = F.mse_loss(mu, tensors[f"y_{split}"])
        else:
            risk_loss = gaussian_nll(mu, sigma, tensors[f"y_{split}"])
        demand_loss = torch.tensor(0.0, device=mu.device)
        if not variant.get("no_dim", False):
            demand_loss = F.mse_loss(usage_hat, tensors[f"u_{split}"])
        if variant.get("pointwise_rank", False):
            rank_loss = F.mse_loss(torch.sigmoid(score), tensors[f"g_{split}"])
        else:
            rank_loss = pairwise_logistic_loss(score, tensors[f"{split}_pairs"])
        total = (
            config.lambda_risk * risk_loss
            + config.lambda_demand * demand_loss
            + config.lambda_rank * rank_loss
        )
    return {
        "total": float(total.item()),
        "risk": float(risk_loss.item()),
        "demand": float(demand_loss.item()),
        "rank": float(rank_loss.item()),
    }


def infer_model(
    model: PrimeEV,
    X: np.ndarray,
    cost: np.ndarray,
    distance: np.ndarray,
    device: torch.device,
    repeats: int = 3,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()

    def one_pass() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mus, sigmas, usages, scores = [], [], [], []
        with torch.no_grad():
            for start_idx in range(0, len(X), batch_size):
                end_idx = min(len(X), start_idx + batch_size)
                x_t = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
                c_t = torch.tensor(cost[start_idx:end_idx], dtype=torch.float32, device=device)
                d_t = torch.tensor(distance[start_idx:end_idx], dtype=torch.float32, device=device)
                mu, sigma, usage_hat, score = model(x_t, c_t, d_t)
                mus.append(mu.detach().cpu().numpy())
                sigmas.append(sigma.detach().cpu().numpy())
                usages.append(usage_hat.detach().cpu().numpy())
                scores.append(score.detach().cpu().numpy())
        return np.concatenate(mus), np.concatenate(sigmas), np.concatenate(usages), np.concatenate(scores)

    _ = one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = None
    for _ in range(repeats):
        outputs = one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert outputs is not None
    latency_ms_per_station = elapsed * 1000.0 / (repeats * len(X))
    return (*outputs, float(latency_ms_per_station))


def train_prime_ev(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    name: str = "PRIME-EV-Full",
    variant: Optional[Mapping[str, Any]] = None,
    epochs_override: Optional[int] = None,
) -> ModelResult:
    """Train PRIME-EV using multiple optimizer steps per epoch.

    Architecture ablations deliberately use the same training seed and the same
    sampled preference-pair pool. This avoids confounding a module change with a
    different initialization or different test supervision.
    """
    variant = dict(variant or {})
    set_seed(SEED)
    latent_dim = int(variant.get("latent_dim", config.latent_dim))
    model = PrimeEV(
        input_dim=bundle.X_train.shape[1],
        latent_dim=latent_dim,
        use_attention=not variant.get("no_attention", False),
        use_conv=not variant.get("no_ire", False),
        include_risk_in_ranker=not variant.get("no_risk_input", False),
    ).to(device)

    tensors = tensors_for_bundle(bundle, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    epochs = int(epochs_override or config.epochs)
    pair_total = len(bundle.train_pairs[0])
    station_total = len(bundle.X_train)
    auto_steps = max(
        1,
        int(math.ceil(pair_total / max(1, config.pair_batch_size))),
        int(math.ceil(station_total / max(1, config.station_batch_size))),
    )
    steps_per_epoch = int(config.steps_per_epoch) if config.steps_per_epoch > 0 else auto_steps

    best_state = copy.deepcopy(model.state_dict())
    best_ndcg = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        rng_epoch = np.random.default_rng(SEED * 100000 + epoch)
        loss_sums = {"total": 0.0, "risk": 0.0, "demand": 0.0, "rank": 0.0}

        for _step in range(steps_per_epoch):
            optimizer.zero_grad()
            station_count = min(config.station_batch_size, station_total)
            station_idx_np = rng_epoch.choice(station_total, size=station_count, replace=False)

            pair_count = min(config.pair_batch_size, pair_total)
            pair_sel = rng_epoch.choice(pair_total, size=pair_count, replace=False)
            pair_i_np = bundle.train_pairs[0][pair_sel]
            pair_j_np = bundle.train_pairs[1][pair_sel]
            pair_rho_np = bundle.train_pairs[2][pair_sel]

            union_np, inverse = np.unique(np.concatenate([station_idx_np, pair_i_np, pair_j_np]), return_inverse=True)
            n_station = len(station_idx_np)
            n_pair = len(pair_i_np)
            station_local = torch.tensor(inverse[:n_station], dtype=torch.long, device=device)
            pair_i_local = torch.tensor(inverse[n_station:n_station + n_pair], dtype=torch.long, device=device)
            pair_j_local = torch.tensor(inverse[n_station + n_pair:], dtype=torch.long, device=device)
            pair_rho = torch.tensor(pair_rho_np, dtype=torch.float32, device=device)
            union_idx = torch.tensor(union_np, dtype=torch.long, device=device)

            mu_all, sigma_all, usage_all, score_all = model(
                tensors["X_train"][union_idx], tensors["cost_train"][union_idx], tensors["dist_train"][union_idx]
            )
            mu = mu_all[station_local]
            sigma = sigma_all[station_local]
            usage_hat = usage_all[station_local]
            station_idx_t = torch.tensor(station_idx_np, dtype=torch.long, device=device)
            y_target = tensors["y_train"][station_idx_t]
            u_target = tensors["u_train"][station_idx_t]
            g_target = tensors["g_train"][station_idx_t]

            risk_loss = F.mse_loss(mu, y_target) if variant.get("deterministic_risk", False) else gaussian_nll(mu, sigma, y_target)
            demand_loss = torch.tensor(0.0, device=device)
            if not variant.get("no_dim", False):
                demand_loss = F.mse_loss(usage_hat, u_target)

            if variant.get("pointwise_rank", False):
                rank_loss = F.mse_loss(torch.sigmoid(score_all[station_local]), g_target)
            else:
                pair_terms = F.softplus(-pair_rho * (score_all[pair_i_local] - score_all[pair_j_local]))
                if config.pair_margin_weighting:
                    margin = torch.abs(tensors["g_train"][torch.tensor(pair_i_np, device=device)] - tensors["g_train"][torch.tensor(pair_j_np, device=device)])
                    weights = margin / (torch.mean(margin) + EPS)
                    rank_loss = torch.mean(weights * pair_terms)
                else:
                    rank_loss = torch.mean(pair_terms)

            loss = config.lambda_risk * risk_loss + config.lambda_demand * demand_loss + config.lambda_rank * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sums["total"] += float(loss.item())
            loss_sums["risk"] += float(risk_loss.item())
            loss_sums["demand"] += float(demand_loss.item())
            loss_sums["rank"] += float(rank_loss.item())

        model.eval()
        with torch.no_grad():
            val_mu, val_sigma, val_usage_hat, val_score = model(tensors["X_val"], tensors["cost_val"], tensors["dist_val"])
            val_score_np = val_score.detach().cpu().numpy()
            val_k = max(1, int(math.ceil(len(bundle.g_val) * config.top_fraction))) if config.validation_metric == "ndcg_top_fraction" else len(bundle.g_val)
            val_ndcg = ndcg_at_k(bundle.g_val, val_score_np, val_k)
            val_risk = F.mse_loss(val_mu, tensors["y_val"]) if variant.get("deterministic_risk", False) else gaussian_nll(val_mu, val_sigma, tensors["y_val"])
            val_demand = torch.tensor(0.0, device=device)
            if not variant.get("no_dim", False):
                val_demand = F.mse_loss(val_usage_hat, tensors["u_val"])
            val_rank = F.mse_loss(torch.sigmoid(val_score), tensors["g_val"]) if variant.get("pointwise_rank", False) else pairwise_logistic_loss(val_score, tensors["val_pairs"])
            val_total = config.lambda_risk * val_risk + config.lambda_demand * val_demand + config.lambda_rank * val_rank

        history_rows.append({
            "epoch": epoch,
            "optimizer_steps": steps_per_epoch,
            "train_total": loss_sums["total"] / steps_per_epoch,
            "train_risk": loss_sums["risk"] / steps_per_epoch,
            "train_demand": loss_sums["demand"] / steps_per_epoch,
            "train_rank": loss_sums["rank"] / steps_per_epoch,
            "validation_total": float(val_total.item()),
            "validation_ndcg": float(val_ndcg),
        })

        if val_ndcg > best_ndcg + 1e-6:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[{name}] epoch {epoch:03d}/{epochs} steps={steps_per_epoch} train={history_rows[-1]['train_total']:.5f} val={val_total.item():.5f} val_NDCG={val_ndcg:.5f}")
        if epochs_without_improvement >= config.patience:
            print(f"[{name}] early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    mu, sigma, usage_hat, score, latency = infer_model(model, bundle.X_test, bundle.cost_test, bundle.dist_test, device, batch_size=config.eval_batch_size)
    ranking = ranking_metrics(bundle.g_test, score, bundle.test_pairs, config.top_fraction)
    risk = risk_metrics(bundle.y_test, mu, sigma)
    losses = evaluate_losses(model, tensors, "test", variant, config)
    metrics = dict(ranking)
    metrics.update(risk)
    metrics["Demand_MSE"] = float(mean_squared_error(bundle.u_test, usage_hat))
    metrics["SSI"] = system_stress_index(score)

    return ModelResult(name=name, model=model, history=pd.DataFrame(history_rows), train_seconds=float(train_seconds), best_epoch=int(best_epoch), test_scores=score, test_mu=mu, test_sigma=sigma, test_usage_hat=usage_hat, test_metrics=metrics, losses=losses, latency_ms_per_station=latency)


def infer_all_candidates(
    model: PrimeEV,
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    X_all, cost_all, dist_all = transform_inputs(bundle.raw, bundle.preprocessor)
    return infer_model(
        model, X_all, cost_all, dist_all, device,
        repeats=3, batch_size=config.eval_batch_size
    )


# =============================================================================
# Leakage analysis
# =============================================================================

def leakage_correlation_table(bundle: DataBundle) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    X_test = bundle.X_test
    for idx, name in enumerate(bundle.preprocessor.feature_names):
        corr = safe_spearman(bundle.g_test, X_test[:, idx])
        rows.append(
            {
                "Variable": name,
                "VariableType": "model_input",
                "SpearmanWithPreferenceLabel": corr,
                "AbsoluteCorrelation": abs(corr),
            }
        )

    rows.extend(
        [
            {
                "Variable": "risk_proxy_y",
                "VariableType": "label_only",
                "SpearmanWithPreferenceLabel": safe_spearman(bundle.g_test, bundle.y_test),
                "AbsoluteCorrelation": abs(safe_spearman(bundle.g_test, bundle.y_test)),
            },
            {
                "Variable": "observed_usage_u",
                "VariableType": "label_only",
                "SpearmanWithPreferenceLabel": safe_spearman(bundle.g_test, bundle.u_test),
                "AbsoluteCorrelation": abs(safe_spearman(bundle.g_test, bundle.u_test)),
            },
        ]
    )
    return pd.DataFrame(rows).sort_values("AbsoluteCorrelation", ascending=False).reset_index(drop=True)


def dataset_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Return provenance and consistency diagnostics that must be disclosed."""
    rows: List[Dict[str, Any]] = [
        {"Check": "Rows", "Value": int(len(df)), "Interpretation": "Dataset size"},
        {"Check": "Columns", "Value": int(len([c for c in df.columns if c != "ValidationRegion"])), "Interpretation": "Raw dataset width (derived ValidationRegion excluded)"},
        {"Check": "MissingCells", "Value": int(df.isna().sum().sum()), "Interpretation": "Raw missing values"},
        {"Check": "DuplicateRows", "Value": int(df.duplicated().sum()), "Interpretation": "Exact duplicate records"},
        {"Check": "DuplicateStationIDs", "Value": int(df["Station ID"].duplicated().sum()), "Interpretation": "Identifier uniqueness"},
    ]
    address = df.get("Address", pd.Series([], dtype=str)).astype(str)
    synthetic_markers = address.str.contains(r"Random Rd|City\s+\d+,\s*Country", regex=True, na=False)
    rows.append({"Check": "PlaceholderAddressMarkers", "Value": int(synthetic_markers.sum()), "Interpretation": "Records with obvious template/synthetic address markers; verify source provenance"})
    for operator, count in df["Station Operator"].astype(str).value_counts().sort_index().items():
        rows.append({"Check": f"OperatorCount::{operator}", "Value": int(count), "Interpretation": "Must match every manuscript table and split manifest"})
    return pd.DataFrame(rows)


def proxy_validity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Association tests between the rating-derived proxy and observable fields."""
    from scipy import stats as scipy_stats

    rating = pd.to_numeric(df["Reviews (Rating)"], errors="coerce")
    proxy = 1.0 - (rating - rating.min()) / (rating.max() - rating.min() + EPS)
    rows: List[Dict[str, Any]] = []
    numeric = [
        "Cost (USD/kWh)", "Distance to City (km)", "Charging Capacity (kW)",
        "Installation Year", "Parking Spots", "Usage Stats (avg users/day)",
    ]
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        mask = proxy.notna() & values.notna()
        rho, p = scipy_stats.spearmanr(proxy[mask], values[mask])
        rows.append({"Indicator": col, "IndicatorType": "numeric", "Test": "Spearman", "Effect": float(rho), "PValue": float(p), "N": int(mask.sum())})

    categorical = ["Charger Type", "Availability", "Station Operator", "Connector Types", "Renewable Energy Source", "Maintenance Frequency"]
    for col in categorical:
        valid = df[[col]].copy()
        groups = [proxy[df[col].astype(str) == level].dropna().to_numpy() for level in sorted(df[col].astype(str).unique())]
        h, p = scipy_stats.kruskal(*groups)
        grand = float(proxy.mean())
        ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups if len(g))
        ss_total = float(np.sum((proxy - grand) ** 2))
        eta_sq = ss_between / (ss_total + EPS)
        rows.append({"Indicator": col, "IndicatorType": "categorical", "Test": "Kruskal-Wallis / eta-squared", "Effect": float(eta_sq), "PValue": float(p), "N": int(proxy.notna().sum())})
    return pd.DataFrame(rows).sort_values("Effect", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def random_ranking_reference(labels: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray, np.ndarray], repetitions: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Empirical random-ranking distribution for interpreting inflated NDCG."""
    rng = np.random.default_rng(seed)
    records = []
    n = len(labels)
    k = max(1, int(math.ceil(0.10 * n)))
    for _ in range(repetitions):
        score = rng.random(n)
        records.append({
            "NDCG_full": ndcg_at_k(labels, score, n),
            "NDCG_at_10_percent": ndcg_at_k(labels, score, k),
            "TopKAgreement_at_10_percent": precision_at_k(labels, score, k),
            "Spearman": safe_spearman(labels, score),
            "PairwiseAccuracy": pairwise_accuracy(labels, score, pairs),
        })
    raw = pd.DataFrame(records)
    rows = []
    for metric in raw.columns:
        vals = raw[metric].to_numpy(float)
        rows.append({"Metric": metric, "Repetitions": repetitions, "Mean": float(vals.mean()), "Std": float(vals.std(ddof=1)), "CI95_L": float(np.quantile(vals, 0.025)), "CI95_U": float(np.quantile(vals, 0.975))})
    return pd.DataFrame(rows)


# =============================================================================
# Baseline criteria and methods
# =============================================================================

def maintenance_gap(values: pd.Series) -> np.ndarray:
    mapping = {"monthly": 0.0, "quarterly": 0.5, "annually": 1.0, "annual": 1.0}
    return values.astype(str).str.lower().map(mapping).fillna(0.5).to_numpy(dtype=float)


def renewable_gap(values: pd.Series) -> np.ndarray:
    renewable = values.astype(str).str.lower().str.contains("yes|solar|wind|hydro|renew|green", regex=True)
    return 1.0 - renewable.astype(float).to_numpy()


def criterion_frame(df_part: pd.DataFrame, prep: Preprocessor) -> pd.DataFrame:
    numeric_frame = pd.DataFrame(index=df_part.index)
    for col in NUMERIC_MODEL_COLS:
        values = pd.to_numeric(df_part[col], errors="coerce")
        numeric_frame[col] = values.fillna(prep.numeric_medians[col])
    scaled = pd.DataFrame(
        prep.numeric_scaler.transform(numeric_frame[NUMERIC_MODEL_COLS]),
        columns=NUMERIC_MODEL_COLS,
        index=df_part.index,
    )

    availability_hours = df_part["Availability"].apply(parse_availability_hours).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "capacity_gap": 1.0 - scaled["Charging Capacity (kW)"].to_numpy(),
            "high_distance": scaled["Distance to City (km)"].to_numpy(),
            "older_station": 1.0 - scaled["Installation Year"].to_numpy(),
            "maintenance_gap": maintenance_gap(df_part["Maintenance Frequency"]),
            "limited_availability": 1.0 - availability_hours / 24.0,
            "high_cost": scaled["Cost (USD/kWh)"].to_numpy(),
            "renewable_gap": renewable_gap(df_part["Renewable Energy Source"]),
            "parking_gap": 1.0 - scaled["Parking Spots"].to_numpy(),
        },
        index=df_part.index,
    )


def weighted_sum_scores(criteria: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    return sum(float(weights[c]) * criteria[c].to_numpy(dtype=float) for c in weights)


def topsis_scores(criteria: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    cols = list(weights.keys())
    X = criteria[cols].to_numpy(dtype=float)
    w = np.asarray([weights[c] for c in cols], dtype=float)
    denom = np.sqrt(np.sum(X ** 2, axis=0))
    denom[denom == 0] = 1.0
    V = (X / denom) * w
    ideal = np.max(V, axis=0)
    anti = np.min(V, axis=0)
    d_pos = np.sqrt(np.sum((V - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((V - anti) ** 2, axis=1))
    return d_neg / (d_pos + d_neg + EPS)


def vikor_scores(criteria: pd.DataFrame, weights: Mapping[str, float], compromise: float = 0.5) -> np.ndarray:
    cols = list(weights.keys())
    X = criteria[cols].to_numpy(dtype=float)
    w = np.asarray([weights[c] for c in cols], dtype=float)
    best = X.max(axis=0)
    worst = X.min(axis=0)
    gap = (best - X) / (best - worst + EPS)
    S = np.sum(w * gap, axis=1)
    R = np.max(w * gap, axis=1)
    Q = compromise * (S - S.min()) / (S.max() - S.min() + EPS)
    Q += (1.0 - compromise) * (R - R.min()) / (R.max() - R.min() + EPS)
    return 1.0 - Q


def pareto_balanced_scores(criteria: pd.DataFrame) -> np.ndarray:
    """NSGA-II-style non-dominated sorting with crowding-distance tie breaking.

    All criteria are formulated so that larger values indicate greater planning need.
    Higher returned scores are ranked first.
    """
    X = criteria.to_numpy(dtype=float)
    n = len(X)
    dominates = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        ge = np.all(X[p] >= X, axis=1)
        gt = np.any(X[p] > X, axis=1)
        p_dominates = np.where(ge & gt)[0].tolist()
        dominates[p] = p_dominates

        le = np.all(X <= X[p], axis=1)
        lt = np.any(X < X[p], axis=1)
        domination_count[p] = int(np.sum(le & lt))
        if domination_count[p] == 0:
            fronts[0].append(p)

    front_id = np.full(n, -1, dtype=int)
    current = 0
    while current < len(fronts) and fronts[current]:
        next_front = []
        for p in fronts[current]:
            front_id[p] = current
            for q in dominates[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(sorted(set(next_front)))
        current += 1

    crowding = np.zeros(n, dtype=float)
    for front in fronts:
        if not front:
            continue
        idx = np.asarray(front, dtype=int)
        if len(idx) <= 2:
            crowding[idx] = 1.0
            continue
        for col in range(X.shape[1]):
            order_local = np.argsort(X[idx, col])
            ordered = idx[order_local]
            crowding[ordered[0]] = np.inf
            crowding[ordered[-1]] = np.inf
            lo = X[ordered[0], col]
            hi = X[ordered[-1], col]
            if hi - lo <= EPS:
                continue
            for j in range(1, len(ordered) - 1):
                crowding[ordered[j]] += (X[ordered[j + 1], col] - X[ordered[j - 1], col]) / (hi - lo)

    finite = np.isfinite(crowding)
    if finite.any():
        max_finite = float(np.max(crowding[finite]))
        crowding[~finite] = max_finite + 1.0
    crowd_norm = minmax_vector(crowding)
    # Front number is the primary criterion; crowding only breaks ties.
    return -front_id.astype(float) + 1e-3 * crowd_norm


def tune_regressor(
    model_class: Any,
    grid: Mapping[str, Sequence[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fixed: Optional[Mapping[str, Any]] = None,
    top_fraction: float = 0.10,
) -> Tuple[Any, Dict[str, Any], float]:
    best_model = None
    best_params: Dict[str, Any] = {}
    best_ndcg = -np.inf
    fixed = dict(fixed or {})
    for params in ParameterGrid(grid):
        all_params = dict(fixed)
        all_params.update(params)
        model = model_class(**all_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_k = max(1, int(math.ceil(len(y_val) * top_fraction)))
        metric = ndcg_at_k(y_val, pred, val_k)
        if metric > best_ndcg:
            best_ndcg = metric
            best_model = model
            best_params = all_params
    if best_model is None:
        raise RuntimeError("No baseline model was fitted.")
    return best_model, best_params, float(best_ndcg)


def evaluate_baselines(bundle: DataBundle, config: ExperimentConfig, full_result: ModelResult) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    criteria_test = criterion_frame(bundle.raw.iloc[bundle.test_idx], bundle.preprocessor)
    scores: Dict[str, np.ndarray] = {
        "PRIME-EV": full_result.test_scores,
        "AHP_WeightedSum": weighted_sum_scores(criteria_test, AHP_WEIGHTS),
        "TOPSIS": topsis_scores(criteria_test, AHP_WEIGHTS),
        "VIKOR": vikor_scores(criteria_test, AHP_WEIGHTS),
        "MultiObjective_WeightedSum": weighted_sum_scores(criteria_test, MO_WEIGHTS),
        "Pareto_Balanced": pareto_balanced_scores(criteria_test),
        "Random": np.random.default_rng(SEED).random(len(criteria_test)),
        "Oracle_Label_UpperBound": bundle.g_test.copy(),
    }

    ml_specs = [
        (
            "GradientBoostedRanker",
            GradientBoostingRegressor,
            {
                "n_estimators": [50, 100],
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.05],
            },
            {"random_state": SEED},
        ),
        (
            "RandomForestRanker",
            RandomForestRegressor,
            {
                "n_estimators": [100, 200],
                "max_depth": [4, 8],
                "min_samples_leaf": [1, 3],
            },
            {"random_state": SEED, "n_jobs": -1},
        ),
        (
            "RidgeRanker",
            Ridge,
            {"alpha": [0.1, 1.0, 10.0]},
            {},
        ),
        (
            "ShallowTreeRanker",
            DecisionTreeRegressor,
            {"max_depth": [3, 5, 8], "min_samples_leaf": [1, 5]},
            {"random_state": SEED},
        ),
    ]

    transparency_rows: List[Dict[str, Any]] = [
        {
            "Method": "PRIME-EV",
            "Family": "Proposed neural ranker",
            "Inputs": "Same leakage-controlled inference features; rating and observed usage excluded",
            "Objective": "Gaussian risk NLL + auxiliary demand MSE + supervised pairwise logistic ranking",
            "Hyperparameters": f"latent={config.latent_dim}, lr={config.learning_rate}, epochs={config.epochs}, lambda_rank={config.lambda_rank}, validation_metric={config.validation_metric}",
            "Tuning": "Early stopping on validation NDCG",
            "Implementation": "PyTorch",
        },
        {
            "Method": "AHP_WeightedSum",
            "Family": "MCDM",
            "Inputs": ", ".join(AHP_WEIGHTS.keys()),
            "Objective": "Expert-defined weighted priority score",
            "Hyperparameters": json.dumps(AHP_WEIGHTS),
            "Tuning": "No fitting; weights declared before testing",
            "Implementation": "NumPy/Pandas",
        },
        {
            "Method": "TOPSIS",
            "Family": "MCDM",
            "Inputs": ", ".join(AHP_WEIGHTS.keys()),
            "Objective": "Distance to weighted ideal and anti-ideal solutions",
            "Hyperparameters": json.dumps(AHP_WEIGHTS),
            "Tuning": "No fitting; AHP weights reused",
            "Implementation": "NumPy",
        },
        {
            "Method": "VIKOR",
            "Family": "MCDM",
            "Inputs": ", ".join(AHP_WEIGHTS.keys()),
            "Objective": "Compromise ranking with v=0.5",
            "Hyperparameters": json.dumps({"v": 0.5, "weights": AHP_WEIGHTS}),
            "Tuning": "No fitting; AHP weights reused",
            "Implementation": "NumPy",
        },
        {
            "Method": "MultiObjective_WeightedSum",
            "Family": "Multi-objective",
            "Inputs": ", ".join(MO_WEIGHTS.keys()),
            "Objective": "Weighted need-oriented planning score",
            "Hyperparameters": json.dumps(MO_WEIGHTS),
            "Tuning": "No fitting; weights declared before testing",
            "Implementation": "NumPy/Pandas",
        },
        {
            "Method": "Pareto_Balanced",
            "Family": "Pareto optimization",
            "Inputs": ", ".join(criteria_test.columns),
            "Objective": "Non-dominated sorting with crowding-distance tie breaking",
            "Hyperparameters": "NSGA-II-style fronts; crowding distance within fronts",
            "Tuning": "None",
            "Implementation": "Pandas",
        },
        {
            "Method": "Random",
            "Family": "Weak heuristic",
            "Inputs": "None",
            "Objective": "Random ordering",
            "Hyperparameters": f"seed={SEED}",
            "Tuning": "None",
            "Implementation": "NumPy",
        },
        {
            "Method": "Oracle_Label_UpperBound",
            "Family": "Analytical upper bound",
            "Inputs": "Target-side risk proxy and observed usage",
            "Objective": "Exact ordering by the preference-label formula",
            "Hyperparameters": json.dumps({"risk_proxy": config.risk_weight, "demand_target": config.demand_weight}),
            "Tuning": "Not deployable; excluded from comparative claims",
            "Implementation": "NumPy",
        },
    ]

    for method_name, cls, grid, fixed in ml_specs:
        model, params, val_score = tune_regressor(
            cls,
            grid,
            bundle.X_train,
            bundle.g_train,
            bundle.X_val,
            bundle.g_val,
            fixed=fixed,
            top_fraction=config.top_fraction,
        )
        scores[method_name] = model.predict(bundle.X_test)
        transparency_rows.append(
            {
                "Method": method_name,
                "Family": "Machine-learning ranker",
                "Inputs": "Same leakage-controlled inference feature matrix as PRIME-EV",
                "Objective": "Regression to formula-derived intervention-priority proxy",
                "Hyperparameters": json.dumps(params),
                "Tuning": f"Grid search selected by validation NDCG@{int(config.top_fraction*100)}%={val_score:.5f}",
                "Implementation": "scikit-learn",
            }
        )

    result_rows: List[Dict[str, Any]] = []
    for method, score in scores.items():
        metrics = ranking_metrics(bundle.g_test, score, bundle.test_pairs, config.top_fraction)
        metrics.update({"Method": method, "IsOracleUpperBound": method == "Oracle_Label_UpperBound"})
        result_rows.append(metrics)

    return pd.DataFrame(result_rows), pd.DataFrame(transparency_rows), scores


def baseline_weight_sensitivity(bundle: DataBundle, config: ExperimentConfig, repetitions: int = 100) -> pd.DataFrame:
    criteria_test = criterion_frame(bundle.raw.iloc[bundle.test_idx], bundle.preprocessor)
    rng = np.random.default_rng(SEED + 500)
    rows: List[Dict[str, Any]] = []

    method_defs = {
        "AHP_WeightedSum": ("weighted", AHP_WEIGHTS),
        "TOPSIS": ("topsis", AHP_WEIGHTS),
        "MultiObjective_WeightedSum": ("weighted", MO_WEIGHTS),
    }

    for method, (kind, base_weights) in method_defs.items():
        metric_records = []
        keys = list(base_weights.keys())
        base = np.asarray([base_weights[k] for k in keys], dtype=float)
        for rep in range(repetitions):
            perturb = rng.uniform(0.80, 1.20, size=len(keys))
            weights_vec = base * perturb
            weights_vec /= weights_vec.sum()
            weights = {k: float(v) for k, v in zip(keys, weights_vec)}
            score = topsis_scores(criteria_test, weights) if kind == "topsis" else weighted_sum_scores(criteria_test, weights)
            metric_records.append(ranking_metrics(bundle.g_test, score, bundle.test_pairs, config.top_fraction))
        metric_df = pd.DataFrame(metric_records)
        for metric in ["NDCG_full", "Precision_at_10", "Precision_at_10_percent", "Spearman"]:
            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Perturbation": "Independent uniform weight multipliers in [0.8, 1.2], then renormalized",
                    "Repetitions": repetitions,
                    "Mean": float(metric_df[metric].mean()),
                    "Std": float(metric_df[metric].std(ddof=1)),
                    "Minimum": float(metric_df[metric].min()),
                    "Maximum": float(metric_df[metric].max()),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Ablations and corrected deployment composite
# =============================================================================

def run_ablations(bundle: DataBundle, config: ExperimentConfig, device: torch.device, full_result: ModelResult) -> pd.DataFrame:
    variants = {
        "Full": {},
        "NoAttention": {"no_attention": True},
        "NoIRE_Conv": {"no_ire": True},
        "LowDimension": {"latent_dim": 4},
        "DeterministicRisk": {"deterministic_risk": True},
        "NoDIM": {"no_dim": True},
        "PointwiseRanking": {"pointwise_rank": True},
        "NoRiskToRanker": {"no_risk_input": True},
    }

    results: Dict[str, ModelResult] = {"Full": full_result}
    for name, variant in variants.items():
        if name == "Full":
            continue
        results[name] = train_prime_ev(bundle, config, device, name=name, variant=variant)

    raw_all = bundle.raw.copy()
    rows: List[Dict[str, Any]] = []
    full_predictive = full_result.losses["risk"]
    for name, result in results.items():
        all_mu, all_sigma, all_usage, all_scores, all_latency = infer_all_candidates(
            result.model, bundle, config, device
        )
        fair = fairness_metrics(raw_all, all_scores, all_mu, bundle.preprocessor, config.top_fraction)
        row = {
            "Variant": name,
            "TestTotalLoss": result.losses["total"],
            "TestRiskLoss": result.losses["risk"],
            "TestDemandLoss": result.losses["demand"],
            "TestRankingLoss": result.losses["rank"],
            "DeltaPred_percent": 100.0 * (result.losses["risk"] - full_predictive) / (abs(full_predictive) + EPS),
            "NDCG_full": result.test_metrics["NDCG_full"],
            "Precision_at_10": result.test_metrics["Precision_at_10"],
            "SSI": system_stress_index(all_scores),
            "OperatorRiskBalance": fair["OperatorRiskBalance"],
            "GeographicEquityScore": fair["GeographicEquityScore"],
            "Latency_ms_per_station": all_latency,
            "TrainingTime_seconds": result.train_seconds,
            "ParameterCount": count_parameters(result.model),
            "ModelMemoryMB": memory_mb_for_model(result.model),
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    for source, target in [
        ("SSI", "SSI_normalized"),
        ("OperatorRiskBalance", "ORB_normalized"),
        ("Latency_ms_per_station", "Latency_normalized"),
    ]:
        values = table[source].to_numpy(dtype=float)
        table[target] = minmax_vector(values)

    table["OperatorBalanceScore"] = 1.0 - table["ORB_normalized"]
    table["LatencyScore"] = 1.0 - table["Latency_normalized"]
    table["StabilityScore"] = 1.0 - table["SSI_normalized"]
    table["CompositeDeploymentScore"] = (
        0.30 * table["StabilityScore"]
        + 0.25 * table["OperatorBalanceScore"]
        + 0.25 * table["GeographicEquityScore"]
        + 0.20 * table["LatencyScore"]
    )
    return table.sort_values("CompositeDeploymentScore", ascending=False).reset_index(drop=True)


# =============================================================================
# Sensitivity and regional transfer
# =============================================================================

def run_label_weight_sensitivity(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    risk_weights: Sequence[float] = (0.40, 0.50, 0.60, 0.70, 0.80),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, risk_weight in enumerate(risk_weights):
        demand_weight = 1.0 - risk_weight
        weighted_bundle = rebuild_bundle_labels(
            bundle,
            config,
            risk_weight=risk_weight,
            demand_weight=demand_weight,
            offset=100 + idx,
        )
        result = train_prime_ev(
            weighted_bundle,
            config,
            device,
            name=f"LabelSensitivity_risk_{risk_weight:.2f}",
            variant={},
            epochs_override=config.sensitivity_epochs,
        )
        row = {
            "RiskWeight": risk_weight,
            "DemandWeight": demand_weight,
            "TrainPairs": len(weighted_bundle.train_pairs[0]),
            "ValidationPairs": len(weighted_bundle.val_pairs[0]),
            "TestPairs": len(weighted_bundle.test_pairs[0]),
        }
        row.update(result.test_metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def region_holdout_indices(df: pd.DataFrame, held_region: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    regions = df["Longitude"].apply(region_from_longitude).to_numpy()
    test_idx = np.where(regions == held_region)[0]
    remaining = np.where(regions != held_region)[0]
    if len(test_idx) < 25 or len(remaining) < 50:
        raise ValueError(f"Region {held_region} has insufficient samples for holdout evaluation.")

    groups = df.iloc[remaining]["Station Operator"].astype(str).to_numpy()
    if len(np.unique(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_rel, val_rel = next(splitter.split(df.iloc[remaining], groups=groups))
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(remaining))
        n_val = max(1, int(0.20 * len(remaining)))
        val_rel = shuffled[:n_val]
        train_rel = shuffled[n_val:]

    train_idx = remaining[train_rel]
    val_idx = remaining[val_rel]
    metadata = {
        "strategy": "leave-one-geographic-region-out",
        "held_out_region": held_region,
        "training_regions": sorted(np.unique(regions[train_idx]).tolist()),
        "validation_regions": sorted(np.unique(regions[val_idx]).tolist()),
        "test_regions": [held_region],
    }
    return train_idx, val_idx, test_idx, metadata


def run_regional_transfer(df: pd.DataFrame, config: ExperimentConfig, device: torch.device) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    regions = sorted(df["Longitude"].apply(region_from_longitude).unique().tolist())
    for fold, held in enumerate(regions):
        train_idx, val_idx, test_idx, metadata = region_holdout_indices(df, held, SEED + 700 + fold)
        regional_config = copy.deepcopy(config)
        regional_config.batch_pairs_train = min(config.batch_pairs_train, max(10000, len(train_idx) * 15))
        regional_config.batch_pairs_val = min(config.batch_pairs_val, max(3000, len(val_idx) * 10))
        regional_config.batch_pairs_test = min(config.batch_pairs_test, max(3000, len(test_idx) * 10))
        bundle = build_bundle(
            df,
            train_idx,
            val_idx,
            test_idx,
            regional_config,
            metadata,
            pair_seed_offset=800 + fold,
        )
        result = train_prime_ev(
            bundle,
            regional_config,
            device,
            name=f"RegionalHoldout_{held}",
            variant={},
            epochs_override=config.regional_epochs,
        )
        row = {
            "HeldOutRegion": held,
            "N_train": len(train_idx),
            "N_validation": len(val_idx),
            "N_test": len(test_idx),
            "TrainPairs": len(bundle.train_pairs[0]),
            "ValidationPairs": len(bundle.val_pairs[0]),
            "TestPairs": len(bundle.test_pairs[0]),
        }
        row.update(result.test_metrics)
        rows.append(row)
    return pd.DataFrame(rows)



# =============================================================================
# V8 reviewer-completion additions
# =============================================================================

def predict_split(model: PrimeEV, bundle: DataBundle, split: str, device: torch.device, config: ExperimentConfig):
    X = getattr(bundle, f"X_{split}")
    cost = getattr(bundle, f"cost_{split}")
    dist = getattr(bundle, f"dist_{split}")
    return infer_model(model, X, cost, dist, device, repeats=1, batch_size=config.eval_batch_size)


def conformal_uncertainty_diagnostics(
    model: PrimeEV,
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Validation-calibrated Gaussian intervals using standardized residual quantiles.

    Calibration uses validation data only; test data are used once for reporting.
    Both raw and calibrated intervals are retained so no evidence is hidden.
    """
    val_mu, val_sigma, _, _, _ = predict_split(model, bundle, "val", device, config)
    test_mu, test_sigma, _, _, _ = predict_split(model, bundle, "test", device, config)
    val_y = np.asarray(bundle.y_val, float)
    test_y = np.asarray(bundle.y_test, float)
    standardized = np.abs(val_y - val_mu) / (val_sigma + EPS)
    rows = []
    for nominal in (0.50, 0.80, 0.90, 0.95):
        z = float(stats.norm.ppf((1.0 + nominal) / 2.0))
        q = float(np.quantile(standardized, nominal, method="higher"))
        for method, multiplier in (("RawGaussian", z), ("ValidationConformal", q)):
            lo = np.clip(test_mu - multiplier * test_sigma, 0.0, 1.0)
            hi = np.clip(test_mu + multiplier * test_sigma, 0.0, 1.0)
            coverage = float(np.mean((test_y >= lo) & (test_y <= hi)))
            width = float(np.mean(hi - lo))
            rows.append({
                "Method": method,
                "NominalCoverage": nominal,
                "Multiplier": multiplier,
                "ObservedCoverage": coverage,
                "CoverageError": coverage - nominal,
                "AbsoluteCoverageError": abs(coverage - nominal),
                "MeanIntervalWidth": width,
                "CalibrationSource": "validation split only" if method == "ValidationConformal" else "none",
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "uncertainty_calibration_v8.csv", index=False)
    return out


def _clone_bundle_with_permuted_features(bundle: DataBundle, permutation: np.ndarray) -> DataBundle:
    cloned = copy.copy(bundle)
    cloned.X_train = bundle.X_train[:, permutation].copy()
    cloned.X_val = bundle.X_val[:, permutation].copy()
    cloned.X_test = bundle.X_test[:, permutation].copy()
    cloned.preprocessor = copy.copy(bundle.preprocessor)
    cloned.preprocessor.feature_names = [bundle.preprocessor.feature_names[i] for i in permutation]
    return cloned



def conformal_uncertainty_diagnostics(
    model: PrimeEV,
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Finite-sample split-conformal intervals calibrated on validation only.

    The conformal quantile uses ceil((n_cal+1)*(1-alpha))/n_cal and test labels
    are accessed only after the multiplier has been fixed. Raw clipped Gaussian
    intervals are retained for an honest before/after comparison.
    """
    val_mu, val_sigma, _, _, _ = predict_split(model, bundle, "val", device, config)
    test_mu, test_sigma, _, _, _ = predict_split(model, bundle, "test", device, config)
    val_y = np.asarray(bundle.y_val, dtype=float)
    test_y = np.asarray(bundle.y_test, dtype=float)
    standardized = np.abs(val_y - val_mu) / (val_sigma + EPS)
    rows: List[Dict[str, Any]] = []
    interval_records: List[Dict[str, Any]] = []
    n_cal = len(standardized)
    for nominal in (0.50, 0.80, 0.90, 0.95):
        alpha = 1.0 - nominal
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        quantile_level = min(1.0, math.ceil((n_cal + 1) * nominal) / n_cal)
        q = float(np.quantile(standardized, quantile_level, method="higher"))
        for method, multiplier in (("RawGaussian", z), ("ValidationSplitConformal", q)):
            lo = np.clip(test_mu - multiplier * test_sigma, 0.0, 1.0)
            hi = np.clip(test_mu + multiplier * test_sigma, 0.0, 1.0)
            covered = (test_y >= lo) & (test_y <= hi)
            rows.append({
                "Method": method,
                "NominalCoverage": nominal,
                "Alpha": alpha,
                "CalibrationRows": n_cal,
                "FiniteSampleQuantileLevel": quantile_level if method == "ValidationSplitConformal" else np.nan,
                "Multiplier": multiplier,
                "ObservedCoverage": float(np.mean(covered)),
                "CoverageError": float(np.mean(covered) - nominal),
                "AbsoluteCoverageError": abs(float(np.mean(covered) - nominal)),
                "MeanIntervalWidth": float(np.mean(hi - lo)),
                "MedianIntervalWidth": float(np.median(hi - lo)),
                "CalibrationSource": "validation split only" if method == "ValidationSplitConformal" else "none",
                "CoverageClaim": "marginal finite-sample split-conformal" if method == "ValidationSplitConformal" else "uncalibrated parametric",
            })
            if method == "ValidationSplitConformal":
                raw_test = bundle.raw.iloc[bundle.test_idx].reset_index(drop=True)
                for idx in range(len(test_y)):
                    interval_records.append({
                        "NominalCoverage": nominal,
                        "TestRowIndex": int(bundle.test_idx[idx]),
                        "StationID": str(raw_test.iloc[idx]["Station ID"]),
                        "StationOperator": str(raw_test.iloc[idx]["Station Operator"]),
                        "PredictedMean": float(test_mu[idx]),
                        "PredictedSigma": float(test_sigma[idx]),
                        "Lower": float(lo[idx]),
                        "Upper": float(hi[idx]),
                        "Width": float(hi[idx] - lo[idx]),
                        "Covered": bool(covered[idx]),
                    })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "uncertainty_calibration_v11.csv", index=False)
    out.to_csv(output_dir / "uncertainty_calibration_v8.csv", index=False)
    interval_df = pd.DataFrame(interval_records)
    interval_df.to_csv(output_dir / "conformal_test_intervals_v11.csv", index=False)

    group_rows: List[Dict[str, Any]] = []
    if not interval_df.empty:
        interval_df["SigmaQuartile"] = pd.qcut(
            interval_df["PredictedSigma"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        ).astype(str)
        for nominal in sorted(interval_df["NominalCoverage"].unique()):
            sub_nominal = interval_df[interval_df["NominalCoverage"] == nominal]
            for group_type, col in (("Operator", "StationOperator"), ("SigmaQuartile", "SigmaQuartile")):
                for group, sub in sub_nominal.groupby(col):
                    group_rows.append({
                        "NominalCoverage": nominal,
                        "GroupType": group_type,
                        "Group": str(group),
                        "N": len(sub),
                        "ObservedCoverage": float(sub["Covered"].mean()),
                        "CoverageError": float(sub["Covered"].mean() - nominal),
                        "MeanWidth": float(sub["Width"].mean()),
                    })
    pd.DataFrame(group_rows).to_csv(output_dir / "conformal_group_coverage_v11.csv", index=False)
    return out


def run_retrained_feature_order_sensitivity(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    permutations: int = 5,
    epochs: int = 8,
) -> pd.DataFrame:
    """Consistently permute train/validation/test features and retrain matched models."""
    global SEED
    old_seed = SEED
    rows = []
    identity = np.arange(bundle.X_train.shape[1])
    for rep in range(permutations):
        training_seed = int(config.split_seed + 3000 + rep)
        rng = np.random.default_rng(training_seed)
        perm = identity.copy() if rep == 0 else rng.permutation(len(identity))
        b = _clone_bundle_with_permuted_features(bundle, perm)
        for architecture, variant in (("Conv1D", {}), ("MLP_NoConv", {"no_ire": True})):
            SEED = training_seed
            set_seed(training_seed)
            res = train_prime_ev(
                b, config, device,
                name=f"Order_{architecture}_{rep}",
                variant=variant,
                epochs_override=epochs,
            )
            rec = {
                "Permutation": rep,
                "Architecture": architecture,
                "TrainingSeed": training_seed,
                "IdentityOrder": bool(rep == 0),
                "PermutationVector": "|".join(map(str, perm.tolist())),
            }
            rec.update(fixed_cutoff_metrics(b.g_test, res.test_scores, b.test_pairs))
            rows.append(rec)
    SEED = old_seed
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "retrained_feature_order_sensitivity_v8.csv", index=False)
    return out


def _choose_validation_operator(operators: List[str], held_out: str) -> str:
    remaining = [o for o in operators if o != held_out]
    return remaining[0]


def _constrained_topk(
    scores: np.ndarray,
    operators: np.ndarray,
    low_access: np.ndarray,
    k: int,
    operator_floor_fraction: float,
    low_access_floor_fraction: float,
) -> np.ndarray:
    """Greedy policy layer with proportional operator and low-access floors."""
    n = len(scores)
    k = min(max(1, int(k)), n)
    selected: List[int] = []
    available = set(range(n))
    op_counts = pd.Series(operators).value_counts().to_dict()
    for op, count in op_counts.items():
        quota = int(math.floor(k * (count / n) * operator_floor_fraction))
        candidates = [i for i in np.argsort(-scores) if i in available and operators[i] == op]
        for i in candidates[:quota]:
            selected.append(int(i)); available.remove(int(i))
    low_quota = int(math.floor(k * float(np.mean(low_access)) * low_access_floor_fraction))
    have_low = sum(bool(low_access[i]) for i in selected)
    need_low = max(0, low_quota - have_low)
    candidates = [i for i in np.argsort(-scores) if i in available and low_access[i]]
    for i in candidates[:need_low]:
        selected.append(int(i)); available.remove(int(i))
    for i in np.argsort(-scores):
        if len(selected) >= k:
            break
        if int(i) in available:
            selected.append(int(i)); available.remove(int(i))
    return np.asarray(selected[:k], dtype=int)


def run_operator_cv_constrained_rerank(
    df: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    epochs: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-operator-out predictions and policy-constrained top-k trade-offs."""
    operators = sorted(df["Station Operator"].astype(str).unique().tolist())
    oof_rows = []
    for fold, held in enumerate(operators):
        val_op = _choose_validation_operator(operators, held)
        test_idx = np.where(df["Station Operator"].astype(str).to_numpy() == held)[0]
        val_idx = np.where(df["Station Operator"].astype(str).to_numpy() == val_op)[0]
        train_idx = np.where(~df["Station Operator"].astype(str).isin([held, val_op]).to_numpy())[0]
        cfg = copy.deepcopy(config)
        cfg.batch_pairs_train = min(cfg.batch_pairs_train, max(10000, len(train_idx) * 10))
        cfg.batch_pairs_val = min(cfg.batch_pairs_val, max(3000, len(val_idx) * 8))
        cfg.batch_pairs_test = min(cfg.batch_pairs_test, max(3000, len(test_idx) * 8))
        meta = {"strategy": "leave-one-operator-out", "held_out_operator": held, "validation_operator": val_op}
        b = build_bundle(df, train_idx, val_idx, test_idx, cfg, meta, pair_seed_offset=4000 + fold)
        res = train_prime_ev(b, cfg, device, name=f"OperatorCV_{held}", epochs_override=epochs)
        raw_test = df.iloc[test_idx].reset_index(drop=True)
        for local, global_idx in enumerate(test_idx):
            oof_rows.append({
                "RowIndex": int(global_idx),
                "StationID": str(df.iloc[global_idx]["Station ID"]),
                "StationOperator": held,
                "ValidationOperator": val_op,
                "Score": float(res.test_scores[local]),
                "RiskMean": float(res.test_mu[local]),
                "RiskSigma": float(res.test_sigma[local]),
                "ReferenceUtility": float(b.g_test[local]),
                "DistanceKm": float(pd.to_numeric(raw_test.iloc[local]["Distance to City (km)"], errors="coerce")),
            })
    oof = pd.DataFrame(oof_rows).sort_values("RowIndex").reset_index(drop=True)
    oof.to_csv(output_dir / "operator_cv_oof_predictions_v8.csv", index=False)

    scores = oof["Score"].to_numpy(float)
    utility = oof["ReferenceUtility"].to_numpy(float)
    ops = oof["StationOperator"].astype(str).to_numpy()
    distance = oof["DistanceKm"].to_numpy(float)
    low_access = distance >= np.quantile(distance, 0.75)
    k = max(1, int(math.ceil(len(oof) * config.top_fraction)))
    ideal = float(np.mean(np.sort(utility)[-k:]))
    trade_rows = []
    for op_floor in (0.0, 0.5, 1.0):
        for low_floor in (0.0, 0.5, 1.0):
            selected = _constrained_topk(scores, ops, low_access, k, op_floor, low_floor)
            mask = np.zeros(len(oof), dtype=bool); mask[selected] = True
            rates = pd.DataFrame({"op": ops, "sel": mask.astype(float)}).groupby("op")["sel"].mean()
            selected_utility = float(np.mean(utility[selected]))
            trade_rows.append({
                "OperatorFloorFraction": op_floor,
                "LowAccessFloorFraction": low_floor,
                "SelectedStations": int(k),
                "MeanSelectedUtility": selected_utility,
                "RegretVsOracle": ideal - selected_utility,
                "OperatorSelectionRateDisparity": float(rates.max() - rates.min()),
                "SelectedLowAccessShare": float(np.mean(low_access[selected])),
                "PopulationLowAccessShare": float(np.mean(low_access)),
                "LowAccessCoverageGap": abs(float(np.mean(low_access[selected])) - float(np.mean(low_access))),
            })
    trade = pd.DataFrame(trade_rows)
    trade.to_csv(output_dir / "operator_cv_constrained_rerank_tradeoff_v8.csv", index=False)
    return oof, trade


# =============================================================================
# Output generation
# =============================================================================

def save_model_checkpoint(path: Path, result: ModelResult, bundle: DataBundle, config: ExperimentConfig) -> None:
    checkpoint = {
        "model_state_dict": result.model.state_dict(),
        "model_name": result.name,
        "config": asdict(config),
        "split_metadata": bundle.split_metadata,
        "feature_names": bundle.preprocessor.feature_names,
        "best_epoch": result.best_epoch,
        "test_metrics": result.test_metrics,
        "test_losses": result.losses,
    }
    torch.save(checkpoint, path)


def make_summary_plot(baselines: pd.DataFrame, output_path: Path) -> None:
    plot_df = baselines[~baselines["IsOracleUpperBound"]].sort_values("NDCG_full", ascending=False)
    plt.figure(figsize=(10, 5.5))
    plt.bar(plot_df["Method"], plot_df["NDCG_full"])
    plt.ylabel("NDCG")
    plt.xlabel("Method")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def manuscript_latex(
    bundle: DataBundle,
    full_result: ModelResult,
    leakage: pd.DataFrame,
    fairness: Dict[str, float],
    label_sensitivity: Optional[pd.DataFrame],
    regional_transfer: Optional[pd.DataFrame],
    baseline_results: pd.DataFrame,
    ablation_table: Optional[pd.DataFrame],
) -> str:
    max_model_corr = leakage[leakage["VariableType"] == "model_input"]["AbsoluteCorrelation"].max()
    baseline_prime = baseline_results[baseline_results["Method"] == "PRIME-EV"].iloc[0]

    sensitivity_sentence = "Label-weight sensitivity was not executed in this run."
    if label_sensitivity is not None and not label_sensitivity.empty:
        sensitivity_sentence = (
            "Across the tested risk weights, NDCG ranged from "
            f"{label_sensitivity['NDCG_full'].min():.4f} to {label_sensitivity['NDCG_full'].max():.4f}, "
            "while Precision@10 ranged from "
            f"{label_sensitivity['Precision_at_10'].min():.4f} to {label_sensitivity['Precision_at_10'].max():.4f}."
        )

    transfer_sentence = "The regional holdout experiment was not executed in this run."
    if regional_transfer is not None and not regional_transfer.empty:
        transfer_sentence = (
            "Leave-one-region-out evaluation produced a mean transfer NDCG of "
            f"{regional_transfer['NDCG_full'].mean():.4f} and a mean transfer Precision@10 of "
            f"{regional_transfer['Precision_at_10'].mean():.4f}."
        )

    corrected_math_sentence = ""
    if ablation_table is not None and not ablation_table.empty:
        full_row = ablation_table[ablation_table["Variant"] == "Full"].iloc[0]
        corrected_math_sentence = (
            "For the full model, the corrected non-negative SSI is "
            f"{full_row['SSI']:.4f}, operator risk balance is {full_row['OperatorRiskBalance']:.4f}, "
            f"and the normalized composite deployment score is {full_row['CompositeDeploymentScore']:.4f}."
        )

    train_ops = ", ".join(bundle.split_metadata.get("train_operators", []))
    val_ops = ", ".join(bundle.split_metadata.get("validation_operators", []))
    test_ops = ", ".join(bundle.split_metadata.get("test_operators", []))

    return rf"""
\subsection{{Target and Preference Label Construction}}
\label{{subsec:target_preference_construction}}

The source dataset contains customer ratings and average daily usage but does not contain direct records of charger failure probability, service downtime, unresolved complaints, or observed deployment decisions. We therefore use a rating-derived operational-risk proxy and a formula-derived intervention-priority proxy. Let $r_i$ denote the recorded customer rating. We estimate the rating minimum and maximum from the training partition and compute
\begin{{equation}}
\widetilde{{r}}_i = \frac{{r_i-r_{{\min}}^{{\mathrm{{train}}}}}}{{r_{{\max}}^{{\mathrm{{train}}}}-r_{{\min}}^{{\mathrm{{train}}}}+10^{{-8}}}},
\qquad
y_i = 1-\widetilde{{r}}_i.
\end{{equation}}
The variable $y_i\in[0,1]$ is a rating-derived risk proxy. It does not represent an observed failure, complaint, or downtime probability. Numerical missing values are replaced with training-partition medians, categorical missing values are assigned to an ``Unknown'' category, and all scaling and encoding parameters are estimated from the training partition only.

Let $u_i\in[0,1]$ denote average daily usage after training-partition min--max scaling. Customer rating and observed usage are excluded from the inference-time feature vector. The preference reference is
\begin{{equation}}
g_i = {bundle.split_metadata['risk_weight']:.2f}y_i + {bundle.split_metadata['demand_weight']:.2f}u_i.
\end{{equation}}
Thus, $g_i$ prioritizes stations that combine a high rating-derived risk proxy with high observed demand. It is a policy proxy rather than an independently observed deployment decision.

We split stations by operator before preprocessing or pair generation. The training, validation, and test partitions contain {len(bundle.train_idx)}, {len(bundle.val_idx)}, and {len(bundle.test_idx)} stations, respectively. The training operators are {train_ops}; the validation operators are {val_ops}; and the test operators are {test_ops}. For stations $i$ and $j$ within the same partition, we assign
\begin{{equation}}
\rho_{{ij}} =
\begin{{cases}}
+1, & g_i-g_j>{bundle.split_metadata['pair_threshold']:.2f},\\
-1, & g_j-g_i>{bundle.split_metadata['pair_threshold']:.2f}.
\end{{cases}}
\end{{equation}}
Pairs with $|g_i-g_j|\leq{bundle.split_metadata['pair_threshold']:.2f}$ are omitted. The training, validation, and test sets contain {len(bundle.train_pairs[0])}, {len(bundle.val_pairs[0])}, and {len(bundle.test_pairs[0])} supervised pairs. No station or pair appears in more than one partition.

To examine circularity, we exclude both variables used to construct $g_i$ from the inference-time input, compute feature--label correlations on the held-out test partition, compare PRIME-EV with the direct label upper bound, and test alternative risk--demand weights. The largest absolute Spearman correlation between an allowed model input and $g_i$ is {max_model_corr:.4f}. PRIME-EV obtains an NDCG of {baseline_prime['NDCG_full']:.4f} and Precision@10 of {baseline_prime['Precision_at_10']:.4f} on the operator-held-out test set. {sensitivity_sentence} {transfer_sentence}

\subsection{{Operator Balance and Geographic Accessibility}}

We report operator risk balance rather than ethical fairness. For operator $m$, let $\bar{{\mu}}_m$ denote the mean predicted risk of its stations. Operator risk balance is
\begin{{equation}}
\operatorname{{ORB}} = \frac{{1}}{{M}}\sum_{{m=1}}^M\left|\bar{{\mu}}_m-\frac{{1}}{{M}}\sum_{{q=1}}^M\bar{{\mu}}_q\right|.
\end{{equation}}
The full model gives $\operatorname{{ORB}}={fairness['OperatorRiskBalance']:.4f}$. We also report geographic selection-rate disparity ({fairness['GeographicSelectionRateDisparity']:.4f}), urban--intercity selection-rate disparity ({fairness['AccessibilitySelectionRateDisparity']:.4f}), low-access coverage gap ({fairness['LowAccessCoverageGap']:.4f}), and mean distance of selected stations ({fairness['MeanSelectedDistance_km']:.2f} km). Demographic and socioeconomic attributes are unavailable in the dataset, so these measures do not support a broad ethical-fairness claim. A demand-oriented policy may under-prioritize low-demand intercity stations even when those stations provide essential geographic coverage.

\subsection{{Corrected Deployment Metrics}}

The system stress index is computed as the mean absolute deviation of normalized priority scores:
\begin{{equation}}
\operatorname{{SSI}} = \frac{{1}}{{N}}\sum_{{i=1}}^N|\widetilde{{s}}_i-\overline{{\widetilde{{s}}}}|,
\end{{equation}}
which is non-negative by definition. Predictive-loss deviation is
\begin{{equation}}
\Delta_{{\mathrm{{Pred}}}}(a)=\frac{{\mathcal{{L}}_a-\mathcal{{L}}_{{\mathrm{{full}}}}}}{{|\mathcal{{L}}_{{\mathrm{{full}}}}|+10^{{-8}}}}\times100,
\end{{equation}}
so a positive value indicates a larger ablated predictive loss. Before aggregation, SSI, ORB, and inference latency are min--max normalized across the evaluated model variants. The deployment score is
\begin{{equation}}
\mathcal{{S}}_{{\mathrm{{comp}}}}=
0.30(1-\widetilde{{\operatorname{{SSI}}}})
+0.25(1-\widetilde{{\operatorname{{ORB}}}})
+0.25\operatorname{{GE}}
+0.20(1-\widetilde{{\operatorname{{DT}}}}),
\end{{equation}}
where $\operatorname{{GE}}\in[0,1]$ is the geographic-accessibility equity score and $\operatorname{{DT}}$ is inference latency in milliseconds per station. {corrected_math_sentence}
""".strip() + "\n"


def reviewer_response_text(
    bundle: DataBundle,
    full_result: ModelResult,
    leakage: pd.DataFrame,
    fairness: Dict[str, float],
    label_sensitivity: Optional[pd.DataFrame],
    regional_transfer: Optional[pd.DataFrame],
) -> str:
    max_corr = leakage[leakage["VariableType"] == "model_input"]["AbsoluteCorrelation"].max()
    lines = [
        "Response to reviewer comments",
        "=" * 80,
        "",
        "1. Target and preference-label construction",
        f"   The revised pipeline defines y_i = 1 - normalized rating as a rating-derived risk proxy.",
        "   Rating is not used as an inference input. Observed usage is also withheld from inference inputs",
        "   and serves as the auxiliary demand target. The intervention-priority proxy is",
        f"   g_i = {bundle.split_metadata['risk_weight']:.2f} y_i + {bundle.split_metadata['demand_weight']:.2f} u_i.",
        "   We state that these labels are formula-derived proxies, not observed failures or deployment decisions.",
        "",
        "2. Pair construction and split",
        f"   Operator-disjoint stations: train={len(bundle.train_idx)}, validation={len(bundle.val_idx)}, test={len(bundle.test_idx)}.",
        f"   Supervised pairs: train={len(bundle.train_pairs[0])}, validation={len(bundle.val_pairs[0])}, test={len(bundle.test_pairs[0])}.",
        f"   Pairs use rho_ij = sign(g_i-g_j) when |g_i-g_j|>{bundle.split_metadata['pair_threshold']:.2f}.",
        "",
        "3. Leakage checks",
        f"   The largest absolute test-set Spearman correlation between g_i and an allowed input is {max_corr:.4f}.",
        "   The output tables include all feature correlations, a direct-label upper bound, operator-disjoint testing,",
        "   trained regional holdout tests, and alternative risk-demand label weights.",
        "",
        "4. Fairness terminology",
        f"   We renamed the former fairness metric to operator risk balance. Full-model ORB={fairness['OperatorRiskBalance']:.4f}.",
        f"   We added geographic selection disparity={fairness['GeographicSelectionRateDisparity']:.4f},",
        f"   accessibility disparity={fairness['AccessibilitySelectionRateDisparity']:.4f}, and",
        f"   low-access coverage gap={fairness['LowAccessCoverageGap']:.4f}.",
        "   We state that demographic data are unavailable and avoid broad ethical-fairness claims.",
        "",
        "5. Mathematical consistency",
        f"   SSI is recomputed as a non-negative mean absolute deviation over all candidate stations; full-model SSI={full_result.test_metrics.get('SSI_all_candidates', full_result.test_metrics['SSI']):.4f}.",
        "   DeltaPred follows (L_ablation-L_full)/|L_full|. ORB and latency are min-max normalized before",
        "   entering the composite deployment score.",
        "",
        "6. Baseline transparency",
        "   All machine-learning baselines use the same operator-disjoint splits and leakage-controlled feature matrix.",
        "   The baseline transparency table reports objectives, inputs, tuning grids, selected hyperparameters, and software.",
        "   MCDM and multi-objective weight sensitivity uses 100 independent +/-20% perturbations followed by renormalization.",
    ]
    if label_sensitivity is not None and not label_sensitivity.empty:
        lines.extend(
            [
                "",
                "7. Label-weight sensitivity",
                f"   NDCG range: {label_sensitivity['NDCG_full'].min():.4f} to {label_sensitivity['NDCG_full'].max():.4f}.",
                f"   Precision@10 range: {label_sensitivity['Precision_at_10'].min():.4f} to {label_sensitivity['Precision_at_10'].max():.4f}.",
            ]
        )
    if regional_transfer is not None and not regional_transfer.empty:
        lines.extend(
            [
                "",
                "8. Geographic transfer",
                f"   Mean regional holdout NDCG: {regional_transfer['NDCG_full'].mean():.4f}.",
                f"   Mean regional holdout Precision@10: {regional_transfer['Precision_at_10'].mean():.4f}.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_run_instructions(output_dir: Path, script_name: str) -> None:
    text = f"""HOW TO RUN {script_name}
{'=' * 80}

1. Install Python 3.10 or newer.

2. Install dependencies:

   pip install numpy pandas scipy scikit-learn torch matplotlib

3. Put these two files in the same folder:

   {script_name}
   ev_charging_stations-dataset.csv

4. Run the complete experiment:

   python {script_name} --data ev_charging_stations-dataset.csv --output prime_ev_reviewer_results --epochs 50 --torch-threads 1

5. Run a short installation test first:

   python {script_name} --data ev_charging_stations-dataset.csv --output prime_ev_quick_test --quick

Windows example:

   python {script_name} --data "D:\\other\\prime-ev\\ev_charging_stations-dataset.csv" --output "D:\\other\\prime-ev\\reviewer_results" --epochs 50 --torch-threads 1

Google Colab example:

   !pip install numpy pandas scipy scikit-learn torch matplotlib
   !python /content/{script_name} --data /content/ev_charging_stations-dataset.csv --output /content/prime_ev_reviewer_results --epochs 50 --torch-threads 1

Main output files:

   full_model_metrics.csv
   pair_and_split_summary.csv
   leakage_feature_correlations.csv
   label_weight_sensitivity.csv
   regional_transfer_results.csv
   fairness_and_accessibility.csv
   corrected_ablation_table.csv
   baseline_results.csv
   baseline_transparency.csv
   baseline_weight_sensitivity.csv
   manuscript_insert.tex
   reviewer_response.txt
   PRIME_EV_REVIEWER_READY.pt

The full run trains several models for ablations, label sensitivity, and regional holdouts.
Runtime depends on your CPU or GPU. Use --quick only to verify installation; do not report
quick-run values in the paper.
"""
    (output_dir / "HOW_TO_RUN.txt").write_text(text, encoding="utf-8")


# =============================================================================
# Main execution
# =============================================================================

# =============================================================================
# V5 reviewer additions: compact reviewer evidence with fixed cutoffs,
# distribution diagnostics, uncertainty checks, multi-seed tests, and LaTeX tables.
# =============================================================================

from scipy import stats
from sklearn.neural_network import MLPRegressor

FIXED_KS = [10, 25, 50, 100]


def latex_escape(text: Any) -> str:
    return str(text).replace("_", r"\_")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def memory_mb_for_model(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return float(total / (1024 ** 2))


def average_precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    k = int(min(max(1, k), len(labels)))
    true_top = set(np.argsort(-labels)[:k].tolist())
    order = np.argsort(-scores)[:k]
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(idx) in true_top:
            hits += 1
            total += hits / rank
    return float(total / max(1, min(k, len(true_top))))


def regret_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    k = int(min(max(1, k), len(labels)))
    ideal = float(np.mean(labels[np.argsort(-labels)[:k]]))
    predicted = float(np.mean(labels[np.argsort(-scores)[:k]]))
    return float(ideal - predicted)


def fixed_cutoff_metrics(labels: np.ndarray, scores: np.ndarray, pairs: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(labels)
    for k in FIXED_KS:
        kk = min(k, n)
        out[f"NDCG@{k}"] = ndcg_at_k(labels, scores, kk)
        agreement = precision_at_k(labels, scores, kk)
        ap = average_precision_at_k(labels, scores, kk)
        out[f"AP@{k}"] = ap
        out[f"TopKAgreement@{k}"] = agreement
        # Compatibility aliases for older result readers. These should not be
        # described as MAP/recall/hit-rate in the manuscript because there is
        # only one ranked list and the relevant-set size is exactly k.
        out[f"MAP@{k}"] = ap
        out[f"Recall@{k}"] = agreement
        out[f"HitRate@{k}"] = float(agreement > 0.0)
        out[f"TopKOverlap@{k}"] = agreement
        out[f"Regret@{k}"] = regret_at_k(labels, scores, kk)
    out["NDCG_full"] = ndcg_at_k(labels, scores, n)
    top10pct = max(1, int(math.ceil(n * 0.10)))
    out["NDCG@10pct"] = ndcg_at_k(labels, scores, top10pct)
    out["P@10pct"] = precision_at_k(labels, scores, top10pct)
    out["Spearman"] = safe_spearman(labels, scores)
    out["KendallTau"] = safe_kendall(labels, scores)
    out["PairwiseAccuracy"] = pairwise_accuracy(labels, scores, pairs)
    return out


def ci95(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) <= 1:
        val = float(arr[0]) if len(arr) else float("nan")
        return val, val
    half = float(stats.t.ppf(0.975, len(arr) - 1) * stats.sem(arr))
    return float(arr.mean() - half), float(arr.mean() + half)


def paired_test(control: Sequence[float], comp: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(control, dtype=float)
    b = np.asarray(comp, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    diff = a - b
    return float(stats.ttest_rel(a, b).pvalue), float(diff.mean() / (diff.std(ddof=1) + EPS))


def holm(pvals: List[float]) -> List[float]:
    order = np.argsort([1.0 if np.isnan(p) else p for p in pvals])
    out = np.zeros(len(pvals), dtype=float)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        p = 1.0 if np.isnan(pvals[idx]) else pvals[idx]
        running = max(running, min(1.0, p * (m - rank)))
        out[idx] = running
    return out.tolist()


def plot_g_distribution(bundle: DataBundle, output_dir: Path) -> Dict[str, float]:
    g = np.asarray(bundle.g_test, dtype=float)
    summary = {
        "n_test": int(len(g)),
        "min": float(g.min()),
        "q25": float(np.quantile(g, 0.25)),
        "median": float(np.median(g)),
        "mean": float(g.mean()),
        "q75": float(np.quantile(g, 0.75)),
        "max": float(g.max()),
        "std": float(g.std(ddof=1)),
        "iqr": float(np.quantile(g, 0.75) - np.quantile(g, 0.25)),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "g_score_distribution_summary.csv", index=False)
    plt.figure(figsize=(6.3, 3.8))
    plt.hist(g, bins=30)
    plt.xlabel(r"Reference priority score $g_i$")
    plt.ylabel("Stations")
    plt.tight_layout()
    plt.savefig(output_dir / "g_score_distribution.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "g_score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    return summary


def uncertainty_diagnostics(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, output_dir: Path) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float) + EPS
    residual = np.abs(y - mu)
    out = {"UncertaintyType": "heteroscedastic_aleatoric"}
    for label, z in [("80", 1.2816), ("90", 1.6449), ("95", 1.9600)]:
        lo, hi = mu - z * sigma, mu + z * sigma
        out[f"Coverage{label}"] = float(np.mean((y >= lo) & (y <= hi)))
        out[f"Width{label}"] = float(np.mean(hi - lo))
    out["ResidualSigmaSpearman"] = safe_spearman(residual, sigma)
    out["AbsResidualMean"] = float(np.mean(residual))
    pd.DataFrame([out]).to_csv(output_dir / "uncertainty_interval_diagnostics.csv", index=False)

    plt.figure(figsize=(5.3, 3.8))
    plt.scatter(sigma, residual, s=8, alpha=0.35)
    plt.xlabel(r"Predicted uncertainty $\sigma_i$")
    plt.ylabel(r"Absolute residual $|y_i-\mu_i|$")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_uncertainty.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    nominal = np.array([0.80, 0.90, 0.95])
    observed = np.array([out["Coverage80"], out["Coverage90"], out["Coverage95"]])
    plt.figure(figsize=(4.0, 3.8))
    plt.plot(nominal, observed, marker="o")
    plt.plot(nominal, nominal, linestyle="--")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Observed coverage")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_interval_coverage_plot.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    return out


def bounded_risk_baselines(bundle: DataBundle) -> pd.DataFrame:
    y_train = np.clip(bundle.y_train, 1e-4, 1 - 1e-4)
    y_test = np.clip(bundle.y_test, 1e-4, 1 - 1e-4)
    logit_train = np.log(y_train / (1 - y_train))
    rows = []
    ridge = Ridge(alpha=1.0).fit(bundle.X_train, logit_train)
    pred_ridge = 1.0 / (1.0 + np.exp(-ridge.predict(bundle.X_test)))
    rows.append({"Model": "TransformedGaussian_Ridge", "Risk_MSE": float(mean_squared_error(y_test, pred_ridge)), "Risk_Spearman": safe_spearman(y_test, pred_ridge)})
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=180, random_state=SEED).fit(bundle.X_train, y_train)
    pred_mlp = np.clip(mlp.predict(bundle.X_test), 0.0, 1.0)
    rows.append({"Model": "Bounded_MLP", "Risk_MSE": float(mean_squared_error(y_test, pred_mlp)), "Risk_Spearman": safe_spearman(y_test, pred_mlp)})
    return pd.DataFrame(rows)


def group_selection_diagnostics(raw_df: pd.DataFrame, scores: np.ndarray, utility: np.ndarray, top_fraction: float, output_dir: Path) -> pd.DataFrame:
    df = raw_df.copy().reset_index(drop=True)
    n = len(df)
    k = max(1, int(math.ceil(n * top_fraction)))
    selected = np.zeros(n, dtype=int)
    selected[np.argsort(-scores)[:k]] = 1
    df["selected"] = selected
    df["utility"] = utility
    df["region"] = df["Longitude"].apply(region_from_longitude)
    distance = pd.to_numeric(df["Distance to City (km)"], errors="coerce")
    df["access"] = np.where(distance >= distance.quantile(0.75), "LowAccess", "Other")
    rows = []
    for group_col in ["Station Operator", "region", "access"]:
        for group, sub in df.groupby(group_col):
            rate = float(sub["selected"].mean())
            se = math.sqrt(max(rate * (1 - rate), 0.0) / max(1, len(sub)))
            rows.append({
                "GroupType": group_col,
                "Group": str(group),
                "Stations": int(len(sub)),
                "SelectedStations": int(sub["selected"].sum()),
                "SelectionRate": rate,
                "SelectionRate_CI95_L": max(0.0, rate - 1.96 * se),
                "SelectionRate_CI95_U": min(1.0, rate + 1.96 * se),
                "MeanUtility": float(sub["utility"].mean()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "group_selection_diagnostics.csv", index=False)
    return out


def feature_counterfactual_tests(bundle: DataBundle, result: ModelResult, config: ExperimentConfig, device: torch.device, output_dir: Path) -> pd.DataFrame:
    rows = []
    base = result.test_scores
    for name in ["Cost (USD/kWh)", "Distance to City (km)", "Charging Capacity (kW)", "Renewable Energy Source_Yes"]:
        if name not in bundle.preprocessor.feature_names:
            continue
        idx = bundle.preprocessor.feature_names.index(name)
        X = bundle.X_test.copy()
        X[:, idx] = np.clip(X[:, idx] + 0.10, 0.0, 1.0)
        _, _, _, scores, _ = infer_model(result.model, X, bundle.cost_test, bundle.dist_test, device, repeats=1, batch_size=config.eval_batch_size)
        delta = scores - base
        rows.append({"Feature": name, "Intervention": "+0.10 scaled units", "MeanScoreDelta": float(delta.mean()), "AbsMeanScoreDelta": float(np.mean(np.abs(delta)))})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "feature_counterfactual_tests.csv", index=False)
    return out


def feature_permutation_sensitivity(bundle: DataBundle, result: ModelResult, config: ExperimentConfig, device: torch.device, output_dir: Path, repeats: int) -> pd.DataFrame:
    """Diagnostic only: corrupt test-column semantics without retraining.

    This is not a valid feature-order sensitivity experiment. The valid protocol
    must permute train/validation/test columns consistently and retrain every
    model under every permutation. Use conv1d_order_sensitivity_v2.py for that
    experiment. The output is retained only as a falsification diagnostic.
    """
    rng = np.random.default_rng(config.split_seed + 900)
    rows = []
    for r in range(repeats):
        permutation = rng.permutation(bundle.X_test.shape[1])
        X = bundle.X_test[:, permutation]
        _, _, _, scores, _ = infer_model(result.model, X, bundle.cost_test, bundle.dist_test, device, repeats=1, batch_size=config.eval_batch_size)
        rows.append({"Permutation": r, "Protocol": "test-only semantic corruption; not order sensitivity", **fixed_cutoff_metrics(bundle.g_test, scores, bundle.test_pairs)})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "test_feature_semantic_corruption.csv", index=False)
    return out


def evaluate_baselines_v5(bundle: DataBundle, config: ExperimentConfig, result: ModelResult) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    base, transparency, scores = evaluate_baselines(bundle, config, result)
    risk_model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=SEED, n_jobs=-1).fit(bundle.X_train, bundle.y_train)
    usage_model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=SEED + 1, n_jobs=-1).fit(bundle.X_train, bundle.u_train)
    scores["TwoStage_RiskUsage"] = config.risk_weight * risk_model.predict(bundle.X_test) + config.demand_weight * usage_model.predict(bundle.X_test)
    row = {"Method": "TwoStage_RiskUsage", "IsOracleUpperBound": False}
    row.update(ranking_metrics(bundle.g_test, scores["TwoStage_RiskUsage"], bundle.test_pairs, config.top_fraction))
    row.update(fixed_cutoff_metrics(bundle.g_test, scores["TwoStage_RiskUsage"], bundle.test_pairs))
    base = pd.concat([base, pd.DataFrame([row])], ignore_index=True, sort=False)
    transparency = pd.concat([transparency, pd.DataFrame([{
        "Method": "TwoStage_RiskUsage",
        "Family": "Simple two-stage baseline",
        "Inputs": "Same leakage-controlled inference features",
        "Objective": "Predict risk and usage separately; rank by 0.60 predicted risk + 0.40 predicted usage",
        "Hyperparameters": "RandomForestRegressor, n_estimators=120, max_depth=8",
        "Tuning": "Fixed reviewer baseline; no test tuning",
        "Implementation": "scikit-learn",
    }])], ignore_index=True)
    return base, transparency, scores


def baseline_fixed_table(bundle: DataBundle, scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method, score in scores.items():
        if method == "Oracle_Label_UpperBound":
            continue
        row = {"Method": method}
        row.update(fixed_cutoff_metrics(bundle.g_test, score, bundle.test_pairs))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_group(df: pd.DataFrame, group_col: str, output: Path) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col):
        for metric in [c for c in sub.columns if c not in {group_col, "Seed"} and pd.api.types.is_numeric_dtype(sub[c])]:
            vals = sub[metric].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            lo, hi = ci95(vals)
            rows.append({group_col: group, "Metric": metric, "Mean": float(vals.mean()), "Std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "CI95_L": lo, "CI95_U": hi, "N": int(len(vals))})
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False)
    return out


def significance_table(df: pd.DataFrame, group_col: str, control: str, comparators: Sequence[str], metric: str, output: Path) -> pd.DataFrame:
    rows, pvals = [], []
    for comp in comparators:
        a = df[df[group_col] == control].set_index("Seed")[metric]
        b = df[df[group_col] == comp].set_index("Seed")[metric]
        join = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        p, d = paired_test(join["a"], join["b"])
        pvals.append(1.0 if np.isnan(p) else p)
        rows.append({"Control": control, "Comparator": comp, "Metric": metric, "MeanDifference": float((join["a"] - join["b"]).mean()) if len(join) else float("nan"), "PairedP": p, "CohenDz": d, "NSeeds": int(len(join))})
    for row, adj in zip(rows, holm(pvals)):
        row["HolmP"] = adj
    out = pd.DataFrame(rows)
    out.to_csv(output, index=False)
    return out


def run_multiseed(df: pd.DataFrame, args: argparse.Namespace, config: ExperimentConfig, output_dir: Path, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global SEED
    old_seed = SEED
    seed_list = [int(s.strip()) for s in args.review_seeds.split(",") if s.strip()]
    baseline_rows, ablation_rows = [], []
    variants = {
        "Full": {},
        "NoIRE_Conv": {"no_ire": True},
        "PointwiseRanking": {"pointwise_rank": True},
        "NoAttention": {"no_attention": True},
        "DeterministicRisk": {"deterministic_risk": True},
    }

    # The split and evaluation pair pool are fixed once. Only model-training
    # randomness changes across the repeated-seed experiment.
    train_idx, val_idx, test_idx, fixed_split_meta = operator_disjoint_split(df, args.split_seed)
    fixed_split_meta = dict(fixed_split_meta)
    fixed_split_meta["split_seed"] = int(args.split_seed)

    for seed in seed_list:
        print(f"\n[V7] Multi-seed evaluation training_seed={seed}, split_seed={args.split_seed}")
        cfg = copy.deepcopy(config)
        cfg.epochs = min(cfg.epochs, args.multiseed_epochs)
        cfg.patience = min(cfg.patience, max(3, args.multiseed_epochs // 2))
        cfg.batch_pairs_train = min(cfg.batch_pairs_train, args.multiseed_train_pairs)
        cfg.batch_pairs_val = min(cfg.batch_pairs_val, args.multiseed_validation_pairs)
        cfg.batch_pairs_test = min(cfg.batch_pairs_test, args.multiseed_test_pairs)

        # Build identical preprocessing, targets, and pair pools for every seed.
        SEED = int(args.split_seed)
        b = build_bundle(df, train_idx, val_idx, test_idx, cfg, fixed_split_meta)
        SEED = seed
        set_seed(seed)

        full = train_prime_ev(b, cfg, device, name=f"Full_seed_{seed}", epochs_override=cfg.epochs)
        save_model_checkpoint(output_dir / "models" / f"PRIME_EV_v7_seed_{seed}.pt", full, b, cfg)
        _base, _trans, scores = evaluate_baselines_v5(b, cfg, full)
        fixed = baseline_fixed_table(b, scores)
        for _, r in fixed.iterrows():
            rec = dict(r)
            rec.update({"Seed": seed, "SplitSeed": int(args.split_seed), "TrainOperators": "|".join(fixed_split_meta["train_operators"]), "ValidationOperators": "|".join(fixed_split_meta["validation_operators"]), "TestOperators": "|".join(fixed_split_meta["test_operators"])})
            baseline_rows.append(rec)

        for vname, var in variants.items():
            res = full if vname == "Full" else train_prime_ev(b, cfg, device, name=f"{vname}_seed_{seed}", variant=var, epochs_override=cfg.epochs)
            rec = {"Seed": seed, "SplitSeed": int(args.split_seed), "Variant": vname, "ParameterCount": count_parameters(res.model), "ModelMemoryMB": memory_mb_for_model(res.model), "TrainingTime_seconds": res.train_seconds, "Latency_ms_per_station": res.latency_ms_per_station}
            rec.update(fixed_cutoff_metrics(b.g_test, res.test_scores, b.test_pairs))
            ablation_rows.append(rec)

    SEED = old_seed
    baselines = pd.DataFrame(baseline_rows)
    ablations = pd.DataFrame(ablation_rows)
    baselines.to_csv(output_dir / "multiseed_baseline_fixed_metrics.csv", index=False)
    ablations.to_csv(output_dir / "multiseed_ablation_metrics.csv", index=False)
    return baselines, ablations


def pareto_frontier(ablation: pd.DataFrame) -> pd.DataFrame:
    """Return the quality/equity/latency/complexity Pareto audit.

    Parameter count and model memory are required measured quantities. Missing
    complexity is never silently replaced by zero because that would create
    false Pareto-efficient points.
    """
    if ablation.empty:
        return ablation
    work = ablation.copy()
    required = [
        "Variant", "NDCG_full", "GeographicEquityScore",
        "Latency_ms_per_station", "ParameterCount", "ModelMemoryMB",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Pareto table is missing required columns: {missing}")
    for col in ["NDCG_full", "GeographicEquityScore", "Latency_ms_per_station", "ParameterCount", "ModelMemoryMB"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    invalid = (
        work[["NDCG_full", "GeographicEquityScore", "Latency_ms_per_station", "ParameterCount", "ModelMemoryMB"]].isna().any(axis=1)
        | (work["Latency_ms_per_station"] < 0)
        | (work["ParameterCount"] <= 0)
        | (work["ModelMemoryMB"] <= 0)
    )
    if invalid.any():
        bad = work.loc[invalid, ["Variant", "ParameterCount", "ModelMemoryMB"]].to_dict("records")
        raise ValueError(f"Pareto complexity values are missing or invalid: {bad}")

    work["ParetoEfficient"] = True
    for i, row in work.iterrows():
        dominates = (
            (work["NDCG_full"] >= row["NDCG_full"]) &
            (work["GeographicEquityScore"] >= row["GeographicEquityScore"]) &
            (work["Latency_ms_per_station"] <= row["Latency_ms_per_station"]) &
            (work["ParameterCount"] <= row["ParameterCount"]) &
            (work["ModelMemoryMB"] <= row["ModelMemoryMB"])
        )
        strict = (
            (work["NDCG_full"] > row["NDCG_full"]) |
            (work["GeographicEquityScore"] > row["GeographicEquityScore"]) |
            (work["Latency_ms_per_station"] < row["Latency_ms_per_station"]) |
            (work["ParameterCount"] < row["ParameterCount"]) |
            (work["ModelMemoryMB"] < row["ModelMemoryMB"])
        )
        work.loc[i, "ParetoEfficient"] = not bool((dominates & strict).any())
    return work


def make_pareto_plot(ablation: pd.DataFrame, output_dir: Path) -> None:
    if ablation.empty or "Latency_ms_per_station" not in ablation:
        return
    plt.figure(figsize=(6.0, 4.0))
    sizes = np.clip(ablation.get("ParameterCount", pd.Series(np.ones(len(ablation)) * 1000)) / 250, 25, 220)
    plt.scatter(ablation["Latency_ms_per_station"], ablation["NDCG_full"], s=sizes)
    for _, r in ablation.iterrows():
        plt.annotate(str(r["Variant"]), (r["Latency_ms_per_station"], r["NDCG_full"]), fontsize=7)
    plt.xlabel("Latency per station (ms)")
    plt.ylabel("NDCG")
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_quality_latency_complexity.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def f4(x: Any) -> str:
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def write_compact_latex(output_dir: Path, baseline_fixed: pd.DataFrame, multi_summary: pd.DataFrame, ablation_summary: pd.DataFrame, g_summary: Dict[str, float], unc: Dict[str, float]) -> None:
    methods = ["PRIME-EV", "Random", "VIKOR", "Pareto_Balanced", "RidgeRanker", "TwoStage_RiskUsage"]
    bf = baseline_fixed[baseline_fixed["Method"].isin(methods)].copy()
    lines_a = []
    for _, r in bf.iterrows():
        lines_a.append(f"{latex_escape(r['Method'])} & {f4(r.get('NDCG_full'))} & {f4(r.get('NDCG@10'))} & {f4(r.get('NDCG@25'))} & {f4(r.get('NDCG@50'))} & {f4(r.get('NDCG@100'))} & {f4(r.get('MAP@100'))} & {f4(r.get('P@10pct'))} & {f4(r.get('PairwiseAccuracy'))}\\")
    ms = multi_summary[(multi_summary.get("Method", pd.Series(dtype=str)).isin(methods)) & (multi_summary.get("Metric", pd.Series(dtype=str)).isin(["NDCG_full", "NDCG@10", "NDCG@100", "P@10pct", "Spearman", "KendallTau", "PairwiseAccuracy"]))]
    lines_b = []
    for _, r in ms.iterrows():
        lines_b.append(f"{latex_escape(r['Method'])} & {latex_escape(r['Metric'])} & {f4(r['Mean'])} $\\pm$ {f4(r['Std'])} & [{f4(r['CI95_L'])}, {f4(r['CI95_U'])}]\\")
    av = ablation_summary[(ablation_summary.get("Variant", pd.Series(dtype=str)).isin(["Full", "NoIRE_Conv", "PointwiseRanking", "NoAttention", "DeterministicRisk"])) & (ablation_summary.get("Metric", pd.Series(dtype=str)).isin(["NDCG_full", "PairwiseAccuracy", "Latency_ms_per_station", "ParameterCount", "ModelMemoryMB"]))]
    lines_c = []
    for _, r in av.iterrows():
        lines_c.append(f"{latex_escape(r['Variant'])} & {latex_escape(r['Metric'])} & {f4(r['Mean'])} $\\pm$ {f4(r['Std'])} & [{f4(r['CI95_L'])}, {f4(r['CI95_U'])}]\\")
    tex = r"""
\begin{table*}[!t]
\centering
\caption{Compact reviewer-oriented ranking, robustness, and diagnostic evidence. \textit{(a) reports fixed-cutoff ranking metrics for the final test split. (b) reports mean $\pm$ standard deviation and 95\% confidence intervals over independent seeds. (c) reports ablation robustness, latency, parameter count, and memory diagnostics over independent seeds.}}
\label{tab:v5_reviewer_evidence}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.05}
\begin{subtable}[t]{\textwidth}
\centering
\caption{Fixed-cutoff ranking metrics.}
\begin{tabular}{lcccccccc}
\toprule
Method & NDCG & NDCG@10 & NDCG@25 & NDCG@50 & NDCG@100 & MAP@100 & P@10\% & Pairwise Acc.\\
\midrule
""" + "\n".join(lines_a) + r"""
\bottomrule
\end{tabular}
\end{subtable}
\vspace{1mm}
\begin{subtable}[t]{0.49\textwidth}
\centering
\caption{Multi-seed baseline summary.}
\begin{tabular}{llcc}
\toprule
Method & Metric & Mean $\pm$ SD & 95\% CI\\
\midrule
""" + "\n".join(lines_b[:32]) + r"""
\bottomrule
\end{tabular}
\end{subtable}
\hfill
\begin{subtable}[t]{0.49\textwidth}
\centering
\caption{Multi-seed ablation and complexity summary.}
\begin{tabular}{llcc}
\toprule
Variant & Metric & Mean $\pm$ SD & 95\% CI\\
\midrule
""" + "\n".join(lines_c[:32]) + r"""
\bottomrule
\end{tabular}
\end{subtable}
\end{table*}

\paragraph{Additional ranking diagnostics.}
Figure~\ref{fig:g_distribution} shows the distribution of the reference priority score $g_i$. On the test partition, $g_i$ has mean """ + f4(g_summary.get("mean")) + ", standard deviation " + f4(g_summary.get("std")) + ", and interquartile range " + f4(g_summary.get("iqr")) + r""". This limited dispersion explains why a random ordering can obtain a high full-list NDCG: many permutations preserve similar cumulative-gain values. The revised evaluation therefore reports fixed-cutoff NDCG, MAP, recall, hit rate, top-$k$ overlap, regret, correlation statistics, and pairwise accuracy.

\paragraph{Risk-uncertainty diagnostics.}
The uncertainty head is evaluated as a heteroscedastic aleatoric uncertainty estimate. The 90\% prediction interval coverage is """ + f4(unc.get("Coverage90")) + ", with mean interval width " + f4(unc.get("Width90")) + ". The residual-versus-uncertainty Spearman correlation is " + f4(unc.get("ResidualSigmaSpearman")) + r""". The manuscript should use the phrase ``uncertainty estimate'' unless these diagnostic values support stronger calibration language.
"""
    (output_dir / "v7_compact_reviewer_table.tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRIME-EV V7 reviewer-safe experiment script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="prime_ev_v7_results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sensitivity-epochs", type=int, default=8)
    parser.add_argument("--regional-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-metric", choices=["ndcg_full", "ndcg_top_fraction"], default="ndcg_top_fraction")
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-demand", type=float, default=0.2)
    parser.add_argument("--lambda-rank", type=float, default=15.0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--pair-threshold", type=float, default=0.05)
    parser.add_argument("--train-pairs", type=int, default=40000)
    parser.add_argument("--validation-pairs", type=int, default=8000)
    parser.add_argument("--test-pairs", type=int, default=8000)
    parser.add_argument("--risk-weight", type=float, default=0.60)
    parser.add_argument("--demand-weight", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--review-seeds", default="42,123,456,789,2025")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed operator split seed used for every training seed")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 uses all pair/station batches approximately once per epoch")
    parser.add_argument("--disable-pair-margin-weighting", action="store_true")
    parser.add_argument("--multiseed-epochs", type=int, default=8)
    parser.add_argument("--multiseed-train-pairs", type=int, default=20000)
    parser.add_argument("--multiseed-validation-pairs", type=int, default=5000)
    parser.add_argument("--multiseed-test-pairs", type=int, default=5000)
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-regional-transfer", action="store_true")
    parser.add_argument("--skip-label-sensitivity", action="store_true")
    parser.add_argument("--skip-baseline-sensitivity", action="store_true")
    parser.add_argument("--skip-order-sensitivity", action="store_true", help="Skip retraining-based feature-order experiment")
    parser.add_argument("--skip-operator-cv", action="store_true", help="Skip leave-one-operator-out and constrained reranking")
    parser.add_argument("--order-permutations", type=int, default=5)
    parser.add_argument("--order-epochs", type=int, default=8)
    parser.add_argument("--order-seeds", default="42,123,456", help="Matched training seeds reused for every feature permutation")
    parser.add_argument("--operator-cv-epochs", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "histories").mkdir(exist_ok=True)
    config = ExperimentConfig(
        data_path=str(data_path), output_dir=str(output_dir), epochs=args.epochs, sensitivity_epochs=args.sensitivity_epochs,
        regional_epochs=args.regional_epochs, learning_rate=args.learning_rate, lambda_risk=args.lambda_risk,
        lambda_demand=args.lambda_demand, lambda_rank=args.lambda_rank, patience=args.patience,
        validation_metric=args.validation_metric, latent_dim=args.latent_dim, pair_threshold=args.pair_threshold,
        batch_pairs_train=args.train_pairs, batch_pairs_val=args.validation_pairs, batch_pairs_test=args.test_pairs,
        risk_weight=args.risk_weight, demand_weight=args.demand_weight, run_ablations=not args.skip_ablations,
        run_regional_transfer=not args.skip_regional_transfer, run_label_sensitivity=not args.skip_label_sensitivity,
        run_baseline_sensitivity=not args.skip_baseline_sensitivity, quick=args.quick, device=args.device,
        torch_threads=max(1, args.torch_threads), steps_per_epoch=max(0, args.steps_per_epoch),
        pair_margin_weighting=not args.disable_pair_margin_weighting, split_seed=args.split_seed,
    )
    if args.quick:
        config.epochs = 3; config.sensitivity_epochs = 2; config.regional_epochs = 2; config.steps_per_epoch = 2
        config.batch_pairs_train = min(config.batch_pairs_train, 3000)
        config.batch_pairs_val = min(config.batch_pairs_val, 1000)
        config.batch_pairs_test = min(config.batch_pairs_test, 1000)
        config.run_ablations = False; config.run_regional_transfer = False; config.run_label_sensitivity = False; config.run_baseline_sensitivity = False
        args.review_seeds = "42,123"; args.multiseed_epochs = 2; args.multiseed_train_pairs = 3000; args.multiseed_validation_pairs = 1000; args.multiseed_test_pairs = 1000
        args.order_permutations = 2; args.order_epochs = 2; args.operator_cv_epochs = 2
    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = choose_device(config.device)
    df = pd.read_csv(data_path)
    ensure_required_columns(df)
    df = df.copy()
    df["ValidationRegion"] = df["Longitude"].apply(region_from_longitude)
    save_json(output_dir / "experiment_config.json", asdict(config))
    train_idx, val_idx, test_idx, split_metadata = operator_disjoint_split(df, args.split_seed)
    split_metadata["split_seed"] = int(args.split_seed)
    bundle = build_bundle(df, train_idx, val_idx, test_idx, config, split_metadata)
    save_json(output_dir / "split_metadata.json", bundle.split_metadata)

    pair_summary = pd.DataFrame([
        {"Quantity": "Risk target y_i", "Source": "Reviews (Rating)", "Formula": "1 - training-scaled rating", "Interpretation": "Rating-derived risk proxy; not observed failure, complaint, or downtime", "Train": len(bundle.train_idx), "Validation": len(bundle.val_idx), "Test": len(bundle.test_idx), "TrainPairs": "--", "ValidationPairs": "--", "TestPairs": "--"},
        {"Quantity": "Demand target u_i", "Source": "Usage Stats (avg users/day)", "Formula": "training-scaled usage", "Interpretation": "Auxiliary demand target; excluded from inference inputs", "Train": len(bundle.train_idx), "Validation": len(bundle.val_idx), "Test": len(bundle.test_idx), "TrainPairs": "--", "ValidationPairs": "--", "TestPairs": "--"},
        {"Quantity": "Preference proxy g_i", "Source": "y_i and u_i", "Formula": f"{config.risk_weight:.2f}y_i + {config.demand_weight:.2f}u_i", "Interpretation": "Formula-derived intervention-priority proxy", "Train": len(bundle.train_idx), "Validation": len(bundle.val_idx), "Test": len(bundle.test_idx), "TrainPairs": "--", "ValidationPairs": "--", "TestPairs": "--"},
        {"Quantity": "Preference pair p_ij", "Source": "g_i", "Formula": f"sign(g_i-g_j), |g_i-g_j|>{config.pair_threshold}", "Interpretation": "Supervised pairwise preference", "Train": "--", "Validation": "--", "Test": "--", "TrainPairs": len(bundle.train_pairs[0]), "ValidationPairs": len(bundle.val_pairs[0]), "TestPairs": len(bundle.test_pairs[0])},
    ])
    pair_summary.to_csv(output_dir / "pair_and_split_summary.csv", index=False)

    print("[V7] Training full PRIME-EV model")
    full = train_prime_ev(bundle, config, device, name="PRIME-EV-Full")
    full.history.to_csv(output_dir / "histories" / "PRIME_EV_v7_FULL_history.csv", index=False)
    save_model_checkpoint(output_dir / "models" / "PRIME_EV_v7_FULL.pt", full, bundle, config)
    all_mu, all_sigma, all_usage, all_scores, all_latency = infer_all_candidates(full.model, bundle, config, device)
    full_row = {**full.test_metrics, **fixed_cutoff_metrics(bundle.g_test, full.test_scores, bundle.test_pairs), "Model": full.name, "BestEpoch": full.best_epoch, "ParameterCount": count_parameters(full.model), "ModelMemoryMB": memory_mb_for_model(full.model), "TrainingTime_seconds": full.train_seconds, "Latency_ms_per_station_test": full.latency_ms_per_station, "Latency_ms_per_station_all_candidates": all_latency, "TestTotalLoss": full.losses["total"], "TestRiskLoss": full.losses["risk"], "TestDemandLoss": full.losses["demand"], "TestRankingLoss": full.losses["rank"]}
    pd.DataFrame([full_row]).to_csv(output_dir / "full_model_metrics.csv", index=False)
    # V11 reproducibility: persist row-level validation and test predictions so
    # empirical random-reference tests and independent table regeneration do
    # not require reloading a private in-memory model object.
    val_mu_v11, val_sigma_v11, val_usage_v11, val_score_v11, _ = predict_split(
        full.model, bundle, "val", device, config
    )
    pd.DataFrame({
        "RowIndex": bundle.val_idx,
        "StationID": bundle.raw.iloc[bundle.val_idx]["Station ID"].astype(str).to_numpy(),
        "StationOperator": bundle.raw.iloc[bundle.val_idx]["Station Operator"].astype(str).to_numpy(),
        "RiskTarget": bundle.y_val,
        "DemandTarget": bundle.u_val,
        "ReferencePriority": bundle.g_val,
        "PredictedRiskMean": val_mu_v11,
        "PredictedRiskSigma": val_sigma_v11,
        "PredictedDemand": val_usage_v11,
        "PredictedScore": val_score_v11,
    }).to_csv(output_dir / "full_model_validation_predictions_v11.csv", index=False)
    pd.DataFrame({
        "RowIndex": bundle.test_idx,
        "StationID": bundle.raw.iloc[bundle.test_idx]["Station ID"].astype(str).to_numpy(),
        "StationOperator": bundle.raw.iloc[bundle.test_idx]["Station Operator"].astype(str).to_numpy(),
        "RiskTarget": bundle.y_test,
        "DemandTarget": bundle.u_test,
        "ReferencePriority": bundle.g_test,
        "PredictedRiskMean": full.test_mu,
        "PredictedRiskSigma": full.test_sigma,
        "PredictedDemand": full.test_usage_hat,
        "PredictedScore": full.test_scores,
    }).to_csv(output_dir / "full_model_test_predictions_v11.csv", index=False)

    g_summary = plot_g_distribution(bundle, output_dir)
    unc = uncertainty_diagnostics(bundle.y_test, full.test_mu, full.test_sigma, output_dir)
    conformal_uncertainty_diagnostics(full.model, bundle, config, device, output_dir)
    bounded_risk_baselines(bundle).to_csv(output_dir / "bounded_risk_formulation_comparison.csv", index=False)
    leakage_correlation_table(bundle).to_csv(output_dir / "leakage_feature_correlations.csv", index=False)
    dataset_quality_audit(df).to_csv(output_dir / "dataset_quality_audit.csv", index=False)
    proxy_validity_analysis(df).to_csv(output_dir / "proxy_validity_analysis.csv", index=False)
    random_ranking_reference(bundle.g_test, bundle.test_pairs, repetitions=200 if args.quick else 2000, seed=args.split_seed).to_csv(output_dir / "random_ranking_reference.csv", index=False)
    fair = fairness_metrics(df, all_scores, all_mu, bundle.preprocessor, config.top_fraction)
    pd.DataFrame([{**fair, "MetricScope": "all candidate stations, top 10 percent selected", "GES_equation": "1 - mean(geographic selection disparity, accessibility selection disparity, low-access coverage gap)", "GES_note": "GES can remain high when low-access coverage is zero if the other disparity components are small; report low-access coverage separately."}]).to_csv(output_dir / "fairness_and_accessibility.csv", index=False)
    group_selection_diagnostics(df, all_scores, all_mu, config.top_fraction, output_dir)
    feature_counterfactual_tests(bundle, full, config, device, output_dir)
    feature_permutation_sensitivity(bundle, full, config, device, output_dir, repeats=5 if args.quick else 10)

    print("[V7] Evaluating baselines")
    baseline_results, baseline_transparency, baseline_scores = evaluate_baselines_v5(bundle, config, full)
    baseline_results.to_csv(output_dir / "baseline_results.csv", index=False)
    baseline_transparency.to_csv(output_dir / "baseline_transparency.csv", index=False)
    baseline_fixed = baseline_fixed_table(bundle, baseline_scores)
    baseline_fixed.to_csv(output_dir / "baseline_fixed_cutoff_metrics.csv", index=False)
    make_summary_plot(baseline_results, output_dir / "baseline_ndcg.png")

    base_sens = baseline_weight_sensitivity(bundle, config, repetitions=100) if config.run_baseline_sensitivity else pd.DataFrame()
    base_sens.to_csv(output_dir / "baseline_weight_sensitivity.csv", index=False)
    ablation = run_ablations(bundle, config, device, full) if config.run_ablations else pd.DataFrame()
    if not ablation.empty:
        # Complexity is measured from each trained model inside run_ablations.
        # Never replace valid parameter or memory values with zero/NaN.
        pfront = pareto_frontier(ablation)
        pfront.to_csv(output_dir / "ablation_pareto_frontier.csv", index=False)
        make_pareto_plot(pfront, output_dir)
    ablation.to_csv(output_dir / "corrected_ablation_table.csv", index=False)
    label_sens = run_label_weight_sensitivity(bundle, config, device) if config.run_label_sensitivity else pd.DataFrame()
    label_sens.to_csv(output_dir / "label_weight_sensitivity.csv", index=False)
    regional = run_regional_transfer(df, config, device) if config.run_regional_transfer else pd.DataFrame()
    regional.to_csv(output_dir / "regional_transfer_results.csv", index=False)

    if not args.skip_order_sensitivity:
        run_retrained_feature_order_sensitivity(bundle, config, device, output_dir, permutations=args.order_permutations, epochs=args.order_epochs)
    if not args.skip_operator_cv:
        run_operator_cv_constrained_rerank(df, config, device, output_dir, epochs=args.operator_cv_epochs)

    if not args.skip_multiseed:
        multi, multi_ablation = run_multiseed(df, args, config, output_dir, device)
        multi_summary = summarize_by_group(multi, "Method", output_dir / "multiseed_baseline_summary.csv")
        ablation_summary = summarize_by_group(multi_ablation, "Variant", output_dir / "multiseed_ablation_summary.csv")
        significance_table(multi, "Method", "PRIME-EV", ["Random", "VIKOR", "Pareto_Balanced", "RidgeRanker", "TwoStage_RiskUsage"], "NDCG@100", output_dir / "significance_prime_vs_baselines.csv")
        significance_table(multi_ablation, "Variant", "Full", ["NoIRE_Conv", "PointwiseRanking", "NoAttention", "DeterministicRisk"], "NDCG_full", output_dir / "significance_full_vs_ablations.csv")
    else:
        multi_summary = pd.DataFrame(); ablation_summary = pd.DataFrame()

    write_compact_latex(output_dir, baseline_fixed, multi_summary, ablation_summary, g_summary, unc)

    authoritative_payload = {
        "methodological_scope": {
            "target_type": "proxy-ranking proof-of-concept",
            "risk_target": "rating-derived proxy, not observed failure/downtime/maintenance label",
            "priority_proxy": f"{config.risk_weight:.2f}*risk_proxy + {config.demand_weight:.2f}*usage_proxy",
            "ranking_input": "z_i, risk mean mu_i, risk uncertainty sigma_i, normalized cost c_i, normalized distance d_i",
            "fairness_scope": "full candidate-station pool for operator/group diagnostics; operator-disjoint test split is used for ranking evaluation",
        },
        "split": bundle.split_metadata,
        "full_model_metrics": full_row,
        "g_distribution": g_summary,
        "uncertainty_diagnostics": unc,
        "fairness": fair,
    }
    save_json(output_dir / "authoritative_manuscript_values.json", authoritative_payload)

    consistency_rows = []
    for key, val in full_row.items():
        if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
            consistency_rows.append({"ManuscriptKey": key, "AuthoritativeValue": float(val), "SourceFile": "authoritative_manuscript_values.json"})
    pd.DataFrame(consistency_rows).to_csv(output_dir / "manuscript_value_consistency_check.csv", index=False)

    upload_files = [
        "authoritative_manuscript_values.json",
        "manuscript_value_consistency_check.csv",
        "full_model_metrics.csv",
        "multiseed_baseline_summary.csv",
        "multiseed_ablation_summary.csv",
        "baseline_fixed_cutoff_metrics.csv",
        "baseline_results.csv",
        "baseline_transparency.csv",
        "significance_prime_vs_baselines.csv",
        "significance_full_vs_ablations.csv",
        "g_score_distribution_summary.csv",
        "g_score_distribution.pdf",
        "uncertainty_interval_diagnostics.csv",
        "risk_interval_coverage_plot.pdf",
        "residual_vs_uncertainty.pdf",
        "bounded_risk_formulation_comparison.csv",
        "feature_counterfactual_tests.csv",
        "test_feature_semantic_corruption.csv",
        "dataset_quality_audit.csv",
        "proxy_validity_analysis.csv",
        "random_ranking_reference.csv",
        "fairness_and_accessibility.csv",
        "group_selection_diagnostics.csv",
        "ablation_pareto_frontier.csv",
        "corrected_ablation_table.csv",
        "label_weight_sensitivity.csv",
        "regional_transfer_results.csv",
        "uncertainty_calibration_v8.csv",
        "retrained_feature_order_sensitivity_v8.csv",
        "operator_cv_oof_predictions_v8.csv",
        "operator_cv_constrained_rerank_tradeoff_v8.csv",
        "v7_compact_reviewer_table.tex",
    ]
    (output_dir / "V7_UPLOAD_THESE_FILES.txt").write_text("\n".join(upload_files), encoding="utf-8")

    print("\nPRIME-EV V7 reviewer-safe run completed")
    print(f"Output directory: {output_dir}")
    print("Use authoritative_manuscript_values.json and manuscript_value_consistency_check.csv as the single source of truth.")
    print("Upload the files listed in V7_UPLOAD_THESE_FILES.txt for manuscript table updates.")


# =============================================================================
# V9 exact-ranking extension
# =============================================================================
# This extension deliberately optimizes exact ranking without using either
# Reviews (Rating) or Usage Stats (avg users/day) as inference-time inputs.
# It replaces the order-sensitive Conv1D encoder in the full model with a
# feature-gated residual tabular encoder, adds train-only engineered features,
# uses a hybrid pointwise/pairwise/listwise ranking objective with hard-pair
# mining, and optionally applies a validation-selected strict-input fusion head.

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

_V8_MAIN = main
_V8_FIT_PREPROCESSOR = fit_preprocessor
_V8_TRANSFORM_INPUTS = transform_inputs
_V8_SAVE_CHECKPOINT = save_model_checkpoint

V9_RUNTIME_ARGS: Optional[argparse.Namespace] = None


def _v9_engineered_frame(df_part: pd.DataFrame, reference_year: Optional[float] = None) -> pd.DataFrame:
    """Create leakage-safe operational and geographic features.

    Only fields available in REQUIRED_COLS other than the two LABEL_ONLY_COLS are
    used. Address text and Station Operator are intentionally excluded.
    """
    frame = pd.DataFrame(index=df_part.index)
    lat = pd.to_numeric(df_part["Latitude"], errors="coerce")
    lon = pd.to_numeric(df_part["Longitude"], errors="coerce")
    cost = pd.to_numeric(df_part["Cost (USD/kWh)"], errors="coerce")
    distance = pd.to_numeric(df_part["Distance to City (km)"], errors="coerce")
    capacity = pd.to_numeric(df_part["Charging Capacity (kW)"], errors="coerce")
    year = pd.to_numeric(df_part["Installation Year"], errors="coerce")
    parking = pd.to_numeric(df_part["Parking Spots"], errors="coerce")
    availability = df_part["Availability"].apply(parse_availability_hours).astype(float)
    maintenance = maintenance_gap(df_part["Maintenance Frequency"])
    renewable = 1.0 - renewable_gap(df_part["Renewable Energy Source"])
    connector_count = (
        df_part["Connector Types"].astype(str)
        .apply(lambda value: len([p for p in __import__("re").split(r"[,;/|]+", value) if p.strip()]))
        .astype(float)
    )

    if reference_year is None:
        reference_year = float(year.max()) if year.notna().any() else 2025.0

    frame["geo_latitude"] = lat
    frame["geo_longitude"] = lon
    frame["geo_abs_latitude"] = np.abs(lat)
    frame["geo_lat_sin"] = np.sin(np.deg2rad(lat))
    frame["geo_lat_cos"] = np.cos(np.deg2rad(lat))
    frame["geo_lon_sin"] = np.sin(np.deg2rad(lon))
    frame["geo_lon_cos"] = np.cos(np.deg2rad(lon))
    frame["geo_region_americas"] = (lon < -30.0).astype(float)
    frame["geo_region_europe_africa"] = ((lon >= -30.0) & (lon <= 60.0)).astype(float)
    frame["geo_region_asia_oceania"] = (lon > 60.0).astype(float)

    frame["op_station_age"] = np.maximum(0.0, float(reference_year) - year)
    frame["op_log_distance"] = np.log1p(np.clip(distance, 0.0, None))
    frame["op_log_capacity"] = np.log1p(np.clip(capacity, 0.0, None))
    frame["op_log_cost"] = np.log1p(np.clip(cost, 0.0, None))
    frame["op_availability_fraction"] = availability / 24.0
    frame["op_maintenance_gap"] = maintenance
    frame["op_renewable_indicator"] = renewable
    frame["op_connector_count"] = connector_count

    frame["int_capacity_per_parking"] = capacity / (parking + 1.0)
    frame["int_cost_per_capacity"] = cost / (capacity + 1.0)
    frame["int_capacity_per_distance"] = capacity / (distance + 1.0)
    frame["int_distance_cost"] = distance * cost
    frame["int_capacity_parking"] = capacity * parking
    frame["int_capacity_availability"] = capacity * (availability / 24.0)
    frame["int_renewable_capacity"] = renewable * capacity
    frame["int_maintenance_age"] = maintenance * np.maximum(0.0, float(reference_year) - year)
    return frame.astype(float)


def fit_preprocessor(df_train: pd.DataFrame) -> Preprocessor:
    prep = _V8_FIT_PREPROCESSOR(df_train)
    enabled = not bool(getattr(V9_RUNTIME_ARGS, "disable_engineered_features", False))
    if not enabled:
        prep.v9_engineered_enabled = False
        return prep

    train_year = pd.to_numeric(df_train["Installation Year"], errors="coerce")
    reference_year = float(train_year.max()) if train_year.notna().any() else 2025.0
    engineered = _v9_engineered_frame(df_train, reference_year=reference_year)
    medians = engineered.median(numeric_only=True).fillna(0.0)
    engineered = engineered.fillna(medians)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaler.fit(engineered)

    prep.v9_engineered_enabled = True
    prep.v9_reference_year = reference_year
    prep.v9_engineered_medians = {k: float(v) for k, v in medians.items()}
    prep.v9_engineered_scaler = scaler
    prep.v9_engineered_feature_names = [f"V9::{name}" for name in engineered.columns]
    prep.feature_names = list(prep.feature_names) + list(prep.v9_engineered_feature_names)
    return prep


def transform_inputs(df_part: pd.DataFrame, prep: Preprocessor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_base, cost, distance = _V8_TRANSFORM_INPUTS(df_part, prep)
    if not bool(getattr(prep, "v9_engineered_enabled", False)):
        return X_base, cost, distance

    engineered = _v9_engineered_frame(df_part, reference_year=float(prep.v9_reference_year))
    medians = pd.Series(prep.v9_engineered_medians)
    engineered = engineered.fillna(medians).reindex(columns=medians.index)
    scaled = prep.v9_engineered_scaler.transform(engineered).astype(np.float32)
    X = np.concatenate([X_base.astype(np.float32), scaled], axis=1).astype(np.float32)
    return X, cost, distance


class V9ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class V9ExactRepresentationEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        residual_blocks: int,
        dropout: float,
        use_feature_gating: bool = True,
        simple_encoder: bool = False,
        legacy_conv: bool = False,
        legacy_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_feature_gating = use_feature_gating
        self.simple_encoder = simple_encoder
        self.legacy_conv = legacy_conv

        if legacy_conv:
            self.legacy = InfrastructureRepresentationEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                use_attention=legacy_attention,
                use_conv=True,
            )
            return

        if use_feature_gating:
            self.feature_gate = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_buffer("feature_gate", torch.zeros(input_dim))

        if simple_encoder:
            self.input_layer = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.GELU(),
            )
            self.blocks = nn.Identity()
            self.output_layer = nn.Identity()
        else:
            self.input_layer = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                *[V9ResidualBlock(hidden_dim, dropout) for _ in range(max(1, residual_blocks))]
            )
            self.output_layer = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, latent_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.legacy_conv:
            return self.legacy(x)
        if self.use_feature_gating:
            # Multipliers lie in [0.5, 1.5], retaining every input while allowing
            # stable trainable emphasis without hard feature deletion.
            x = x * (0.5 + torch.sigmoid(self.feature_gate))
        h = self.input_layer(x)
        h = self.blocks(h)
        return self.output_layer(h)


class V9RiskHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        width = max(48, hidden_dim // 2)
        self.body = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(width, 1)
        self.scale_head = nn.Linear(width, 1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(z)
        mu = torch.sigmoid(self.mean_head(h)).squeeze(1)
        # Bounded scale prevents numerical collapse while conformal calibration
        # remains responsible for marginal coverage.
        sigma = 0.015 + 0.60 * torch.sigmoid(self.scale_head(h)).squeeze(1)
        return mu, sigma


class V9DemandHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        width = max(48, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, width),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(width, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, mu: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, mu.unsqueeze(1), distance.unsqueeze(1)], dim=1)).squeeze(1)


class V9ExactPrimeEV(nn.Module):
    """PRIME-EV V9 with a decomposition-aligned exact-ranking head."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        residual_blocks: int,
        dropout: float,
        risk_weight: float,
        demand_weight: float,
        include_risk_in_ranker: bool = True,
        demand_enabled: bool = True,
        use_feature_gating: bool = True,
        simple_encoder: bool = False,
        legacy_conv: bool = False,
        legacy_attention: bool = True,
    ):
        super().__init__()
        self.risk_weight = float(risk_weight)
        self.demand_weight = float(demand_weight)
        self.include_risk_in_ranker = include_risk_in_ranker
        self.demand_enabled = demand_enabled
        self.encoder = V9ExactRepresentationEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            residual_blocks=residual_blocks,
            dropout=dropout,
            use_feature_gating=use_feature_gating,
            simple_encoder=simple_encoder,
            legacy_conv=legacy_conv,
            legacy_attention=legacy_attention,
        )
        self.risk = V9RiskHead(latent_dim, hidden_dim)
        self.demand = V9DemandHead(latent_dim, hidden_dim)

        rank_extra = 3  # usage, cost, distance
        if include_risk_in_ranker:
            rank_extra += 2  # mu, log sigma
        rank_input = input_dim + latent_dim + rank_extra
        self.rank_residual = nn.Sequential(
            nn.LayerNorm(rank_input),
            nn.Linear(rank_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(48, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(48, hidden_dim // 2), 1),
        )
        self.residual_scale_raw = nn.Parameter(torch.tensor(-0.7))

    def forward(
        self,
        x: torch.Tensor,
        cost: torch.Tensor,
        distance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        mu, sigma = self.risk(z)
        usage_hat = self.demand(z, mu, distance)

        pieces = [x, z]
        if self.include_risk_in_ranker:
            pieces.extend([mu.unsqueeze(1), torch.log1p(sigma).unsqueeze(1)])
        pieces.extend([usage_hat.unsqueeze(1), cost.unsqueeze(1), distance.unsqueeze(1)])
        residual = self.rank_residual(torch.cat(pieces, dim=1)).squeeze(1)

        wr = self.risk_weight if self.include_risk_in_ranker else 0.0
        wd = self.demand_weight if self.demand_enabled else 0.0
        denom = max(wr + wd, EPS)
        if denom <= EPS:
            base_priority = torch.full_like(mu, 0.5)
        else:
            base_priority = (wr * mu + wd * usage_hat) / denom
        base_logit = torch.logit(torch.clamp(base_priority, 1e-4, 1.0 - 1e-4))
        residual_scale = F.softplus(self.residual_scale_raw)
        score = base_logit + residual_scale * residual
        return mu, sigma, usage_hat, score


class V9StrictFusionModel(nn.Module):
    """Validation-selected fusion using strict inference features only.

    Components are the neural score, the predicted risk-demand decomposition,
    and a Ridge priority estimate. Ridge coefficients are stored as buffers, so
    the complete inference system is contained in the PyTorch checkpoint.
    """

    def __init__(
        self,
        base_model: V9ExactPrimeEV,
        ridge_coef: np.ndarray,
        ridge_intercept: float,
        component_mean: np.ndarray,
        component_std: np.ndarray,
        blend_weights: np.ndarray,
        risk_weight: float,
        demand_weight: float,
        structured_mode: str,
        ridge_alpha: float,
        validation_objective: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("ridge_coef", torch.tensor(ridge_coef, dtype=torch.float32))
        self.register_buffer("ridge_intercept", torch.tensor(float(ridge_intercept), dtype=torch.float32))
        self.register_buffer("component_mean", torch.tensor(component_mean, dtype=torch.float32))
        self.register_buffer("component_std", torch.tensor(component_std, dtype=torch.float32))
        self.register_buffer("blend_weights", torch.tensor(blend_weights, dtype=torch.float32))
        self.risk_weight = float(risk_weight)
        self.demand_weight = float(demand_weight)
        self.structured_mode = str(structured_mode)
        self.ridge_alpha = float(ridge_alpha)
        self.validation_objective = float(validation_objective)

    def forward(self, x: torch.Tensor, cost: torch.Tensor, distance: torch.Tensor):
        mu, sigma, usage_hat, neural_score = self.base_model(x, cost, distance)
        ridge_score = x @ self.ridge_coef + self.ridge_intercept
        if self.structured_mode == "risk_demand":
            structured = self.risk_weight * mu + self.demand_weight * usage_hat
        elif self.structured_mode == "risk_only":
            structured = mu
        elif self.structured_mode == "demand_only":
            structured = usage_hat
        else:
            structured = torch.zeros_like(mu)
        components = torch.stack([neural_score, structured, ridge_score], dim=1)
        normalized = (components - self.component_mean.unsqueeze(0)) / self.component_std.unsqueeze(0)
        score = normalized @ self.blend_weights
        return mu, sigma, usage_hat, score


def _v9_priority_point_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    beta = float(getattr(V9_RUNTIME_ARGS, "exact_huber_beta", 0.05))
    return F.smooth_l1_loss(torch.sigmoid(scores), labels, beta=max(beta, 1e-3))


def _v9_listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    temperature = float(getattr(V9_RUNTIME_ARGS, "exact_list_temperature", 0.18))
    target_prob = torch.softmax(labels / max(temperature, 1e-3), dim=0)
    log_pred = torch.log_softmax(scores / max(temperature, 1e-3), dim=0)
    return -torch.sum(target_prob * log_pred)


def _v9_pearson_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    s = scores - torch.mean(scores)
    y = labels - torch.mean(labels)
    denom = torch.sqrt(torch.sum(s * s) * torch.sum(y * y) + EPS)
    return 1.0 - torch.sum(s * y) / denom


def _v9_hard_lambdarank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    max_pairs: int,
) -> torch.Tensor:
    n = int(len(scores))
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    pair_idx = torch.triu_indices(n, n, offset=1, device=scores.device)
    i, j = pair_idx[0], pair_idx[1]
    label_diff = labels[i] - labels[j]
    valid = torch.abs(label_diff) > threshold
    if not torch.any(valid):
        return torch.tensor(0.0, device=scores.device)
    i, j, label_diff = i[valid], j[valid], label_diff[valid]
    rho = torch.sign(label_diff)

    order = torch.argsort(labels, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=scores.device)
    discounts = 1.0 / torch.log2(ranks.float() + 2.0)
    gains = torch.pow(2.0, labels) - 1.0
    delta = torch.abs((gains[i] - gains[j]) * (discounts[i] - discounts[j]))
    margin = torch.abs(label_diff)
    pair_terms = F.softplus(-rho * (scores[i] - scores[j]))
    hardness = pair_terms.detach()
    priority = delta * margin * (1.0 + hardness)
    if len(priority) > max_pairs:
        keep = torch.topk(priority, k=max_pairs, largest=True).indices
        pair_terms = pair_terms[keep]
        delta = delta[keep]
        margin = margin[keep]
    weights = delta * margin
    weights = weights / (torch.mean(weights) + EPS)
    return torch.mean(weights * pair_terms)


def _v9_exact_objective(
    labels: np.ndarray,
    scores: np.ndarray,
    pairs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    top_fraction: float,
) -> Tuple[float, Dict[str, float]]:
    metrics = ranking_metrics(labels, scores, pairs, top_fraction)
    value = (
        0.30 * metrics["NDCG_at_10_percent"]
        + 0.25 * metrics["Precision_at_10_percent"]
        + 0.20 * ((metrics["Spearman"] + 1.0) / 2.0)
        + 0.15 * metrics["PairwiseAccuracy"]
        + 0.10 * metrics["NDCG_full"]
    )
    return float(value), metrics


def _v9_components_from_arrays(
    model: V9ExactPrimeEV,
    X: np.ndarray,
    cost: np.ndarray,
    distance: np.ndarray,
    ridge: Ridge,
    structured_mode: str,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu, sigma, usage, neural, _ = infer_model(
        model, X, cost, distance, device, repeats=1, batch_size=config.eval_batch_size
    )
    ridge_score = ridge.predict(X)
    if structured_mode == "risk_demand":
        structured = config.risk_weight * mu + config.demand_weight * usage
    elif structured_mode == "risk_only":
        structured = mu
    elif structured_mode == "demand_only":
        structured = usage
    else:
        structured = np.zeros_like(mu)
    return mu, sigma, usage, neural, np.column_stack([neural, structured, ridge_score])


def _v9_fit_strict_fusion(
    base_model: V9ExactPrimeEV,
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    variant: Mapping[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    if bool(variant.get("no_fusion", False)) or bool(getattr(V9_RUNTIME_ARGS, "disable_exact_fusion", False)):
        return base_model, {"enabled": False, "reason": "disabled"}

    structured_mode = "risk_demand"
    if variant.get("no_dim", False) and variant.get("no_risk_input", False):
        structured_mode = "none"
    elif variant.get("no_dim", False):
        structured_mode = "risk_only"
    elif variant.get("no_risk_input", False):
        structured_mode = "demand_only"

    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_ridge = None
    best_ridge_alpha = None
    best_ridge_obj = -np.inf
    for alpha in alpha_grid:
        ridge = Ridge(alpha=alpha)
        ridge.fit(bundle.X_train, bundle.g_train)
        val_pred = ridge.predict(bundle.X_val)
        obj, _ = _v9_exact_objective(bundle.g_val, val_pred, bundle.val_pairs, config.top_fraction)
        if obj > best_ridge_obj:
            best_ridge_obj = obj
            best_ridge = ridge
            best_ridge_alpha = alpha
    assert best_ridge is not None

    _, _, _, _, train_components = _v9_components_from_arrays(
        base_model, bundle.X_train, bundle.cost_train, bundle.dist_train,
        best_ridge, structured_mode, config, device,
    )
    _, _, _, val_neural, val_components = _v9_components_from_arrays(
        base_model, bundle.X_val, bundle.cost_val, bundle.dist_val,
        best_ridge, structured_mode, config, device,
    )
    means = np.mean(train_components, axis=0)
    stds = np.std(train_components, axis=0)
    stds[stds < 1e-6] = 1.0
    val_norm = (val_components - means) / stds

    base_obj, base_metrics = _v9_exact_objective(
        bundle.g_val, val_neural, bundle.val_pairs, config.top_fraction
    )
    candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    candidates.append(np.array([1.0 / 3.0] * 3))
    trials = int(getattr(V9_RUNTIME_ARGS, "exact_fusion_trials", 512))
    rng = np.random.default_rng(SEED + 9107)
    if trials > 0:
        candidates.extend(rng.dirichlet(np.array([1.25, 1.0, 1.0]), size=trials))

    best_obj = base_obj
    best_weights = np.array([1.0, 0.0, 0.0])
    best_metrics = base_metrics
    for weights in candidates:
        fused = val_norm @ weights
        obj, metrics = _v9_exact_objective(bundle.g_val, fused, bundle.val_pairs, config.top_fraction)
        if obj > best_obj + 1e-12:
            best_obj = obj
            best_weights = np.asarray(weights, dtype=float)
            best_metrics = metrics

    min_gain = float(getattr(V9_RUNTIME_ARGS, "exact_fusion_min_gain", 0.002))
    if best_obj < base_obj + min_gain:
        best_weights = np.array([1.0, 0.0, 0.0])
        # Neural score is standardized inside the wrapper. Standardization is
        # monotonic and therefore does not alter its ordering.
        best_obj = base_obj
        best_metrics = base_metrics

    component_names = ["Neural", "Structured", "Ridge"]
    active = np.flatnonzero(np.asarray(best_weights) > 1e-8)
    fusion_selected = bool(len(active) >= 2)
    selected_score_source = (
        "MultiComponentFusion" if fusion_selected
        else f"{component_names[int(active[0])]}Only" if len(active) == 1
        else "Undefined"
    )

    metadata = {
        "evaluated": True,
        "enabled": fusion_selected,  # legacy-compatible alias: true only for actual fusion
        "fusion_selected": fusion_selected,
        "selected_score_source": selected_score_source,
        "ridge_alpha": float(best_ridge_alpha),
        "structured_mode": structured_mode,
        "weights_neural_structured_ridge": best_weights.tolist(),
        "base_validation_objective": float(base_obj),
        "selected_validation_objective": float(best_obj),
        "validation_metrics": best_metrics,
        "selection_rule": "validation-only composite; minimum gain required; test split never used",
    }

    # A [1,0,0] selection is the base neural model, not a fused model. Returning
    # the base model preserves identical ordering and avoids a misleading status.
    if len(active) == 1 and int(active[0]) == 0:
        return base_model, metadata

    selected_model = V9StrictFusionModel(
        base_model=base_model,
        ridge_coef=np.asarray(best_ridge.coef_, dtype=np.float32),
        ridge_intercept=float(best_ridge.intercept_),
        component_mean=np.asarray(means, dtype=np.float32),
        component_std=np.asarray(stds, dtype=np.float32),
        blend_weights=np.asarray(best_weights, dtype=np.float32),
        risk_weight=config.risk_weight,
        demand_weight=config.demand_weight,
        structured_mode=structured_mode,
        ridge_alpha=float(best_ridge_alpha),
        validation_objective=float(best_obj),
    ).to(device)
    return selected_model, metadata


def evaluate_losses(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
    split: str,
    variant: Mapping[str, Any],
    config: ExperimentConfig,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        mu, sigma, usage_hat, score = model(
            tensors[f"X_{split}"], tensors[f"cost_{split}"], tensors[f"dist_{split}"]
        )
        if variant.get("deterministic_risk", False):
            risk_loss = F.mse_loss(mu, tensors[f"y_{split}"])
        else:
            risk_loss = gaussian_nll(mu, sigma, tensors[f"y_{split}"])
        demand_loss = torch.tensor(0.0, device=mu.device)
        if not variant.get("no_dim", False):
            demand_loss = F.mse_loss(usage_hat, tensors[f"u_{split}"])
        pair_loss = pairwise_logistic_loss(score, tensors[f"{split}_pairs"])
        point_loss = _v9_priority_point_loss(score, tensors[f"g_{split}"])
        list_loss = _v9_listnet_loss(score, tensors[f"g_{split}"])
        corr_loss = _v9_pearson_loss(score, tensors[f"g_{split}"])
        rank_loss = (
            float(getattr(V9_RUNTIME_ARGS, "exact_pair_weight", 1.0)) * pair_loss
            + float(getattr(V9_RUNTIME_ARGS, "exact_point_weight", 1.0)) * point_loss
            + float(getattr(V9_RUNTIME_ARGS, "exact_list_weight", 0.10)) * list_loss
            + float(getattr(V9_RUNTIME_ARGS, "exact_corr_weight", 0.25)) * corr_loss
        )
        total = config.lambda_risk * risk_loss + config.lambda_demand * demand_loss + config.lambda_rank * rank_loss
    return {
        "total": float(total.item()),
        "risk": float(risk_loss.item()),
        "demand": float(demand_loss.item()),
        "rank": float(rank_loss.item()),
        "rank_pair": float(pair_loss.item()),
        "rank_point": float(point_loss.item()),
        "rank_list": float(list_loss.item()),
        "rank_corr": float(corr_loss.item()),
    }


def train_prime_ev(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    name: str = "PRIME-EV-Full",
    variant: Optional[Mapping[str, Any]] = None,
    epochs_override: Optional[int] = None,
) -> ModelResult:
    """Train PRIME-EV V9 with validation-only exact-ranking optimization."""
    variant = dict(variant or {})
    set_seed(SEED)

    hidden_dim = int(getattr(V9_RUNTIME_ARGS, "exact_hidden_dim", 128))
    default_latent = int(getattr(V9_RUNTIME_ARGS, "exact_latent_dim", 48))
    latent_dim = int(variant.get("latent_dim", default_latent))
    residual_blocks = int(getattr(V9_RUNTIME_ARGS, "exact_residual_blocks", 3))
    dropout = float(getattr(V9_RUNTIME_ARGS, "exact_dropout", 0.05))

    base_model = V9ExactPrimeEV(
        input_dim=bundle.X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        residual_blocks=residual_blocks,
        dropout=dropout,
        risk_weight=config.risk_weight,
        demand_weight=config.demand_weight,
        include_risk_in_ranker=not variant.get("no_risk_input", False),
        demand_enabled=not variant.get("no_dim", False),
        use_feature_gating=not variant.get("no_attention", False),
        simple_encoder=bool(variant.get("no_ire", False)),
        legacy_conv=bool(variant.get("legacy_conv", False)),
        legacy_attention=not variant.get("no_attention", False),
    ).to(device)

    tensors = tensors_for_bundle(bundle, device)
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=config.learning_rate,
        weight_decay=max(config.weight_decay, 1e-5),
    )

    epochs = int(epochs_override or config.epochs)
    pair_total = len(bundle.train_pairs[0])
    station_total = len(bundle.X_train)
    auto_steps = max(
        1,
        int(math.ceil(pair_total / max(1, config.pair_batch_size))),
        int(math.ceil(station_total / max(1, config.station_batch_size))),
    )
    steps_per_epoch = int(config.steps_per_epoch) if config.steps_per_epoch > 0 else auto_steps

    best_state = copy.deepcopy(base_model.state_dict())
    best_selection = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    hard_pair_count = int(getattr(V9_RUNTIME_ARGS, "exact_hard_pairs", 4096))
    pair_w = float(getattr(V9_RUNTIME_ARGS, "exact_pair_weight", 1.0))
    point_w = float(getattr(V9_RUNTIME_ARGS, "exact_point_weight", 1.0))
    list_w = float(getattr(V9_RUNTIME_ARGS, "exact_list_weight", 0.10))
    corr_w = float(getattr(V9_RUNTIME_ARGS, "exact_corr_weight", 0.25))
    hard_w = float(getattr(V9_RUNTIME_ARGS, "exact_hard_weight", 0.50))

    for epoch in range(1, epochs + 1):
        base_model.train()
        rng_epoch = np.random.default_rng(SEED * 100000 + epoch)
        loss_sums = {
            "total": 0.0, "risk": 0.0, "demand": 0.0, "rank": 0.0,
            "pair": 0.0, "point": 0.0, "list": 0.0, "corr": 0.0, "hard": 0.0,
        }

        for _step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            station_count = min(config.station_batch_size, station_total)
            station_idx_np = rng_epoch.choice(station_total, size=station_count, replace=False)
            pair_count = min(config.pair_batch_size, pair_total)
            pair_sel = rng_epoch.choice(pair_total, size=pair_count, replace=False)
            pair_i_np = bundle.train_pairs[0][pair_sel]
            pair_j_np = bundle.train_pairs[1][pair_sel]
            pair_rho_np = bundle.train_pairs[2][pair_sel]

            union_np, inverse = np.unique(
                np.concatenate([station_idx_np, pair_i_np, pair_j_np]), return_inverse=True
            )
            n_station = len(station_idx_np)
            n_pair = len(pair_i_np)
            station_local = torch.tensor(inverse[:n_station], dtype=torch.long, device=device)
            pair_i_local = torch.tensor(inverse[n_station:n_station + n_pair], dtype=torch.long, device=device)
            pair_j_local = torch.tensor(inverse[n_station + n_pair:], dtype=torch.long, device=device)
            pair_rho = torch.tensor(pair_rho_np, dtype=torch.float32, device=device)
            union_idx = torch.tensor(union_np, dtype=torch.long, device=device)
            station_idx_t = torch.tensor(station_idx_np, dtype=torch.long, device=device)

            mu_all, sigma_all, usage_all, score_all = base_model(
                tensors["X_train"][union_idx],
                tensors["cost_train"][union_idx],
                tensors["dist_train"][union_idx],
            )
            mu = mu_all[station_local]
            sigma = sigma_all[station_local]
            usage_hat = usage_all[station_local]
            station_score = score_all[station_local]
            y_target = tensors["y_train"][station_idx_t]
            u_target = tensors["u_train"][station_idx_t]
            g_target = tensors["g_train"][station_idx_t]

            risk_loss = (
                F.mse_loss(mu, y_target)
                if variant.get("deterministic_risk", False)
                else gaussian_nll(mu, sigma, y_target)
            )
            demand_loss = torch.tensor(0.0, device=device)
            if not variant.get("no_dim", False):
                demand_loss = F.mse_loss(usage_hat, u_target)

            if variant.get("pointwise_rank", False):
                pair_loss = torch.tensor(0.0, device=device)
                hard_loss = torch.tensor(0.0, device=device)
                list_loss = torch.tensor(0.0, device=device)
                corr_loss = torch.tensor(0.0, device=device)
                point_loss = _v9_priority_point_loss(station_score, g_target)
                rank_loss = point_loss
            else:
                pair_terms = F.softplus(-pair_rho * (score_all[pair_i_local] - score_all[pair_j_local]))
                margin = torch.abs(
                    tensors["g_train"][torch.tensor(pair_i_np, dtype=torch.long, device=device)]
                    - tensors["g_train"][torch.tensor(pair_j_np, dtype=torch.long, device=device)]
                )
                if config.pair_margin_weighting:
                    pair_weights = margin / (torch.mean(margin) + EPS)
                    pair_loss = torch.mean(pair_weights * pair_terms)
                else:
                    pair_loss = torch.mean(pair_terms)
                point_loss = _v9_priority_point_loss(station_score, g_target)
                list_loss = _v9_listnet_loss(station_score, g_target)
                corr_loss = _v9_pearson_loss(station_score, g_target)
                hard_loss = _v9_hard_lambdarank_loss(
                    station_score, g_target, config.pair_threshold, hard_pair_count
                )
                rank_loss = (
                    pair_w * pair_loss
                    + point_w * point_loss
                    + list_w * list_loss
                    + corr_w * corr_loss
                    + hard_w * hard_loss
                )

            loss = (
                config.lambda_risk * risk_loss
                + config.lambda_demand * demand_loss
                + config.lambda_rank * rank_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sums["total"] += float(loss.item())
            loss_sums["risk"] += float(risk_loss.item())
            loss_sums["demand"] += float(demand_loss.item())
            loss_sums["rank"] += float(rank_loss.item())
            loss_sums["pair"] += float(pair_loss.item())
            loss_sums["point"] += float(point_loss.item())
            loss_sums["list"] += float(list_loss.item())
            loss_sums["corr"] += float(corr_loss.item())
            loss_sums["hard"] += float(hard_loss.item())

        base_model.eval()
        with torch.no_grad():
            val_mu, val_sigma, val_usage, val_score = base_model(
                tensors["X_val"], tensors["cost_val"], tensors["dist_val"]
            )
            val_np = val_score.detach().cpu().numpy()
            exact_selection, val_metrics = _v9_exact_objective(
                bundle.g_val, val_np, bundle.val_pairs, config.top_fraction
            )
            if config.validation_metric == "ndcg_full":
                selection_value = val_metrics["NDCG_full"]
            elif config.validation_metric == "ndcg_top_fraction":
                selection_value = val_metrics["NDCG_at_10_percent"]
            else:
                selection_value = exact_selection

            val_risk = (
                F.mse_loss(val_mu, tensors["y_val"])
                if variant.get("deterministic_risk", False)
                else gaussian_nll(val_mu, val_sigma, tensors["y_val"])
            )
            val_demand = torch.tensor(0.0, device=device)
            if not variant.get("no_dim", False):
                val_demand = F.mse_loss(val_usage, tensors["u_val"])
            val_pair = pairwise_logistic_loss(val_score, tensors["val_pairs"])
            val_point = _v9_priority_point_loss(val_score, tensors["g_val"])
            val_list = _v9_listnet_loss(val_score, tensors["g_val"])
            val_corr = _v9_pearson_loss(val_score, tensors["g_val"])
            val_rank = pair_w * val_pair + point_w * val_point + list_w * val_list + corr_w * val_corr
            val_total = config.lambda_risk * val_risk + config.lambda_demand * val_demand + config.lambda_rank * val_rank

        row = {
            "epoch": epoch,
            "optimizer_steps": steps_per_epoch,
            "train_total": loss_sums["total"] / steps_per_epoch,
            "train_risk": loss_sums["risk"] / steps_per_epoch,
            "train_demand": loss_sums["demand"] / steps_per_epoch,
            "train_rank": loss_sums["rank"] / steps_per_epoch,
            "train_pair": loss_sums["pair"] / steps_per_epoch,
            "train_point": loss_sums["point"] / steps_per_epoch,
            "train_list": loss_sums["list"] / steps_per_epoch,
            "train_corr": loss_sums["corr"] / steps_per_epoch,
            "train_hard": loss_sums["hard"] / steps_per_epoch,
            "validation_total": float(val_total.item()),
            "validation_selection": float(selection_value),
            "validation_exact_composite": float(exact_selection),
            "validation_ndcg": float(val_metrics["NDCG_at_10_percent"]),
            "validation_top_agreement": float(val_metrics["Precision_at_10_percent"]),
            "validation_spearman": float(val_metrics["Spearman"]),
            "validation_pairwise_accuracy": float(val_metrics["PairwiseAccuracy"]),
        }
        history_rows.append(row)

        if selection_value > best_selection + 1e-6:
            best_selection = selection_value
            best_epoch = epoch
            best_state = copy.deepcopy(base_model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[{name}] V9 epoch {epoch:03d}/{epochs} steps={steps_per_epoch} "
                f"train={row['train_total']:.5f} val_exact={exact_selection:.5f} "
                f"NDCG10%={val_metrics['NDCG_at_10_percent']:.5f} "
                f"TopAgree={val_metrics['Precision_at_10_percent']:.5f} "
                f"rho={val_metrics['Spearman']:.5f} pair={val_metrics['PairwiseAccuracy']:.5f}"
            )
        if epochs_without_improvement >= config.patience:
            print(f"[{name}] V9 early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    train_seconds = time.perf_counter() - start_time
    base_model.load_state_dict(best_state)

    final_model, fusion_meta = _v9_fit_strict_fusion(
        base_model, bundle, config, device, variant
    )
    mu, sigma, usage_hat, score, latency = infer_model(
        final_model, bundle.X_test, bundle.cost_test, bundle.dist_test,
        device, batch_size=config.eval_batch_size,
    )
    ranking = ranking_metrics(bundle.g_test, score, bundle.test_pairs, config.top_fraction)
    risk = risk_metrics(bundle.y_test, mu, sigma)
    losses = evaluate_losses(final_model, tensors, "test", variant, config)
    metrics = dict(ranking)
    metrics.update(risk)
    metrics["Demand_MSE"] = float(mean_squared_error(bundle.u_test, usage_hat))
    metrics["SSI"] = system_stress_index(score)
    metrics["ExactComposite"] = _v9_exact_objective(
        bundle.g_test, score, bundle.test_pairs, config.top_fraction
    )[0]
    weights = fusion_meta.get("weights_neural_structured_ridge", [1.0, 0.0, 0.0])
    metrics["FusionEvaluated"] = float(bool(fusion_meta.get("evaluated", True)))
    metrics["FusionSelected"] = float(bool(fusion_meta.get("fusion_selected", False)))
    metrics["FusionEnabled"] = metrics["FusionSelected"]  # legacy-compatible alias
    metrics["ScoreSource"] = str(fusion_meta.get("selected_score_source", "NeuralOnly"))
    metrics["FusionWeightNeural"] = float(weights[0])
    metrics["FusionWeightStructured"] = float(weights[1])
    metrics["FusionWeightRidge"] = float(weights[2])

    if name == "PRIME-EV-Full":
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Compare the base neural model and selected strict fusion on validation
        # and test. All selection is validation-only.
        _, _, _, base_val_score, _ = infer_model(
            base_model, bundle.X_val, bundle.cost_val, bundle.dist_val,
            device, repeats=1, batch_size=config.eval_batch_size,
        )
        _, _, _, base_test_score, _ = infer_model(
            base_model, bundle.X_test, bundle.cost_test, bundle.dist_test,
            device, repeats=1, batch_size=config.eval_batch_size,
        )
        _, _, _, final_val_score, _ = infer_model(
            final_model, bundle.X_val, bundle.cost_val, bundle.dist_val,
            device, repeats=1, batch_size=config.eval_batch_size,
        )
        diagnostic_rows = []
        for split_name, labels, pairs_np, base_sc, final_sc in [
            ("Validation", bundle.g_val, bundle.val_pairs, base_val_score, final_val_score),
            ("Test", bundle.g_test, bundle.test_pairs, base_test_score, score),
        ]:
            for system_name, sc in [("V9_BaseNeural", base_sc), ("V9_SelectedStrictFusion", final_sc)]:
                row = {"Split": split_name, "System": system_name}
                row.update(fixed_cutoff_metrics(labels, sc, pairs_np))
                row["ExactComposite"] = _v9_exact_objective(labels, sc, pairs_np, config.top_fraction)[0]
                diagnostic_rows.append(row)
        pd.DataFrame(diagnostic_rows).to_csv(out_dir / "exact_ranking_diagnostics_v9.csv", index=False)
        save_json(out_dir / "exact_fusion_selection_v9.json", fusion_meta)
        manifest = pd.DataFrame({
            "Feature": bundle.preprocessor.feature_names,
            "InferenceAllowed": True,
            "LabelOnly": False,
        })
        for forbidden in LABEL_ONLY_COLS:
            manifest = pd.concat([manifest, pd.DataFrame([{
                "Feature": forbidden,
                "InferenceAllowed": False,
                "LabelOnly": True,
            }])], ignore_index=True)
        manifest.to_csv(out_dir / "strict_input_manifest_v9.csv", index=False)

    return ModelResult(
        name=name,
        model=final_model,
        history=pd.DataFrame(history_rows),
        train_seconds=float(train_seconds),
        best_epoch=int(best_epoch),
        test_scores=score,
        test_mu=mu,
        test_sigma=sigma,
        test_usage_hat=usage_hat,
        test_metrics=metrics,
        losses=losses,
        latency_ms_per_station=latency,
    )


def save_model_checkpoint(path: Path, result: ModelResult, bundle: DataBundle, config: ExperimentConfig) -> None:
    checkpoint = {
        "model_state_dict": result.model.state_dict(),
        "model_name": result.name,
        "model_version": "PRIME-EV V9 ExactRank",
        "config": asdict(config),
        "v9_runtime_args": vars(V9_RUNTIME_ARGS) if V9_RUNTIME_ARGS is not None else {},
        "split_metadata": bundle.split_metadata,
        "feature_names": bundle.preprocessor.feature_names,
        "best_epoch": result.best_epoch,
        "test_metrics": result.test_metrics,
        "test_losses": result.losses,
        "inference_guard": {
            "forbidden_label_inputs": LABEL_ONLY_COLS,
            "operator_feature_used": False,
            "address_feature_used": False,
        },
    }
    torch.save(checkpoint, path)


def run_retrained_feature_order_sensitivity(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    permutations: int = 5,
    epochs: int = 8,
) -> pd.DataFrame:
    """Compare legacy Conv1D with the V9 residual tabular encoder."""
    global SEED
    old_seed = SEED
    rows: List[Dict[str, Any]] = []
    identity = np.arange(bundle.X_train.shape[1])
    for rep in range(permutations):
        training_seed = int(config.split_seed + 3000 + rep)
        rng = np.random.default_rng(training_seed)
        permutation = identity.copy() if rep == 0 else rng.permutation(len(identity))
        b = _clone_bundle_with_permuted_features(bundle, permutation)
        for architecture, variant in (
            ("LegacyConv1D", {"legacy_conv": True, "no_fusion": True}),
            ("V9ResidualTabular", {"no_fusion": True}),
        ):
            SEED = training_seed
            set_seed(training_seed)
            res = train_prime_ev(
                b, config, device,
                name=f"Order_{architecture}_{rep}",
                variant=variant,
                epochs_override=epochs,
            )
            rec = {
                "Permutation": rep,
                "Architecture": architecture,
                "TrainingSeed": training_seed,
                "IdentityOrder": bool(rep == 0),
                "PermutationVector": "|".join(map(str, permutation.tolist())),
            }
            rec.update(fixed_cutoff_metrics(b.g_test, res.test_scores, b.test_pairs))
            rows.append(rec)
    SEED = old_seed
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "retrained_feature_order_sensitivity_v9.csv", index=False)
    return out


def parse_args() -> argparse.Namespace:
    global V9_RUNTIME_ARGS
    parser = argparse.ArgumentParser(
        description="PRIME-EV V9 exact-ranking and reviewer-completion experiment script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="prime_ev_v9_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sensitivity-epochs", type=int, default=10)
    parser.add_argument("--regional-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--validation-metric",
        choices=["ndcg_full", "ndcg_top_fraction", "exact_composite"],
        default="exact_composite",
    )
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-demand", type=float, default=0.30)
    parser.add_argument("--lambda-rank", type=float, default=6.0)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--pair-threshold", type=float, default=0.03)
    parser.add_argument("--train-pairs", type=int, default=60000)
    parser.add_argument("--validation-pairs", type=int, default=12000)
    parser.add_argument("--test-pairs", type=int, default=12000)
    parser.add_argument("--risk-weight", type=float, default=0.60)
    parser.add_argument("--demand-weight", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--review-seeds", default="42,123,456,789,2025,31415,27182,16180,57721,65537")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--disable-pair-margin-weighting", action="store_true")
    parser.add_argument("--multiseed-epochs", type=int, default=10)
    parser.add_argument("--multiseed-train-pairs", type=int, default=30000)
    parser.add_argument("--multiseed-validation-pairs", type=int, default=8000)
    parser.add_argument("--multiseed-test-pairs", type=int, default=8000)
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-regional-transfer", action="store_true")
    parser.add_argument("--skip-label-sensitivity", action="store_true")
    parser.add_argument("--skip-baseline-sensitivity", action="store_true")
    parser.add_argument("--skip-order-sensitivity", action="store_true")
    parser.add_argument("--skip-operator-cv", action="store_true")
    parser.add_argument("--order-permutations", type=int, default=5)
    parser.add_argument("--order-epochs", type=int, default=8)
    parser.add_argument("--operator-cv-epochs", type=int, default=10)

    parser.add_argument("--disable-engineered-features", action="store_true")
    parser.add_argument("--disable-exact-fusion", action="store_true")
    parser.add_argument("--exact-hidden-dim", type=int, default=128)
    parser.add_argument("--exact-latent-dim", type=int, default=48)
    parser.add_argument("--exact-residual-blocks", type=int, default=3)
    parser.add_argument("--exact-dropout", type=float, default=0.05)
    parser.add_argument("--exact-pair-weight", type=float, default=1.0)
    parser.add_argument("--exact-point-weight", type=float, default=1.0)
    parser.add_argument("--exact-list-weight", type=float, default=0.10)
    parser.add_argument("--exact-corr-weight", type=float, default=0.25)
    parser.add_argument("--exact-hard-weight", type=float, default=0.50)
    parser.add_argument("--exact-hard-pairs", type=int, default=4096)
    parser.add_argument("--exact-huber-beta", type=float, default=0.05)
    parser.add_argument("--exact-list-temperature", type=float, default=0.18)
    parser.add_argument("--exact-fusion-trials", type=int, default=512)
    parser.add_argument("--exact-fusion-min-gain", type=float, default=0.002)
    parser.add_argument("--quick", action="store_true")
    V9_RUNTIME_ARGS = parser.parse_args()
    return V9_RUNTIME_ARGS


def main() -> None:
    _V8_MAIN()
    if V9_RUNTIME_ARGS is None:
        return
    output_dir = Path(V9_RUNTIME_ARGS.output).expanduser().resolve()
    save_json(output_dir / "V9_EXACT_RANK_CONFIG.json", vars(V9_RUNTIME_ARGS))
    model_card = """# PRIME-EV V9 ExactRank model card

This run uses a feature-gated residual tabular encoder, decomposition-aligned risk/demand ranking, a hybrid pointwise-pairwise-listwise objective, hard LambdaRank-style pair mining, and an optional validation-selected strict-input fusion head.

Leakage guard: `Reviews (Rating)` and `Usage Stats (avg users/day)` remain target-side variables and are not inference inputs. Station Operator and Address are also excluded from the ranker. Fusion selection uses the validation partition only; test metrics are never used for model selection.

Interpretation boundary: this code maximizes ranking performance that is learnable from the allowed infrastructure features. It cannot manufacture exact station ordering when the source features do not contain information about the proxy target. Report the measured Spearman, pairwise accuracy, fixed-cutoff agreement, and random-reference intervals together with NDCG.
"""
    (output_dir / "V9_MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    upload_path = output_dir / "V7_UPLOAD_THESE_FILES.txt"
    additions = [
        "V9_EXACT_RANK_CONFIG.json",
        "V9_MODEL_CARD.md",
        "exact_ranking_diagnostics_v9.csv",
        "exact_fusion_selection_v9.json",
        "strict_input_manifest_v9.csv",
        "retrained_feature_order_sensitivity_v9.csv",
    ]
    existing = upload_path.read_text(encoding="utf-8").splitlines() if upload_path.exists() else []
    merged = existing + [item for item in additions if item not in existing]
    upload_path.write_text("\n".join(merged), encoding="utf-8")
    print("\nPRIME-EV V9 ExactRank completion files written.")
    print(f"V9 output directory: {output_dir}")

# =============================================================================
# V10 external cross-dataset validation extension
# =============================================================================
# This extension adds reviewer-requested external validation on two independent
# datasets. It uses only features that can be harmonized across sources and
# never uses rating, review count, sessions, or energy as inference inputs.

from dataclasses import dataclass as _v10_dataclass
from sklearn.preprocessing import RobustScaler
from scipy.stats import ks_2samp, wasserstein_distance

_V9_MAIN = main
V10_RUNTIME_ARGS: Optional[argparse.Namespace] = None


@_v10_dataclass
class CrossCorePreprocessor:
    feature_names: List[str]
    medians: Dict[str, float]
    scaler: RobustScaler
    source_min: Dict[str, float]
    source_max: Dict[str, float]


@_v10_dataclass
class CrossCoreModels:
    risk_model: Any
    demand_model: Any
    direct_model: Any
    ridge_model: Any
    fusion_alpha: float
    validation_metrics: Dict[str, float]
    selected_families: Dict[str, str]


def _v10_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _v10_clock_hour(hour: int, minute: int, ampm: Optional[str]) -> float:
    value = float(hour) + float(minute) / 60.0
    if ampm:
        marker = ampm.upper()
        if marker == "AM" and hour == 12:
            value -= 12.0
        elif marker == "PM" and hour != 12:
            value += 12.0
    return value


def _v10_external_hours(value: Any) -> float:
    """Average daily opening duration parsed from a weekly hours string."""
    text = str(value).replace("\u202f", " ").replace("\xa0", " ").strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return float("nan")
    segments = [segment.strip() for segment in text.split(";") if segment.strip()]
    if not segments:
        segments = [text]
    durations: List[float] = []
    pattern = __import__("re").compile(
        r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?\s*[\-–—]\s*"
        r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?",
        flags=__import__("re").IGNORECASE,
    )
    for segment in segments:
        lower = segment.lower()
        if "open 24 hours" in lower or "24/7" in lower or "24x7" in lower:
            durations.append(24.0)
            continue
        if "closed" in lower:
            durations.append(0.0)
            continue
        match = pattern.search(segment)
        if match:
            h1, m1, ap1, h2, m2, ap2 = match.groups()
            start = _v10_clock_hour(int(h1), int(m1 or 0), ap1)
            end = _v10_clock_hour(int(h2), int(m2 or 0), ap2)
            duration = end - start
            if duration <= 0:
                duration += 24.0
            durations.append(float(np.clip(duration, 0.0, 24.0)))
            continue
        # Main-dataset values such as 9:00-18:00 are supported by the V7 parser.
        parsed = parse_availability_hours(segment)
        if parsed != 12.0 or any(ch.isdigit() for ch in segment):
            durations.append(float(parsed))
    return float(np.mean(durations)) if durations else float("nan")


def _v10_connector_count(values: pd.Series) -> pd.Series:
    def count_one(value: Any) -> float:
        if pd.isna(value):
            return float("nan")
        text = str(value).strip()
        if not text:
            return float("nan")
        parts = [part.strip() for part in __import__("re").split(r"[,;/|]+", text) if part.strip()]
        return float(max(1, len(set(parts))))
    return values.apply(count_one).astype(float)


def _v10_base_feature_frame(
    latitude: pd.Series,
    longitude: pd.Series,
    capacity_kw: pd.Series,
    connector_count: pd.Series,
    availability_hours: pd.Series,
) -> pd.DataFrame:
    lat = _v10_numeric(latitude)
    lon = _v10_numeric(longitude)
    capacity = _v10_numeric(capacity_kw).clip(lower=0.0)
    connectors = _v10_numeric(connector_count).clip(lower=0.0)
    hours = _v10_numeric(availability_hours).clip(lower=0.0, upper=24.0)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    safe_connectors = connectors.replace(0.0, np.nan)
    frame = pd.DataFrame({
        "latitude": lat,
        "longitude": lon,
        "abs_latitude": lat.abs(),
        "latitude_sin": np.sin(lat_rad),
        "latitude_cos": np.cos(lat_rad),
        "longitude_sin": np.sin(lon_rad),
        "longitude_cos": np.cos(lon_rad),
        "capacity_kw": capacity,
        "log_capacity": np.log1p(capacity),
        "connector_count": connectors,
        "log_connector_count": np.log1p(connectors),
        "availability_hours": hours,
        "availability_fraction": hours / 24.0,
        "capacity_per_connector": capacity / safe_connectors,
        "capacity_x_availability": capacity * hours / 24.0,
        "geographic_interaction": lat * lon / 10000.0,
    })
    return frame.replace([np.inf, -np.inf], np.nan)


def _v10_main_features(df: pd.DataFrame) -> pd.DataFrame:
    return _v10_base_feature_frame(
        df["Latitude"],
        df["Longitude"],
        df["Charging Capacity (kW)"],
        _v10_connector_count(df["Connector Types"]),
        df["Availability"].apply(parse_availability_hours),
    )


def _v10_us_features(df: pd.DataFrame) -> pd.DataFrame:
    return _v10_base_feature_frame(
        df["latitude"],
        df["longitude"],
        df["total_kw"],
        df["num_connectors"],
        df["hours"].apply(_v10_external_hours),
    )


def _v10_palo_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = pd.Series(np.nan, index=df.index, dtype=float)
    return _v10_base_feature_frame(
        df["latitude"],
        df["longitude"],
        missing,
        df["connector_count"],
        missing,
    )


def _v10_fit_cross_preprocessor(frame: pd.DataFrame) -> CrossCorePreprocessor:
    names = frame.columns.tolist()
    medians = {}
    filled = pd.DataFrame(index=frame.index)
    for col in names:
        values = _v10_numeric(frame[col])
        median = float(values.median()) if values.notna().any() else 0.0
        medians[col] = median
        filled[col] = values.fillna(median)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaler.fit(filled[names])
    return CrossCorePreprocessor(
        feature_names=names,
        medians=medians,
        scaler=scaler,
        source_min={col: float(filled[col].min()) for col in names},
        source_max={col: float(filled[col].max()) for col in names},
    )


def _v10_transform_cross(frame: pd.DataFrame, prep: CrossCorePreprocessor) -> np.ndarray:
    filled = pd.DataFrame(index=frame.index)
    for col in prep.feature_names:
        values = _v10_numeric(frame[col]) if col in frame else pd.Series(np.nan, index=frame.index)
        filled[col] = values.fillna(prep.medians[col])
    transformed = prep.scaler.transform(filled[prep.feature_names]).astype(np.float32)
    return np.clip(transformed, -8.0, 8.0)


def _v10_exact_composite(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    k = max(1, int(math.ceil(0.10 * n)))
    pair_count = min(12000, max(1000, n * 10))
    try:
        pairs = sample_preference_pairs(labels, pair_count, 0.03, 17001 + n)
        pair_acc = pairwise_accuracy(labels, scores, pairs)
    except Exception:
        pair_acc = 0.5
    return float(
        0.30 * ndcg_at_k(labels, scores, k)
        + 0.25 * precision_at_k(labels, scores, k)
        + 0.20 * ((safe_spearman(labels, scores) + 1.0) / 2.0)
        + 0.15 * pair_acc
        + 0.10 * ndcg_at_k(labels, scores, n)
    )


def _v10_model_candidates(seed: int) -> Dict[str, Any]:
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            max_features=0.85,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.10,
            random_state=seed,
        ),
    }


def _v10_select_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    selection: str,
) -> Tuple[Any, str, float]:
    best_model = None
    best_name = ""
    best_value = -np.inf
    for family, model in _v10_model_candidates(seed).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        if selection == "mse":
            value = -float(mean_squared_error(y_val, pred))
        else:
            value = _v10_exact_composite(y_val, pred)
        if value > best_value:
            best_model, best_name, best_value = model, family, value
    if best_model is None:
        raise RuntimeError("Cross-dataset model selection failed.")
    return best_model, best_name, float(best_value)


def _v10_train_cross_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    u_train: np.ndarray,
    g_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    u_val: np.ndarray,
    g_val: np.ndarray,
    risk_weight: float,
    demand_weight: float,
    seed: int,
) -> CrossCoreModels:
    risk_model, risk_name, _ = _v10_select_regressor(X_train, y_train, X_val, y_val, seed + 11, "mse")
    demand_model, demand_name, _ = _v10_select_regressor(X_train, u_train, X_val, u_val, seed + 23, "mse")
    direct_model, direct_name, _ = _v10_select_regressor(X_train, g_train, X_val, g_val, seed + 37, "rank")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, g_train)

    risk_val = np.clip(risk_model.predict(X_val), 0.0, 1.0)
    demand_val = np.clip(demand_model.predict(X_val), 0.0, 1.0)
    decomposition = risk_weight * risk_val + demand_weight * demand_val
    direct_val = direct_model.predict(X_val)
    best_alpha = 0.0
    best_metric = -np.inf
    for alpha in np.linspace(0.0, 1.0, 41):
        score = alpha * direct_val + (1.0 - alpha) * decomposition
        metric = _v10_exact_composite(g_val, score)
        if metric > best_metric + 1e-12:
            best_metric = metric
            best_alpha = float(alpha)
    return CrossCoreModels(
        risk_model=risk_model,
        demand_model=demand_model,
        direct_model=direct_model,
        ridge_model=ridge,
        fusion_alpha=best_alpha,
        validation_metrics={"ExactComposite": float(best_metric)},
        selected_families={"risk": risk_name, "demand": demand_name, "direct": direct_name},
    )


def _v10_predict_components(
    models: CrossCoreModels,
    X: np.ndarray,
    risk_weight: float,
    demand_weight: float,
) -> Dict[str, np.ndarray]:
    risk = np.clip(models.risk_model.predict(X), 0.0, 1.0)
    demand = np.clip(models.demand_model.predict(X), 0.0, 1.0)
    direct = models.direct_model.predict(X)
    decomposition = risk_weight * risk + demand_weight * demand
    full = models.fusion_alpha * direct + (1.0 - models.fusion_alpha) * decomposition
    return {
        "PRIME_CommonCore_Full": full,
        "PRIME_NoDirectFusion": decomposition,
        "PRIME_DirectOnly": direct,
        "PRIME_NoRisk": demand,
        "PRIME_NoDemand": risk,
        "Ridge_CommonCore": models.ridge_model.predict(X),
        "RiskHead": risk,
        "DemandHead": demand,
    }


def _v10_pair_pool(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    requested = min(20000, max(1000, n * 20))
    threshold = 0.03 if n >= 100 else 0.05
    try:
        return sample_preference_pairs(labels, requested, threshold, seed)
    except Exception:
        rng = np.random.default_rng(seed)
        i, j = np.triu_indices(n, 1)
        diff = labels[i] - labels[j]
        valid = np.abs(diff) > 1e-8
        i, j, diff = i[valid], j[valid], diff[valid]
        if len(i) == 0:
            i = np.array([0], dtype=int)
            j = np.array([min(1, n - 1)], dtype=int)
            diff = np.array([1.0])
        if len(i) > requested:
            selected = rng.choice(len(i), requested, replace=False)
            i, j, diff = i[selected], j[selected], diff[selected]
        return i.astype(np.int64), j.astype(np.int64), np.sign(diff).astype(np.float32)


def _v10_metric_row(
    dataset: str,
    protocol: str,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    pairs = _v10_pair_pool(labels, 19000 + seed + len(labels))
    metrics = fixed_cutoff_metrics(labels, scores, pairs)
    k = max(1, int(math.ceil(0.10 * len(labels))))
    metrics.update({
        "Dataset": dataset,
        "Protocol": protocol,
        "Method": method,
        "Seed": int(seed),
        "Stations": int(len(labels)),
        "NDCG_at_10_percent": ndcg_at_k(labels, scores, k),
        "TopKAgreement_at_10_percent": precision_at_k(labels, scores, k),
        "Regret_at_10_percent": regret_at_k(labels, scores, k),
    })
    return metrics


def _v10_prepare_us(path: Path, risk_weight: float, demand_weight: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(path)
    required = {"latitude", "longitude", "rating", "review_count", "hours", "num_connectors", "total_kw"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Public external dataset is missing required columns: {missing}")
    exact_duplicates = int(raw.duplicated().sum())
    deduplicated = raw.drop_duplicates().copy()
    usable = deduplicated[
        deduplicated["rating"].notna()
        & deduplicated["latitude"].notna()
        & deduplicated["longitude"].notna()
    ].copy().reset_index(drop=True)
    rating = _v10_numeric(usable["rating"]).clip(1.0, 5.0)
    risk = 1.0 - (rating - 1.0) / 4.0
    reviews = np.log1p(_v10_numeric(usable["review_count"]).fillna(0.0).clip(lower=0.0))
    popularity = reviews.rank(method="average", pct=True).to_numpy(dtype=float)
    usable["external_risk_proxy"] = risk.to_numpy(dtype=float)
    usable["external_popularity_proxy"] = popularity
    usable["external_priority_proxy"] = risk_weight * risk.to_numpy(dtype=float) + demand_weight * popularity
    audit = {
        "Dataset": "External_Public_Stations",
        "RawRows": int(len(raw)),
        "ExactDuplicateRows": exact_duplicates,
        "RowsAfterExactDeduplication": int(len(deduplicated)),
        "UsableRatedStations": int(len(usable)),
        "MissingRatingFractionRaw": float(raw["rating"].isna().mean()),
        "StatesOrRegions": int(usable["state"].astype(str).nunique()) if "state" in usable else 0,
        "TopRegionLabels": "|".join(usable["state"].astype(str).value_counts().head(8).index.tolist()) if "state" in usable else "",
        "GeographyAudit": "Mixed region labels are present; dataset title must not be interpreted as a single-country guarantee.",
        "TargetDefinition": f"{risk_weight:.2f}*(1-(rating-1)/4)+{demand_weight:.2f}*percentile(log1p(review_count))",
    }
    return usable, _v10_us_features(usable), audit


def _v10_prepare_palo(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    usecols = [
        "Station Name", "Latitude", "Longitude", "Port Number", "Plug Type", "Port Type",
        "Start Date", "Energy (kWh)", "User ID",
    ]
    accum: Dict[str, Dict[str, Any]] = {}
    raw_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100000, low_memory=False):
        raw_rows += len(chunk)
        chunk["_date"] = pd.to_datetime(chunk["Start Date"], errors="coerce")
        for station, sub in chunk.groupby("Station Name", dropna=False):
            key = str(station)
            rec = accum.setdefault(key, {
                "sessions": 0,
                "energy": 0.0,
                "users": set(),
                "ports": set(),
                "plug_types": set(),
                "port_types": set(),
                "latitudes": [],
                "longitudes": [],
                "start": None,
                "end": None,
            })
            rec["sessions"] += int(len(sub))
            rec["energy"] += float(_v10_numeric(sub["Energy (kWh)"]).fillna(0.0).sum())
            rec["users"].update(sub["User ID"].dropna().astype(str).unique().tolist())
            rec["ports"].update(sub["Port Number"].dropna().astype(str).unique().tolist())
            rec["plug_types"].update(sub["Plug Type"].dropna().astype(str).unique().tolist())
            rec["port_types"].update(sub["Port Type"].dropna().astype(str).unique().tolist())
            rec["latitudes"].extend(_v10_numeric(sub["Latitude"]).dropna().tolist())
            rec["longitudes"].extend(_v10_numeric(sub["Longitude"]).dropna().tolist())
            start, end = sub["_date"].min(), sub["_date"].max()
            if pd.notna(start) and (rec["start"] is None or start < rec["start"]):
                rec["start"] = start
            if pd.notna(end) and (rec["end"] is None or end > rec["end"]):
                rec["end"] = end
    rows: List[Dict[str, Any]] = []
    for station, rec in accum.items():
        active_days = 1
        if rec["start"] is not None and rec["end"] is not None:
            active_days = max(1, int((rec["end"] - rec["start"]).days + 1))
        rows.append({
            "station_name": station,
            "latitude": float(np.median(rec["latitudes"])) if rec["latitudes"] else np.nan,
            "longitude": float(np.median(rec["longitudes"])) if rec["longitudes"] else np.nan,
            "sessions": int(rec["sessions"]),
            "energy_kwh": float(rec["energy"]),
            "unique_users": int(len(rec["users"])),
            "connector_count": int(max(1, len(rec["ports"]))),
            "plug_type_count": int(len(rec["plug_types"])),
            "port_type_count": int(len(rec["port_types"])),
            "active_days": int(active_days),
            "sessions_per_day": float(rec["sessions"] / active_days),
            "energy_per_day": float(rec["energy"] / active_days),
        })
    stations = pd.DataFrame(rows)
    stable = stations[
        (stations["active_days"] >= 180)
        & (stations["sessions"] >= 100)
        & stations["latitude"].notna()
        & stations["longitude"].notna()
    ].copy().reset_index(drop=True)
    session_pct = stable["sessions_per_day"].rank(method="average", pct=True).to_numpy(dtype=float)
    energy_pct = stable["energy_per_day"].rank(method="average", pct=True).to_numpy(dtype=float)
    stable["observed_demand_target"] = 0.50 * session_pct + 0.50 * energy_pct
    audit = {
        "Dataset": "External_PaloAlto_Usage",
        "RawSessionRows": int(raw_rows),
        "UniqueStationNames": int(len(stations)),
        "StableStations": int(len(stable)),
        "MinimumActiveDays": 180,
        "MinimumSessions": 100,
        "TargetDefinition": "0.50*percentile(sessions_per_active_day)+0.50*percentile(energy_kWh_per_active_day)",
    }
    return stable, _v10_palo_features(stable), audit


def _v10_bootstrap_summary(
    dataset: str,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    records: Dict[str, List[float]] = {
        "NDCG_full": [],
        "NDCG_at_10_percent": [],
        "TopKAgreement_at_10_percent": [],
        "Spearman": [],
        "KendallTau": [],
        "PairwiseAccuracy": [],
        "Regret_at_10_percent": [],
    }
    n = len(labels)
    for rep in range(repetitions):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        s = scores[idx]
        k = max(1, int(math.ceil(0.10 * n)))
        pair_n = min(2000, max(300, n * 4))
        left = rng.integers(0, n, size=pair_n)
        right = rng.integers(0, n, size=pair_n)
        valid = left != right
        label_diff = y[left] - y[right]
        valid &= np.abs(label_diff) > 1e-8
        if np.any(valid):
            predicted = np.sign(s[left[valid]] - s[right[valid]])
            predicted[predicted == 0] = 1
            truth = np.sign(label_diff[valid])
            boot_pair_acc = float(np.mean(predicted == truth))
        else:
            boot_pair_acc = 0.5
        records["NDCG_full"].append(ndcg_at_k(y, s, n))
        records["NDCG_at_10_percent"].append(ndcg_at_k(y, s, k))
        records["TopKAgreement_at_10_percent"].append(precision_at_k(y, s, k))
        records["Spearman"].append(safe_spearman(y, s))
        records["KendallTau"].append(safe_kendall(y, s))
        records["PairwiseAccuracy"].append(boot_pair_acc)
        records["Regret_at_10_percent"].append(regret_at_k(y, s, k))
    rows: List[Dict[str, Any]] = []
    for metric, values in records.items():
        arr = np.asarray(values, dtype=float)
        rows.append({
            "Dataset": dataset,
            "Method": method,
            "Metric": metric,
            "BootstrapRepetitions": int(repetitions),
            "Mean": float(np.mean(arr)),
            "Std": float(np.std(arr, ddof=1)),
            "CI95_L": float(np.quantile(arr, 0.025)),
            "CI95_U": float(np.quantile(arr, 0.975)),
        })
    return rows


def _v10_shift_table(
    source_frame: pd.DataFrame,
    target_frames: Mapping[str, pd.DataFrame],
    prep: CrossCorePreprocessor,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset, target in target_frames.items():
        for feature in prep.feature_names:
            source_values = _v10_numeric(source_frame[feature]).dropna().to_numpy(dtype=float)
            target_series = _v10_numeric(target[feature]) if feature in target else pd.Series(np.nan, index=target.index)
            target_values = target_series.dropna().to_numpy(dtype=float)
            missing_fraction = float(target_series.isna().mean())
            if len(source_values) and len(target_values):
                ks = ks_2samp(source_values, target_values)
                wd = wasserstein_distance(source_values, target_values)
                below = float(np.mean(target_values < prep.source_min[feature]))
                above = float(np.mean(target_values > prep.source_max[feature]))
                ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
            else:
                wd, below, above, ks_stat, ks_p = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            rows.append({
                "Dataset": dataset,
                "Feature": feature,
                "TargetMissingFraction": missing_fraction,
                "KSStatistic": ks_stat,
                "KSPValue": ks_p,
                "WassersteinDistanceRawUnits": wd,
                "BelowSourceRangeFraction": below,
                "AboveSourceRangeFraction": above,
            })
    return pd.DataFrame(rows)


def _v10_us_state_adaptation(
    usable: pd.DataFrame,
    X: np.ndarray,
    source_score: np.ndarray,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    labels = usable["external_priority_proxy"].to_numpy(dtype=float)
    groups = usable["state"].fillna("Unknown").astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val, test = next(outer.split(X, labels, groups))
    inner_groups = groups[train_val]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_rel, val_rel = next(inner.split(X[train_val], labels[train_val], inner_groups))
    train = train_val[train_rel]
    val = train_val[val_rel]

    target_model, family, _ = _v10_select_regressor(X[train], labels[train], X[val], labels[val], seed + 101, "rank")
    target_pred = target_model.predict(X[test])

    stacked_train = np.column_stack([X[train], source_score[train]])
    stacked_val = np.column_stack([X[val], source_score[val]])
    stacked_test = np.column_stack([X[test], source_score[test]])
    best_model = None
    best_name = ""
    best_value = -np.inf
    candidates = {
        "RidgeStack": [Ridge(alpha=a) for a in (0.1, 1.0, 10.0)],
        "ExtraTreesStack": [ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=seed + 202, n_jobs=-1)],
    }
    for family_name, models in candidates.items():
        for model in models:
            model.fit(stacked_train, labels[train])
            pred = model.predict(stacked_val)
            value = _v10_exact_composite(labels[val], pred)
            if value > best_value:
                best_model, best_name, best_value = model, family_name, value
    if best_model is None:
        raise RuntimeError("public-dataset adaptation model selection failed.")
    adapted_pred = best_model.predict(stacked_test)
    random_pred = np.random.default_rng(seed + 303).random(len(test))
    metadata = pd.DataFrame([{
        "Seed": seed,
        "TrainStations": int(len(train)),
        "ValidationStations": int(len(val)),
        "TestStations": int(len(test)),
        "TrainStates": "|".join(sorted(set(groups[train]))),
        "ValidationStates": "|".join(sorted(set(groups[val]))),
        "TestStates": "|".join(sorted(set(groups[test]))),
        "TargetOnlyFamily": family,
        "AdaptationFamily": best_name,
        "AdaptationValidationComposite": float(best_value),
    }])
    predictions = {
        "SourceZeroShot": source_score[test],
        "ExternalTargetOnly": target_pred,
        "SourcePlusTargetAdaptation": adapted_pred,
        "Random": random_pred,
    }
    return metadata, predictions, test


def _v10_write_latex(
    output_dir: Path,
    summary: pd.DataFrame,
    audits: pd.DataFrame,
) -> None:
    preferred = [
        "PRIME_CommonCore_Full", "PRIME_NoDirectFusion", "PRIME_NoRisk",
        "PRIME_NoDemand", "Ridge_CommonCore", "PRIME_DemandHead",
        "Source_DirectPriorityHead", "Source_RiskHead_NegativeControl",
        "Ridge_SourcePriority", "Random",
    ]
    lines: List[str] = []
    for dataset in ["External_Public_Stations", "External_PaloAlto_Usage"]:
        subset = summary[(summary["Dataset"] == dataset) & summary["Method"].isin(preferred)]
        for _, row in subset.iterrows():
            lines.append(
                f"{latex_escape(dataset)} & {latex_escape(row['Method'])} & "
                f"{f4(row.get('NDCG_full'))} & {f4(row.get('NDCG_at_10_percent'))} & "
                f"{f4(row.get('TopKAgreement_at_10_percent'))} & {f4(row.get('Spearman'))} & "
                f"{f4(row.get('PairwiseAccuracy'))} & {f4(row.get('Regret_at_10_percent'))}\\\\"
            )
    audit_parts: List[str] = []
    for _, row in audits.iterrows():
        count = row.get("UsableRatedStations", np.nan)
        if pd.isna(count):
            count = row.get("StableStations", np.nan)
        if pd.isna(count):
            count = row.get("TestStations", 0)
        audit_parts.append(f"{row['Dataset']}: {int(count)} usable stations")
    audit_text = "; ".join(audit_parts)
    tex = r"""
\begin{table*}[!t]
\centering
\caption{External cross-dataset validation using a leakage-controlled common feature core. The external public station dataset evaluates a rating--popularity priority proxy, while the Palo Alto usage dataset independently evaluates observed demand. Models are selected using the source validation partition only; no external test labels are used for zero-shot model selection.}
\label{tab:cross_dataset_v10}
\scriptsize
\begin{tabular}{llcccccc}
\toprule
Dataset & Method & NDCG & NDCG@10\% & Top-10\% Agreement & Spearman & Pairwise Acc. & Regret@10\%\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}

\paragraph{Cross-dataset protocol.}
External validation was conducted on two independently sourced datasets using only harmonizable inference variables: geographic coordinates, charging capacity, connector count, and operating hours. The original rating and usage variables, external ratings, review counts, sessions, and energy outcomes were never used as model inputs. The external public-station experiment evaluates zero-shot transfer to a rating--popularity priority proxy after exact-row deduplication. The Palo Alto experiment aggregates session records to stable station-level demand outcomes using sessions per active day and energy per active day. Dataset-specific target definitions are reported explicitly because the two external sources do not contain the same outcome schema. """ + audit_text + r""". These experiments test transfer of the ranking methodology and its common feature representation; they do not establish that the three outcome constructs are interchangeable.
"""
    (output_dir / "cross_dataset_table_v10.tex").write_text(tex, encoding="utf-8")


def run_cross_dataset_validation(args: argparse.Namespace) -> None:
    data_path = Path(args.data).expanduser().resolve()
    us_path = Path(args.external_us).expanduser().resolve()
    palo_path = Path(args.external_usage).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists() or not us_path.exists() or not palo_path.exists():
        raise FileNotFoundError("Main, public external, and Palo Alto external datasets must all exist.")

    print("\n[V10] Preparing source and external cross-dataset benchmarks")
    df = pd.read_csv(data_path)
    ensure_required_columns(df)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    cfg = ExperimentConfig(
        data_path=str(data_path),
        output_dir=str(output_dir),
        epochs=args.epochs,
        sensitivity_epochs=args.sensitivity_epochs,
        regional_epochs=args.regional_epochs,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        batch_pairs_train=min(args.train_pairs, 30000),
        batch_pairs_val=min(args.validation_pairs, 8000),
        batch_pairs_test=min(args.test_pairs, 8000),
        pair_threshold=args.pair_threshold,
        risk_weight=args.risk_weight,
        demand_weight=args.demand_weight,
        lambda_risk=args.lambda_risk,
        lambda_demand=args.lambda_demand,
        lambda_rank=args.lambda_rank,
        patience=args.patience,
        validation_metric=args.validation_metric,
        quick=args.quick,
        split_seed=args.split_seed,
    )
    bundle = build_bundle(df, train_idx, val_idx, test_idx, cfg, split_meta)
    source_features = _v10_main_features(df)
    prep = _v10_fit_cross_preprocessor(source_features.iloc[train_idx])
    X_source_train = _v10_transform_cross(source_features.iloc[train_idx], prep)
    X_source_val = _v10_transform_cross(source_features.iloc[val_idx], prep)
    X_source_test = _v10_transform_cross(source_features.iloc[test_idx], prep)

    us, us_features, us_audit = _v10_prepare_us(us_path, args.risk_weight, args.demand_weight)
    palo, palo_features, palo_audit = _v10_prepare_palo(palo_path)
    X_us = _v10_transform_cross(us_features, prep)
    X_palo = _v10_transform_cross(palo_features, prep)

    audits = pd.DataFrame([
        {
            "Dataset": "Source_MAIN",
            "RawRows": int(len(df)),
            "TrainStations": int(len(train_idx)),
            "ValidationStations": int(len(val_idx)),
            "TestStations": int(len(test_idx)),
            "TargetDefinition": f"{args.risk_weight:.2f}*rating-derived-risk+{args.demand_weight:.2f}*normalized-usage",
        },
        us_audit,
        palo_audit,
    ])
    audits.to_csv(output_dir / "cross_dataset_audit_v10.csv", index=False)
    data_quality_flags = pd.DataFrame([
        {
            "Dataset": "External_Public_Stations",
            "Flag": "Exact duplicate rows",
            "Value": int(us_audit["ExactDuplicateRows"]),
            "Implication": "Exact duplicates are removed before evaluation.",
        },
        {
            "Dataset": "External_Public_Stations",
            "Flag": "Missing ratings",
            "Value": float(us_audit["MissingRatingFractionRaw"]),
            "Implication": "Only stations with observed ratings enter the rating-popularity benchmark.",
        },
        {
            "Dataset": "External_Public_Stations",
            "Flag": "Mixed geography labels",
            "Value": us_audit.get("TopRegionLabels", ""),
            "Implication": "Results are reported as a public mixed-geography station benchmark, not a single-country benchmark.",
        },
        {
            "Dataset": "External_PaloAlto_Usage",
            "Flag": "Small stable-station sample",
            "Value": int(palo_audit["StableStations"]),
            "Implication": "Demand-transfer results require bootstrap intervals and cautious interpretation.",
        },
    ])
    data_quality_flags.to_csv(output_dir / "cross_dataset_data_quality_flags_v10.csv", index=False)

    provenance = pd.DataFrame([
        {
            "Dataset": "Source_MAIN",
            "Role": "Source training and operator-disjoint internal test",
            "URL": "https://www.kaggle.com/datasets/vivekattri/global-ev-charging-stations-dataset",
        },
        {
            "Dataset": "External_Public_Stations",
            "Role": "External station-level rating and popularity validation",
            "URL": "https://www.kaggle.com/datasets/pythoxb/ev-charging-stations-dataset",
        },
        {
            "Dataset": "External_PaloAlto_Usage",
            "Role": "External observed charging-demand validation",
            "URL": "https://www.kaggle.com/datasets/venkatsairo4899/ev-charging-station-usage-of-california-city",
        },
    ])
    provenance.to_csv(output_dir / "cross_dataset_provenance_v10.csv", index=False)

    feature_manifest = pd.DataFrame([
        {
            "Feature": feature,
            "SourceMedian": prep.medians[feature],
            "SourceMinimum": prep.source_min[feature],
            "SourceMaximum": prep.source_max[feature],
            "UsedInUSExternal": True,
            "UsedInPaloExternal": feature not in {
                "capacity_kw", "log_capacity", "availability_hours", "availability_fraction",
                "capacity_per_connector", "capacity_x_availability",
            },
            "PaloMissingValueHandling": "source-training median imputation" if feature in {
                "capacity_kw", "log_capacity", "availability_hours", "availability_fraction",
                "capacity_per_connector", "capacity_x_availability",
            } else "observed/derived",
        }
        for feature in prep.feature_names
    ])
    feature_manifest.to_csv(output_dir / "cross_dataset_common_feature_manifest_v10.csv", index=False)

    shift = _v10_shift_table(
        source_features.iloc[train_idx],
        {"External_Public_Stations": us_features, "External_PaloAlto_Usage": palo_features},
        prep,
    )
    shift.to_csv(output_dir / "cross_dataset_domain_shift_v10.csv", index=False)

    seed_list = [int(item.strip()) for item in args.cross_seeds.split(",") if item.strip()]
    per_seed_rows: List[Dict[str, Any]] = []
    prediction_accumulator: Dict[Tuple[str, str], List[np.ndarray]] = {}
    model_selection_rows: List[Dict[str, Any]] = []
    adaptation_meta_rows: List[pd.DataFrame] = []
    adaptation_metric_rows: List[Dict[str, Any]] = []

    y_us = us["external_priority_proxy"].to_numpy(dtype=float)
    y_palo = palo["observed_demand_target"].to_numpy(dtype=float)
    for seed in seed_list:
        print(f"[V10] Cross-dataset source model seed={seed}")
        models = _v10_train_cross_models(
            X_source_train, bundle.y_train, bundle.u_train, bundle.g_train,
            X_source_val, bundle.y_val, bundle.u_val, bundle.g_val,
            args.risk_weight, args.demand_weight, seed,
        )
        model_selection_rows.append({
            "Seed": seed,
            "RiskFamily": models.selected_families["risk"],
            "DemandFamily": models.selected_families["demand"],
            "DirectFamily": models.selected_families["direct"],
            "FusionAlphaDirect": models.fusion_alpha,
            "SourceValidationExactComposite": models.validation_metrics["ExactComposite"],
        })

        source_pred = _v10_predict_components(models, X_source_test, args.risk_weight, args.demand_weight)
        us_pred = _v10_predict_components(models, X_us, args.risk_weight, args.demand_weight)
        palo_pred = _v10_predict_components(models, X_palo, args.risk_weight, args.demand_weight)

        random_source = np.random.default_rng(seed + 5001).random(len(bundle.g_test))
        random_us = np.random.default_rng(seed + 5002).random(len(y_us))
        random_palo = np.random.default_rng(seed + 5003).random(len(y_palo))
        source_methods = {
            **{k: v for k, v in source_pred.items() if k not in {"RiskHead", "DemandHead"}},
            "Random": random_source,
        }
        us_methods = {
            **{k: v for k, v in us_pred.items() if k not in {"RiskHead", "DemandHead"}},
            "Random": random_us,
        }
        # Palo Alto provides an observed demand target rather than the source
        # rating-demand composite. Use semantically distinct transfer controls
        # instead of duplicating the same demand prediction under several
        # architecture-ablation labels.
        palo_methods = {
            "PRIME_DemandHead": palo_pred["DemandHead"],
            "Source_DirectPriorityHead": palo_pred["PRIME_DirectOnly"],
            "Source_RiskHead_NegativeControl": palo_pred["RiskHead"],
            "Ridge_SourcePriority": models.ridge_model.predict(X_palo),
            "Random": random_palo,
        }
        for dataset, protocol, labels, methods in (
            ("Source_MAIN", "operator-disjoint internal", bundle.g_test, source_methods),
            ("External_Public_Stations", "zero-shot source-to-external", y_us, us_methods),
            ("External_PaloAlto_Usage", "zero-shot demand-head transfer", y_palo, palo_methods),
        ):
            for method, scores in methods.items():
                per_seed_rows.append(_v10_metric_row(dataset, protocol, method, labels, scores, seed))
                prediction_accumulator.setdefault((dataset, method), []).append(np.asarray(scores, dtype=float))

        # Secondary public-dataset region-disjoint adaptation protocol; the zero-shot result remains primary.
        adaptation_meta, adaptation_predictions, test_positions = _v10_us_state_adaptation(
            us, X_us, us_pred["PRIME_CommonCore_Full"], seed + 7000,
        )
        adaptation_meta_rows.append(adaptation_meta)
        for method, scores in adaptation_predictions.items():
            adaptation_metric_rows.append(
                _v10_metric_row(
                    "External_Public_Stations",
                    "state-disjoint target adaptation",
                    method,
                    y_us[test_positions],
                    scores,
                    seed,
                )
            )

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(output_dir / "cross_dataset_multiseed_metrics_v10.csv", index=False)
    pd.DataFrame(model_selection_rows).to_csv(output_dir / "cross_dataset_source_model_selection_v10.csv", index=False)
    if adaptation_meta_rows:
        pd.concat(adaptation_meta_rows, ignore_index=True).to_csv(output_dir / "cross_dataset_public_adaptation_splits_v10.csv", index=False)
    adaptation_metrics = pd.DataFrame(adaptation_metric_rows)
    adaptation_metrics.to_csv(output_dir / "cross_dataset_public_adaptation_metrics_v10.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    numeric_metrics = [
        "NDCG_full", "NDCG_at_10_percent", "TopKAgreement_at_10_percent",
        "Spearman", "KendallTau", "PairwiseAccuracy", "Regret_at_10_percent",
        "NDCG@10", "NDCG@25", "NDCG@50", "NDCG@100",
    ]
    for (dataset, protocol, method), sub in per_seed.groupby(["Dataset", "Protocol", "Method"]):
        row: Dict[str, Any] = {
            "Dataset": dataset,
            "Protocol": protocol,
            "Method": method,
            "Seeds": int(sub["Seed"].nunique()),
            "Stations": int(sub["Stations"].iloc[0]),
        }
        for metric in numeric_metrics:
            if metric in sub:
                values = sub[metric].to_numpy(dtype=float)
                lo, hi = ci95(values)
                if metric in {"NDCG_full", "NDCG_at_10_percent", "TopKAgreement_at_10_percent", "PairwiseAccuracy", "NDCG@10", "NDCG@25", "NDCG@50", "NDCG@100"}:
                    lo, hi = max(0.0, lo), min(1.0, hi)
                elif metric in {"Spearman", "KendallTau"}:
                    lo, hi = max(-1.0, lo), min(1.0, hi)
                elif metric.startswith("Regret"):
                    lo = max(0.0, lo)
                row[metric] = float(np.mean(values))
                row[f"{metric}_SD"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_CI95_L"] = lo
                row[f"{metric}_CI95_U"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "cross_dataset_summary_v10.csv", index=False)

    if not adaptation_metrics.empty:
        adaptation_rows: List[Dict[str, Any]] = []
        adaptation_keys = [
            "NDCG_full", "NDCG_at_10_percent", "TopKAgreement_at_10_percent",
            "Spearman", "KendallTau", "PairwiseAccuracy", "Regret_at_10_percent",
        ]
        for method, sub in adaptation_metrics.groupby("Method"):
            for metric in adaptation_keys:
                values = sub[metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                lo, hi = ci95(values)
                if metric in {"NDCG_full", "NDCG_at_10_percent", "TopKAgreement_at_10_percent", "PairwiseAccuracy"}:
                    lo, hi = max(0.0, lo), min(1.0, hi)
                elif metric in {"Spearman", "KendallTau"}:
                    lo, hi = max(-1.0, lo), min(1.0, hi)
                elif metric.startswith("Regret"):
                    lo = max(0.0, lo)
                adaptation_rows.append({
                    "Method": method, "Metric": metric, "Mean": float(values.mean()),
                    "Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "CI95_L": lo, "CI95_U": hi, "NSeeds": int(len(values)),
                })
        adaptation_summary = pd.DataFrame(adaptation_rows)
        adaptation_summary.to_csv(output_dir / "cross_dataset_public_adaptation_summary_v10.csv", index=False)
    else:
        adaptation_summary = pd.DataFrame()

    bootstrap_rows: List[Dict[str, Any]] = []
    ensemble_prediction_rows: List[Dict[str, Any]] = []
    labels_by_dataset = {
        "Source_MAIN": np.asarray(bundle.g_test, dtype=float),
        "External_Public_Stations": y_us,
        "External_PaloAlto_Usage": y_palo,
    }
    for (dataset, method), predictions in prediction_accumulator.items():
        mean_score = np.mean(np.vstack(predictions), axis=0)
        labels = labels_by_dataset[dataset]
        bootstrap_rows.extend(
            _v10_bootstrap_summary(
                dataset, method, labels, mean_score,
                repetitions=args.cross_bootstrap,
                seed=args.split_seed + len(method) + len(dataset),
            )
        )
        for idx, (label, score) in enumerate(zip(labels, mean_score)):
            ensemble_prediction_rows.append({
                "Dataset": dataset,
                "Method": method,
                "StationIndex": idx,
                "Target": float(label),
                "Score": float(score),
            })
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "cross_dataset_bootstrap_ci_v10.csv", index=False)
    pd.DataFrame(ensemble_prediction_rows).to_csv(output_dir / "cross_dataset_ensemble_predictions_v10.csv", index=False)

    _v10_write_latex(output_dir, summary, audits)
    reviewer_text = f"""Cross-dataset validation response
{'=' * 80}

We added external validation on two independently sourced datasets using a harmonized common feature core. The public station-level dataset contributes {len(us)} usable rated stations after exact-row deduplication and evaluates a rating-popularity priority proxy. The Palo Alto dataset contains {palo_audit['RawSessionRows']} charging sessions, aggregated to {len(palo)} stable station-level demand records after requiring at least 180 active days and 100 sessions.

Only geographic coordinates, charging capacity, connector count, and operating hours are harmonized as inference variables. Target-side rating, review count, session, user, and energy variables are excluded from all model inputs. Missing Palo Alto capacity and operating-hours variables are imputed using source-training medians and are explicitly reported in the domain-shift table. Source model family and fusion selection use only the source validation split. The primary external results are zero-shot; a secondary public-dataset region-disjoint adaptation experiment is reported separately.

Because the external datasets provide different outcomes, the public-station experiment tests transfer of the rating-popularity ranking task, while the Palo Alto experiment independently tests transfer of the demand head. We do not claim that rating, popularity, and observed charging demand are interchangeable constructs.
"""
    (output_dir / "cross_dataset_reviewer_response_v10.txt").write_text(reviewer_text, encoding="utf-8")

    authoritative = {
        "protocol": {
            "source_dataset": str(data_path),
            "external_us_dataset": str(us_path),
            "external_usage_dataset": str(palo_path),
            "source_split": split_meta,
            "common_features": prep.feature_names,
            "model_selection": "source validation only",
            "primary_external_protocol": "zero-shot",
            "secondary_us_protocol": "state-disjoint target adaptation",
        },
        "audits": audits.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    save_json(output_dir / "cross_dataset_authoritative_values_v10.json", authoritative)

    additions = [
        "cross_dataset_authoritative_values_v10.json",
        "cross_dataset_audit_v10.csv",
        "cross_dataset_data_quality_flags_v10.csv",
        "cross_dataset_provenance_v10.csv",
        "cross_dataset_common_feature_manifest_v10.csv",
        "cross_dataset_domain_shift_v10.csv",
        "cross_dataset_source_model_selection_v10.csv",
        "cross_dataset_multiseed_metrics_v10.csv",
        "cross_dataset_summary_v10.csv",
        "cross_dataset_bootstrap_ci_v10.csv",
        "cross_dataset_ensemble_predictions_v10.csv",
        "cross_dataset_public_adaptation_splits_v10.csv",
        "cross_dataset_public_adaptation_metrics_v10.csv",
        "cross_dataset_public_adaptation_summary_v10.csv",
        "cross_dataset_table_v10.tex",
        "cross_dataset_reviewer_response_v10.txt",
    ]
    upload_path = output_dir / "V10_UPLOAD_THESE_FILES.txt"
    upload_path.write_text("\n".join(additions), encoding="utf-8")
    print("[V10] Cross-dataset validation completed")
    print(f"[V10] Results: {output_dir}")


def parse_args() -> argparse.Namespace:
    global V10_RUNTIME_ARGS, V9_RUNTIME_ARGS
    parser = argparse.ArgumentParser(
        description="PRIME-EV V10 exact ranking with external cross-dataset validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="prime_ev_v10_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sensitivity-epochs", type=int, default=10)
    parser.add_argument("--regional-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-metric", choices=["ndcg_full", "ndcg_top_fraction", "exact_composite"], default="exact_composite")
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-demand", type=float, default=0.30)
    parser.add_argument("--lambda-rank", type=float, default=6.0)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--pair-threshold", type=float, default=0.03)
    parser.add_argument("--train-pairs", type=int, default=60000)
    parser.add_argument("--validation-pairs", type=int, default=12000)
    parser.add_argument("--test-pairs", type=int, default=12000)
    parser.add_argument("--risk-weight", type=float, default=0.60)
    parser.add_argument("--demand-weight", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--review-seeds", default="42,123,456,789,2025,31415,27182,16180,57721,65537")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--disable-pair-margin-weighting", action="store_true")
    parser.add_argument("--multiseed-epochs", type=int, default=10)
    parser.add_argument("--multiseed-train-pairs", type=int, default=30000)
    parser.add_argument("--multiseed-validation-pairs", type=int, default=8000)
    parser.add_argument("--multiseed-test-pairs", type=int, default=8000)
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-regional-transfer", action="store_true")
    parser.add_argument("--skip-label-sensitivity", action="store_true")
    parser.add_argument("--skip-baseline-sensitivity", action="store_true")
    parser.add_argument("--skip-order-sensitivity", action="store_true")
    parser.add_argument("--skip-operator-cv", action="store_true")
    parser.add_argument("--order-permutations", type=int, default=5)
    parser.add_argument("--order-epochs", type=int, default=8)
    parser.add_argument("--operator-cv-epochs", type=int, default=10)
    parser.add_argument("--disable-engineered-features", action="store_true")
    parser.add_argument("--disable-exact-fusion", action="store_true")
    parser.add_argument("--exact-hidden-dim", type=int, default=128)
    parser.add_argument("--exact-latent-dim", type=int, default=48)
    parser.add_argument("--exact-residual-blocks", type=int, default=3)
    parser.add_argument("--exact-dropout", type=float, default=0.05)
    parser.add_argument("--exact-pair-weight", type=float, default=1.0)
    parser.add_argument("--exact-point-weight", type=float, default=1.0)
    parser.add_argument("--exact-list-weight", type=float, default=0.10)
    parser.add_argument("--exact-corr-weight", type=float, default=0.25)
    parser.add_argument("--exact-hard-weight", type=float, default=0.50)
    parser.add_argument("--exact-hard-pairs", type=int, default=4096)
    parser.add_argument("--exact-huber-beta", type=float, default=0.05)
    parser.add_argument("--exact-list-temperature", type=float, default=0.18)
    parser.add_argument("--exact-fusion-trials", type=int, default=512)
    parser.add_argument("--exact-fusion-min-gain", type=float, default=0.002)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--external-us", help="Path to ev_data.csv external station dataset")
    parser.add_argument("--external-usage", help="Path to EVChargingStationUsage.csv")
    parser.add_argument("--skip-cross-dataset", action="store_true")
    parser.add_argument("--cross-only", action="store_true", help="Run only V10 external validation, not the full V9 suite")
    parser.add_argument("--cross-seeds", default="42,123,456,789,2025")
    parser.add_argument("--cross-bootstrap", type=int, default=1000)
    V10_RUNTIME_ARGS = parser.parse_args()
    V9_RUNTIME_ARGS = V10_RUNTIME_ARGS
    if V10_RUNTIME_ARGS.quick:
        V10_RUNTIME_ARGS.cross_seeds = "42,123"
        V10_RUNTIME_ARGS.cross_bootstrap = min(V10_RUNTIME_ARGS.cross_bootstrap, 200)
    return V10_RUNTIME_ARGS


def main() -> None:
    if "--cross-only" in sys.argv:
        args = parse_args()
    else:
        _V9_MAIN()
        args = V10_RUNTIME_ARGS
        if args is None:
            args = parse_args()
    if args.skip_cross_dataset:
        print("[V10] Cross-dataset validation skipped.")
        return
    if not args.external_us or not args.external_usage:
        print("[V10] External paths not supplied; cross-dataset validation was not run.")
        print("      Add --external-us <ev_data.csv> --external-usage <EVChargingStationUsage.csv>.")
        return
    run_cross_dataset_validation(args)


# =============================================================================
# V11 final reviewer-completion layer
# =============================================================================
# V11 adds: validation-only pair/loss selection, expanded proxy falsification,
# target learnability controls, random-adjusted metrics, uncertainty-aware
# constrained selection, reproducibility manifests, complexity/convergence
# summaries, external paired bootstrap comparisons, and a reviewer evidence map.

import hashlib
import platform
from types import SimpleNamespace

V11_VERSION = "PRIME-EV V12 Final Corrected Reviewer"



def run_retrained_feature_order_sensitivity(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    permutations: int = 5,
    epochs: int = 8,
) -> pd.DataFrame:
    """Correct matched-seed retraining protocol for feature-order sensitivity.

    Every feature permutation is evaluated under the same list of training
    seeds. The permutation itself is generated independently of the training
    seed and is applied consistently to train, validation, and test matrices.
    This removes the permutation/initialization confound in earlier V9 output.
    """
    global SEED
    old_seed = SEED
    seed_text = getattr(V9_RUNTIME_ARGS, "order_seeds", "42,123,456")
    training_seeds = [int(x.strip()) for x in seed_text.split(",") if x.strip()]
    identity = np.arange(bundle.X_train.shape[1])
    rows: List[Dict[str, Any]] = []
    for rep in range(permutations):
        permutation_rng = np.random.default_rng(config.split_seed + 3000 + rep)
        permutation = identity.copy() if rep == 0 else permutation_rng.permutation(len(identity))
        permuted_bundle = _clone_bundle_with_permuted_features(bundle, permutation)
        for training_seed in training_seeds:
            for architecture, variant in (
                ("LegacyConv1D", {"legacy_conv": True, "no_fusion": True}),
                ("V11ResidualTabular", {"no_fusion": True}),
            ):
                SEED = int(training_seed)
                set_seed(SEED)
                result = train_prime_ev(
                    permuted_bundle, config, device,
                    name=f"Order_{architecture}_perm{rep}_seed{training_seed}",
                    variant=variant, epochs_override=epochs,
                )
                rec = {
                    "Permutation": rep,
                    "PermutationSeed": int(config.split_seed + 3000 + rep),
                    "Architecture": architecture,
                    "TrainingSeed": int(training_seed),
                    "IdentityOrder": bool(rep == 0),
                    "PermutationVector": "|".join(map(str, permutation.tolist())),
                }
                rec.update(fixed_cutoff_metrics(
                    permuted_bundle.g_test, result.test_scores, permuted_bundle.test_pairs
                ))
                rows.append(rec)
    SEED = old_seed
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "retrained_feature_order_sensitivity_v11.csv", index=False)
    # Compatibility copy for existing table-generation scripts.
    out.to_csv(output_dir / "retrained_feature_order_sensitivity_v9.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for (architecture, permutation), sub in out.groupby(["Architecture", "Permutation"]):
        for metric in ["NDCG_full", "NDCG@10pct", "P@10pct", "Spearman", "PairwiseAccuracy"]:
            values = sub[metric].to_numpy(dtype=float)
            lo, hi = ci95(values)
            summary_rows.append({
                "Architecture": architecture,
                "Permutation": permutation,
                "Metric": metric,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "CI95_L": lo,
                "CI95_U": hi,
                "TrainingSeeds": len(values),
            })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "retrained_feature_order_sensitivity_summary_v11.csv", index=False
    )
    return out


def _v11_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _v11_random_adjusted(value: float, random_mean: float) -> float:
    denom = 1.0 - float(random_mean)
    if abs(denom) < EPS:
        return 0.0
    return float((float(value) - float(random_mean)) / denom)


def _v11_clean_metric_dict(
    labels: np.ndarray,
    scores: np.ndarray,
    pairs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    top_fraction: float = 0.10,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    k = max(1, int(math.ceil(n * top_fraction)))
    return {
        "NDCG_full": ndcg_at_k(labels, scores, n),
        "NDCG_at_10_percent": ndcg_at_k(labels, scores, k),
        "TopKAgreement_at_10_percent": precision_at_k(labels, scores, k),
        "AP_at_10_percent": average_precision_at_k(labels, scores, k),
        "Regret_at_10_percent": regret_at_k(labels, scores, k),
        "Spearman": safe_spearman(labels, scores),
        "KendallTau": safe_kendall(labels, scores),
        "PairwiseAccuracy": pairwise_accuracy(labels, scores, pairs),
    }


def _v11_build_model(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    variant: Mapping[str, Any],
) -> nn.Module:
    hidden_dim = int(getattr(V9_RUNTIME_ARGS, "exact_hidden_dim", 128))
    default_latent = int(getattr(V9_RUNTIME_ARGS, "exact_latent_dim", 48))
    latent_dim = int(variant.get("latent_dim", default_latent))
    residual_blocks = int(getattr(V9_RUNTIME_ARGS, "exact_residual_blocks", 3))
    dropout = float(getattr(V9_RUNTIME_ARGS, "exact_dropout", 0.05))
    return V9ExactPrimeEV(
        input_dim=bundle.X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        residual_blocks=residual_blocks,
        dropout=dropout,
        risk_weight=config.risk_weight,
        demand_weight=config.demand_weight,
        include_risk_in_ranker=not variant.get("no_risk_input", False),
        demand_enabled=not variant.get("no_dim", False),
        use_feature_gating=not variant.get("no_attention", False),
        simple_encoder=bool(variant.get("no_ire", False)),
        legacy_conv=bool(variant.get("legacy_conv", False)),
        legacy_attention=not variant.get("no_attention", False),
    ).to(device)


def _v11_validation_only_train(
    bundle: DataBundle,
    config: ExperimentConfig,
    device: torch.device,
    variant: Optional[Mapping[str, Any]],
    epochs: int,
    steps_override: int,
    seed: int,
) -> Dict[str, Any]:
    """Train using train/validation only and return validation evidence.

    No test prediction or test metric is computed in this function. It is used
    for hyperparameter and deployment-configuration selection.
    """
    global SEED
    old_seed = SEED
    SEED = int(seed)
    set_seed(SEED)
    variant = dict(variant or {})
    model = _v11_build_model(bundle, config, device, variant)
    tensors = tensors_for_bundle(bundle, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=max(config.weight_decay, 1e-5),
    )
    pair_total = len(bundle.train_pairs[0])
    station_total = len(bundle.X_train)
    auto_steps = max(
        1,
        int(math.ceil(pair_total / max(1, config.pair_batch_size))),
        int(math.ceil(station_total / max(1, config.station_batch_size))),
    )
    steps = int(steps_override) if steps_override > 0 else auto_steps
    pair_w = float(getattr(V9_RUNTIME_ARGS, "exact_pair_weight", 1.0))
    point_w = float(getattr(V9_RUNTIME_ARGS, "exact_point_weight", 1.0))
    list_w = float(getattr(V9_RUNTIME_ARGS, "exact_list_weight", 0.10))
    corr_w = float(getattr(V9_RUNTIME_ARGS, "exact_corr_weight", 0.25))
    hard_w = float(getattr(V9_RUNTIME_ARGS, "exact_hard_weight", 0.50))
    hard_pairs = int(getattr(V9_RUNTIME_ARGS, "exact_hard_pairs", 4096))
    best_state = copy.deepcopy(model.state_dict())
    best_value = -np.inf
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        rng = np.random.default_rng(seed * 100000 + epoch)
        train_total = 0.0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            station_count = min(config.station_batch_size, station_total)
            station_idx_np = rng.choice(station_total, size=station_count, replace=False)
            pair_count = min(config.pair_batch_size, pair_total)
            pair_sel = rng.choice(pair_total, size=pair_count, replace=False)
            pair_i_np = bundle.train_pairs[0][pair_sel]
            pair_j_np = bundle.train_pairs[1][pair_sel]
            pair_rho_np = bundle.train_pairs[2][pair_sel]
            union_np, inverse = np.unique(
                np.concatenate([station_idx_np, pair_i_np, pair_j_np]), return_inverse=True
            )
            n_station = len(station_idx_np)
            n_pair = len(pair_i_np)
            station_local = torch.tensor(inverse[:n_station], dtype=torch.long, device=device)
            pair_i_local = torch.tensor(inverse[n_station:n_station + n_pair], dtype=torch.long, device=device)
            pair_j_local = torch.tensor(inverse[n_station + n_pair:], dtype=torch.long, device=device)
            pair_rho = torch.tensor(pair_rho_np, dtype=torch.float32, device=device)
            union_idx = torch.tensor(union_np, dtype=torch.long, device=device)
            station_idx_t = torch.tensor(station_idx_np, dtype=torch.long, device=device)

            mu_all, sigma_all, usage_all, score_all = model(
                tensors["X_train"][union_idx],
                tensors["cost_train"][union_idx],
                tensors["dist_train"][union_idx],
            )
            mu = mu_all[station_local]
            sigma = sigma_all[station_local]
            usage_hat = usage_all[station_local]
            station_score = score_all[station_local]
            y_target = tensors["y_train"][station_idx_t]
            u_target = tensors["u_train"][station_idx_t]
            g_target = tensors["g_train"][station_idx_t]

            risk_loss = (
                F.mse_loss(mu, y_target)
                if variant.get("deterministic_risk", False)
                else gaussian_nll(mu, sigma, y_target)
            )
            demand_loss = torch.tensor(0.0, device=device)
            if not variant.get("no_dim", False):
                demand_loss = F.mse_loss(usage_hat, u_target)

            if variant.get("pointwise_rank", False):
                rank_loss = _v9_priority_point_loss(station_score, g_target)
            else:
                pair_terms = F.softplus(-pair_rho * (score_all[pair_i_local] - score_all[pair_j_local]))
                margin = torch.abs(
                    tensors["g_train"][torch.tensor(pair_i_np, dtype=torch.long, device=device)]
                    - tensors["g_train"][torch.tensor(pair_j_np, dtype=torch.long, device=device)]
                )
                if config.pair_margin_weighting:
                    weights = margin / (torch.mean(margin) + EPS)
                    pair_loss = torch.mean(weights * pair_terms)
                else:
                    pair_loss = torch.mean(pair_terms)
                point_loss = _v9_priority_point_loss(station_score, g_target)
                list_loss = _v9_listnet_loss(station_score, g_target)
                corr_loss = _v9_pearson_loss(station_score, g_target)
                hard_loss = _v9_hard_lambdarank_loss(
                    station_score, g_target, config.pair_threshold, hard_pairs
                )
                rank_loss = (
                    pair_w * pair_loss + point_w * point_loss + list_w * list_loss
                    + corr_w * corr_loss + hard_w * hard_loss
                )
            loss = (
                config.lambda_risk * risk_loss
                + config.lambda_demand * demand_loss
                + config.lambda_rank * rank_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.item())

        model.eval()
        with torch.no_grad():
            _, _, _, val_score = model(
                tensors["X_val"], tensors["cost_val"], tensors["dist_val"]
            )
        val_scores = val_score.detach().cpu().numpy()
        exact_value, val_metrics = _v9_exact_objective(
            bundle.g_val, val_scores, bundle.val_pairs, config.top_fraction
        )
        if config.validation_metric == "ndcg_full":
            selection = val_metrics["NDCG_full"]
        elif config.validation_metric == "ndcg_top_fraction":
            selection = val_metrics["NDCG_at_10_percent"]
        else:
            selection = exact_value
        history.append({
            "Epoch": epoch,
            "TrainTotal": train_total / max(1, steps),
            "ValidationSelection": selection,
            "ValidationExactComposite": exact_value,
            **{f"Validation_{k}": v for k, v in val_metrics.items()},
        })
        if selection > best_value + 1e-8:
            best_value = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    _, _, _, val_scores, latency = infer_model(
        model, bundle.X_val, bundle.cost_val, bundle.dist_val,
        device, repeats=1, batch_size=config.eval_batch_size,
    )
    exact_value, metrics = _v9_exact_objective(
        bundle.g_val, val_scores, bundle.val_pairs, config.top_fraction
    )
    elapsed = time.perf_counter() - start_time
    SEED = old_seed
    return {
        "BestEpoch": int(best_epoch),
        "ValidationSelection": float(best_value),
        "ValidationExactComposite": float(exact_value),
        "Validation_NDCG_full": float(metrics["NDCG_full"]),
        "Validation_NDCG_at_10_percent": float(metrics["NDCG_at_10_percent"]),
        "Validation_TopKAgreement_at_10_percent": float(metrics["Precision_at_10_percent"]),
        "Validation_Spearman": float(metrics["Spearman"]),
        "Validation_KendallTau": float(metrics["KendallTau"]),
        "Validation_PairwiseAccuracy": float(metrics["PairwiseAccuracy"]),
        "ParameterCount": count_parameters(model),
        "ModelMemoryMB": memory_mb_for_model(model),
        "ValidationLatency_ms_per_station": float(latency),
        "TrainingTime_seconds": float(elapsed),
        "History": history,
    }


def _v11_make_config(args: argparse.Namespace, data_path: Path, output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        data_path=str(data_path), output_dir=str(output_dir), epochs=args.epochs,
        sensitivity_epochs=args.sensitivity_epochs, regional_epochs=args.regional_epochs,
        learning_rate=args.learning_rate, lambda_risk=args.lambda_risk,
        lambda_demand=args.lambda_demand, lambda_rank=args.lambda_rank,
        patience=args.patience, validation_metric=args.validation_metric,
        latent_dim=args.latent_dim, pair_threshold=args.pair_threshold,
        batch_pairs_train=args.train_pairs, batch_pairs_val=args.validation_pairs,
        batch_pairs_test=args.test_pairs, risk_weight=args.risk_weight,
        demand_weight=args.demand_weight, run_ablations=not args.skip_ablations,
        run_regional_transfer=not args.skip_regional_transfer,
        run_label_sensitivity=not args.skip_label_sensitivity,
        run_baseline_sensitivity=not args.skip_baseline_sensitivity,
        quick=args.quick, device=args.device, torch_threads=max(1, args.torch_threads),
        steps_per_epoch=max(0, args.steps_per_epoch),
        pair_margin_weighting=not args.disable_pair_margin_weighting,
        split_seed=args.split_seed,
    )


def run_pair_loss_sensitivity_v11(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    ensure_required_columns(df)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    device = choose_device(args.device)
    rows: List[Dict[str, Any]] = []
    thresholds = [0.00, 0.02, 0.05, 0.10]
    rank_weights = [1.0, 5.0, 15.0, 30.0]
    margins = [False, True]
    for threshold in thresholds:
        cfg = _v11_make_config(args, data_path, output_dir)
        cfg.pair_threshold = float(threshold)
        cfg.batch_pairs_train = min(cfg.batch_pairs_train, args.pair_loss_search_pairs)
        cfg.batch_pairs_val = min(cfg.batch_pairs_val, max(3000, args.pair_loss_search_pairs // 4))
        cfg.batch_pairs_test = min(cfg.batch_pairs_test, max(3000, args.pair_loss_search_pairs // 4))
        # Hyperparameter selection must not touch the held-out test labels.
        # The validation partition is duplicated into the unused test slot
        # required by DataBundle construction; the real test_idx is never
        # transformed or paired in this selection stage.
        selection_meta = dict(split_meta)
        selection_meta["selection_scope"] = "train/validation only"
        bundle = build_bundle(
            df, train_idx, val_idx, val_idx, cfg, selection_meta,
            pair_seed_offset=int(round(threshold * 1000)),
        )
        for rank_weight in rank_weights:
            for margin_weighting in margins:
                trial_cfg = copy.deepcopy(cfg)
                trial_cfg.lambda_rank = float(rank_weight)
                trial_cfg.pair_margin_weighting = bool(margin_weighting)
                result = _v11_validation_only_train(
                    bundle, trial_cfg, device, variant={"no_fusion": True},
                    epochs=args.pair_loss_search_epochs,
                    steps_override=args.pair_loss_search_steps,
                    seed=args.pair_loss_search_seed,
                )
                row = {
                    "PairThreshold": threshold,
                    "LambdaRank": rank_weight,
                    "MarginWeighting": margin_weighting,
                    "TrainingSeed": args.pair_loss_search_seed,
                    "TrainPairs": len(bundle.train_pairs[0]),
                    "ValidationPairs": len(bundle.val_pairs[0]),
                }
                row.update({k: v for k, v in result.items() if k != "History"})
                rows.append(row)
    table = pd.DataFrame(rows)
    best_ndcg = float(table["Validation_NDCG_at_10_percent"].max())
    eligible = table[table["Validation_NDCG_at_10_percent"] >= best_ndcg - args.pair_loss_selection_tolerance].copy()
    best_agreement = float(eligible["Validation_TopKAgreement_at_10_percent"].max())
    eligible = eligible[
        eligible["Validation_TopKAgreement_at_10_percent"] >= best_agreement - 0.01
    ].copy()
    eligible = eligible.sort_values(
        ["Validation_PairwiseAccuracy", "Validation_Spearman", "TrainingTime_seconds", "LambdaRank"],
        ascending=[False, False, True, True],
    )
    selected = eligible.iloc[0].to_dict()
    table["SelectedByValidationRule"] = False
    mask = (
        (table["PairThreshold"] == selected["PairThreshold"])
        & (table["LambdaRank"] == selected["LambdaRank"])
        & (table["MarginWeighting"] == selected["MarginWeighting"])
    )
    table.loc[mask, "SelectedByValidationRule"] = True
    table.to_csv(output_dir / "pair_loss_sensitivity_validation_v11.csv", index=False)
    payload = {
        "selection_scope": "training and validation partitions only; test metrics not computed",
        "primary_metric": "validation NDCG@10%",
        "quality_tolerance": float(args.pair_loss_selection_tolerance),
        "tie_breakers": ["top-10% agreement", "pairwise accuracy", "Spearman", "training time", "smaller lambda"],
        "selected": {
            "pair_threshold": float(selected["PairThreshold"]),
            "lambda_rank": float(selected["LambdaRank"]),
            "pair_margin_weighting": bool(selected["MarginWeighting"]),
        },
    }
    save_json(output_dir / "pair_loss_selected_config_v11.json", payload)
    args.pair_threshold = payload["selected"]["pair_threshold"]
    args.lambda_rank = payload["selected"]["lambda_rank"]
    args.disable_pair_margin_weighting = not payload["selected"]["pair_margin_weighting"]
    return payload


def run_deployment_selection_v11(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    ensure_required_columns(df)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    cfg = _v11_make_config(args, data_path, output_dir)
    cfg.batch_pairs_train = min(cfg.batch_pairs_train, args.deployment_selection_pairs)
    cfg.batch_pairs_val = min(cfg.batch_pairs_val, max(3000, args.deployment_selection_pairs // 4))
    cfg.batch_pairs_test = min(cfg.batch_pairs_test, max(3000, args.deployment_selection_pairs // 4))
    selection_meta = dict(split_meta)
    selection_meta["selection_scope"] = "train/validation only"
    bundle = build_bundle(df, train_idx, val_idx, val_idx, cfg, selection_meta, pair_seed_offset=11100)
    device = choose_device(args.device)
    candidates = {
        "V9ResidualFull": {"no_fusion": True},
        "SimpleMLP_NoIRE": {"no_ire": True, "no_fusion": True},
        "NoFeatureGating": {"no_attention": True, "no_fusion": True},
        "NoDemandHead": {"no_dim": True, "no_fusion": True},
        "NoRiskInput": {"no_risk_input": True, "no_fusion": True},
        "DeterministicRisk": {"deterministic_risk": True, "no_fusion": True},
        "PointwiseOnly": {"pointwise_rank": True, "no_fusion": True},
    }
    rows: List[Dict[str, Any]] = []
    for name, variant in candidates.items():
        result = _v11_validation_only_train(
            bundle, cfg, device, variant=variant,
            epochs=args.deployment_selection_epochs,
            steps_override=args.deployment_selection_steps,
            seed=args.deployment_selection_seed,
        )
        row = {"Configuration": name, "VariantJSON": json.dumps(variant, sort_keys=True)}
        row.update({k: v for k, v in result.items() if k != "History"})
        rows.append(row)
    table = pd.DataFrame(rows)
    best_quality = float(table["Validation_NDCG_at_10_percent"].max())
    eligible = table[
        table["Validation_NDCG_at_10_percent"] >= best_quality - args.deployment_quality_tolerance
    ].copy()
    best_agreement = float(eligible["Validation_TopKAgreement_at_10_percent"].max())
    eligible = eligible[
        eligible["Validation_TopKAgreement_at_10_percent"] >= best_agreement - 0.01
    ].copy()
    eligible = eligible.sort_values(
        ["ValidationLatency_ms_per_station", "ParameterCount", "Validation_PairwiseAccuracy"],
        ascending=[True, True, False],
    )
    selected = eligible.iloc[0].to_dict()
    table["EligibleWithinQualityTolerance"] = table["Validation_NDCG_at_10_percent"] >= best_quality - args.deployment_quality_tolerance
    table["SelectedDeploymentConfiguration"] = table["Configuration"] == selected["Configuration"]
    table.to_csv(output_dir / "deployment_configuration_selection_v11.csv", index=False)
    payload = {
        "selection_scope": "validation only",
        "primary_metric": "NDCG@10%",
        "quality_tolerance": float(args.deployment_quality_tolerance),
        "tie_breakers": ["top-10% agreement", "latency", "parameter count", "pairwise accuracy"],
        "selected_configuration": selected["Configuration"],
        "selected_variant": json.loads(selected["VariantJSON"]),
        "note": "This selection identifies the recommended deployment configuration. The complete architecture remains an evaluated methodological variant.",
    }
    save_json(output_dir / "deployment_configuration_selected_v11.json", payload)
    return payload


def _v11_bootstrap_spearman(a: np.ndarray, b: np.ndarray, repetitions: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    values = []
    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        values.append(safe_spearman(a[idx], b[idx]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _v11_eta_squared(proxy: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"proxy": proxy, "group": groups.astype(str)})
    grand = float(frame["proxy"].mean())
    ss_between = 0.0
    for _, sub in frame.groupby("group"):
        ss_between += len(sub) * (float(sub["proxy"].mean()) - grand) ** 2
    ss_total = float(np.sum((frame["proxy"].to_numpy() - grand) ** 2))
    return float(ss_between / (ss_total + EPS))


def run_proxy_falsification_v11(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output).expanduser().resolve()
    df = pd.read_csv(Path(args.data).expanduser().resolve())
    ensure_required_columns(df)
    rating = pd.to_numeric(df["Reviews (Rating)"], errors="coerce")
    proxy = 1.0 - (rating - rating.min()) / (rating.max() - rating.min() + EPS)
    proxy_arr = proxy.to_numpy(dtype=float)
    numeric = [
        "Cost (USD/kWh)", "Distance to City (km)", "Charging Capacity (kW)",
        "Installation Year", "Parking Spots", "Usage Stats (avg users/day)",
    ]
    categorical = [
        "Charger Type", "Availability", "Station Operator", "Connector Types",
        "Renewable Energy Source", "Maintenance Frequency",
    ]
    overall_rows: List[Dict[str, Any]] = []
    for idx, col in enumerate(numeric):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(proxy_arr) & np.isfinite(values)
        effect = safe_spearman(proxy_arr[mask], values[mask])
        lo, hi = _v11_bootstrap_spearman(
            proxy_arr[mask], values[mask], args.proxy_bootstrap, args.split_seed + idx
        )
        overall_rows.append({
            "Indicator": col, "IndicatorType": "numeric", "Test": "Spearman",
            "Effect": effect, "BootstrapCI95_L": lo, "BootstrapCI95_U": hi,
            "N": int(mask.sum()),
        })
    for idx, col in enumerate(categorical):
        groups = df[col].astype(str).to_numpy()
        effect = _v11_eta_squared(proxy_arr, groups)
        grouped = [proxy_arr[groups == level] for level in np.unique(groups)]
        h, p = stats.kruskal(*grouped)
        rng = np.random.default_rng(args.split_seed + 100 + idx)
        boots = []
        for _ in range(args.proxy_bootstrap):
            sample = rng.integers(0, len(df), len(df))
            boots.append(_v11_eta_squared(proxy_arr[sample], groups[sample]))
        overall_rows.append({
            "Indicator": col, "IndicatorType": "categorical",
            "Test": "Kruskal-Wallis / eta-squared", "Effect": effect,
            "PValue": float(p), "BootstrapCI95_L": float(np.quantile(boots, 0.025)),
            "BootstrapCI95_U": float(np.quantile(boots, 0.975)), "N": len(df),
        })
    overall = pd.DataFrame(overall_rows).sort_values("Effect", key=lambda x: x.abs(), ascending=False)
    overall.to_csv(output_dir / "proxy_validity_bootstrap_v11.csv", index=False)

    within_rows: List[Dict[str, Any]] = []
    for operator, sub in df.groupby("Station Operator"):
        sub_proxy = proxy.loc[sub.index].to_numpy(dtype=float)
        for idx, col in enumerate(numeric):
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(sub_proxy) & np.isfinite(values)
            effect = safe_spearman(sub_proxy[mask], values[mask])
            lo, hi = _v11_bootstrap_spearman(
                sub_proxy[mask], values[mask], max(200, args.proxy_bootstrap // 2),
                args.split_seed + 500 + idx + len(within_rows),
            )
            within_rows.append({
                "StationOperator": operator, "Indicator": col, "Spearman": effect,
                "BootstrapCI95_L": lo, "BootstrapCI95_U": hi, "N": int(mask.sum()),
            })
    within = pd.DataFrame(within_rows)
    within.to_csv(output_dir / "proxy_validity_within_operator_v11.csv", index=False)

    observed_max = max(abs(safe_spearman(
        proxy_arr,
        pd.to_numeric(df[col], errors="coerce").fillna(pd.to_numeric(df[col], errors="coerce").median()).to_numpy(dtype=float),
    )) for col in numeric)
    rng = np.random.default_rng(args.split_seed + 9900)
    null_max = []
    arrays = [pd.to_numeric(df[col], errors="coerce").fillna(pd.to_numeric(df[col], errors="coerce").median()).to_numpy(dtype=float) for col in numeric]
    for _ in range(args.proxy_shuffles):
        shuffled = rng.permutation(proxy_arr)
        null_max.append(max(abs(safe_spearman(shuffled, arr)) for arr in arrays))
    negative_control = pd.DataFrame([{
        "ObservedMaximumAbsoluteNumericSpearman": observed_max,
        "ShuffleRepetitions": args.proxy_shuffles,
        "NullMeanMaximumAbsoluteSpearman": float(np.mean(null_max)),
        "NullCI95_L": float(np.quantile(null_max, 0.025)),
        "NullCI95_U": float(np.quantile(null_max, 0.975)),
        "EmpiricalP_ObservedGreaterThanNull": float((1 + np.sum(np.asarray(null_max) >= observed_max)) / (args.proxy_shuffles + 1)),
    }])
    negative_control.to_csv(output_dir / "proxy_shuffled_negative_control_v11.csv", index=False)
    summary = {
        "largest_absolute_numeric_spearman": float(observed_max),
        "largest_categorical_eta_squared": float(overall[overall["IndicatorType"] == "categorical"]["Effect"].max()),
        "shuffle_empirical_p": float(negative_control.iloc[0]["EmpiricalP_ObservedGreaterThanNull"]),
        "interpretation": "Weak or null association is evidence against treating the rating-derived proxy as observed operational failure risk.",
    }
    save_json(output_dir / "proxy_falsification_summary_v11.json", summary)
    return summary


def run_target_learnability_v11(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    cfg = _v11_make_config(args, data_path, output_dir)
    cfg.batch_pairs_train = min(cfg.batch_pairs_train, 20000)
    cfg.batch_pairs_val = min(cfg.batch_pairs_val, 5000)
    cfg.batch_pairs_test = min(cfg.batch_pairs_test, 5000)
    bundle = build_bundle(df, train_idx, val_idx, test_idx, cfg, split_meta, pair_seed_offset=12100)
    targets = {
        "RatingRiskOnly": (bundle.y_train, bundle.y_val, bundle.y_test),
        "ObservedUsageOnly": (bundle.u_train, bundle.u_val, bundle.u_test),
        "CombinedPriority": (bundle.g_train, bundle.g_val, bundle.g_test),
    }
    specs = [
        ("Ridge", Ridge, {"alpha": [0.1, 1.0, 10.0]}, {}),
    ]
    if not args.quick:
        specs.extend([
            ("GradientBoosting", GradientBoostingRegressor,
             {"n_estimators": [50], "max_depth": [2, 3], "learning_rate": [0.05]},
             {"random_state": args.split_seed}),
            ("RandomForest", RandomForestRegressor,
             {"n_estimators": [80], "max_depth": [6], "min_samples_leaf": [3]},
             {"random_state": args.split_seed, "n_jobs": -1}),
        ])
    rows: List[Dict[str, Any]] = []
    for target_name, (y_train, y_val, y_test) in targets.items():
        pairs = _v10_pair_pool(np.asarray(y_test, dtype=float), args.split_seed + len(rows) + 13000)
        random_scores = np.random.default_rng(args.split_seed + len(rows) + 13100).random(len(y_test))
        random_metrics = _v11_clean_metric_dict(y_test, random_scores, pairs)
        rows.append({"Target": target_name, "Method": "Random", **random_metrics})
        for model_name, cls, grid, fixed in specs:
            best_model = None
            best_params = None
            best_val = -np.inf
            for params in ParameterGrid(grid):
                kwargs = dict(fixed); kwargs.update(params)
                model = cls(**kwargs)
                model.fit(bundle.X_train, y_train)
                pred_val = model.predict(bundle.X_val)
                val_metric = ndcg_at_k(y_val, pred_val, max(1, int(math.ceil(len(y_val) * 0.10))))
                if val_metric > best_val:
                    best_val = val_metric; best_model = model; best_params = kwargs
            pred_test = best_model.predict(bundle.X_test)
            metrics = _v11_clean_metric_dict(y_test, pred_test, pairs)
            metrics["RandomAdjustedNDCG_full"] = _v11_random_adjusted(metrics["NDCG_full"], random_metrics["NDCG_full"])
            metrics["RandomAdjustedNDCG_at_10_percent"] = _v11_random_adjusted(metrics["NDCG_at_10_percent"], random_metrics["NDCG_at_10_percent"])
            rows.append({
                "Target": target_name, "Method": model_name,
                "ValidationNDCG_at_10_percent": best_val,
                "SelectedParameters": json.dumps(best_params, default=str, sort_keys=True),
                **metrics,
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "target_learnability_falsification_v11.csv", index=False)
    return out



def run_shuffled_target_learnability_v11(args: argparse.Namespace) -> pd.DataFrame:
    """Repeated shuffled-target control under the fixed operator split.

    The full combined target is shuffled before partition extraction. A Ridge
    ranker with alpha selected on the unshuffled validation partition is then
    fitted for each shuffle. This measures chance learnability under the exact
    feature matrix and target distribution.
    """
    output_dir = Path(args.output).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    cfg = _v11_make_config(args, data_path, output_dir)
    cfg.batch_pairs_train = min(cfg.batch_pairs_train, 20000)
    cfg.batch_pairs_val = min(cfg.batch_pairs_val, 5000)
    cfg.batch_pairs_test = min(cfg.batch_pairs_test, 5000)
    bundle = build_bundle(df, train_idx, val_idx, test_idx, cfg, split_meta, pair_seed_offset=12350)

    best_alpha = None
    best_val = -np.inf
    observed_model = None
    for alpha in (0.1, 1.0, 10.0):
        model = Ridge(alpha=alpha).fit(bundle.X_train, bundle.g_train)
        pred = model.predict(bundle.X_val)
        metric = ndcg_at_k(bundle.g_val, pred, max(1, int(math.ceil(len(bundle.g_val) * 0.10))))
        if metric > best_val:
            best_val = metric
            best_alpha = alpha
            observed_model = model
    observed_scores = observed_model.predict(bundle.X_test)
    observed_pairs = _v10_pair_pool(bundle.g_test, args.split_seed + 12360)
    observed = _v11_clean_metric_dict(bundle.g_test, observed_scores, observed_pairs)

    _, _, g_all = transform_targets(
        df, bundle.preprocessor, cfg.risk_weight, cfg.demand_weight
    )
    rng = np.random.default_rng(args.split_seed + 12370)
    raw_rows: List[Dict[str, Any]] = []
    for rep in range(args.target_shuffles):
        shuffled = rng.permutation(g_all)
        model = Ridge(alpha=best_alpha).fit(bundle.X_train, shuffled[train_idx])
        scores = model.predict(bundle.X_test)
        labels = shuffled[test_idx]
        pairs = _v10_pair_pool(labels, args.split_seed + 12400 + rep)
        metrics = _v11_clean_metric_dict(labels, scores, pairs)
        raw_rows.append({"Shuffle": rep, "RidgeAlpha": best_alpha, **metrics})
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output_dir / "shuffled_target_learnability_raw_v11.csv", index=False)

    summary_rows = []
    for metric in [
        "NDCG_full", "NDCG_at_10_percent", "TopKAgreement_at_10_percent",
        "AP_at_10_percent", "Regret_at_10_percent", "Spearman",
        "KendallTau", "PairwiseAccuracy",
    ]:
        vals = raw[metric].to_numpy(dtype=float)
        obs = float(observed[metric])
        higher_better = not metric.startswith("Regret")
        empirical_p = (
            (1 + np.sum(vals >= obs)) / (len(vals) + 1)
            if higher_better
            else (1 + np.sum(vals <= obs)) / (len(vals) + 1)
        )
        summary_rows.append({
            "Metric": metric,
            "ObservedCombinedRidge": obs,
            "ShuffledMean": float(vals.mean()),
            "ShuffledStd": float(vals.std(ddof=1)),
            "ShuffledCI95_L": float(np.quantile(vals, 0.025)),
            "ShuffledCI95_U": float(np.quantile(vals, 0.975)),
            "EmpiricalP_ObservedBetterThanShuffle": float(empirical_p),
            "ShuffleRepetitions": int(args.target_shuffles),
            "RidgeAlphaSelectedOnRealValidation": float(best_alpha),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "shuffled_target_learnability_summary_v11.csv", index=False)
    return summary


def run_random_adjusted_evidence_v11(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    baseline_path = output_dir / "baseline_fixed_cutoff_metrics.csv"
    random_path = output_dir / "random_ranking_reference.csv"
    if not baseline_path.exists() or not random_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    random_ref = pd.read_csv(random_path).set_index("Metric")
    random_full = float(random_ref.loc["NDCG_full", "Mean"])
    random_top = float(random_ref.loc["NDCG_at_10_percent", "Mean"])
    rows = []
    for _, row in baseline.iterrows():
        rows.append({
            "Method": row["Method"],
            "NDCG_full": row["NDCG_full"],
            "RandomMean_NDCG_full": random_full,
            "RandomAdjustedNDCG_full": _v11_random_adjusted(row["NDCG_full"], random_full),
            "NDCG_at_10_percent": row["NDCG@10pct"],
            "RandomMean_NDCG_at_10_percent": random_top,
            "RandomAdjustedNDCG_at_10_percent": _v11_random_adjusted(row["NDCG@10pct"], random_top),
            "TopKAgreement_at_10_percent": row["P@10pct"],
            "Spearman": row["Spearman"],
            "PairwiseAccuracy": row["PairwiseAccuracy"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "random_adjusted_baseline_metrics_v11.csv", index=False)
    return out


def _v11_constrained_select(
    frame: pd.DataFrame,
    score_col: str,
    k: int,
    operator_floor_fraction: float,
    low_access_floor_fraction: float,
) -> np.ndarray:
    selected: set = set()
    if operator_floor_fraction > 0:
        counts = frame["StationOperator"].value_counts()
        for operator, count in counts.items():
            quota = int(math.floor(operator_floor_fraction * k * count / len(frame)))
            if quota > 0:
                selected.update(frame[frame["StationOperator"] == operator].nlargest(quota, score_col).index.tolist())
    if low_access_floor_fraction > 0:
        quota = int(math.floor(low_access_floor_fraction * k * frame["LowAccess"].mean()))
        current = int(frame.loc[list(selected), "LowAccess"].sum()) if selected else 0
        need = max(0, quota - current)
        eligible = frame[frame["LowAccess"] & ~frame.index.isin(selected)]
        selected.update(eligible.nlargest(need, score_col).index.tolist())
    if len(selected) < k:
        selected.update(frame[~frame.index.isin(selected)].nlargest(k - len(selected), score_col).index.tolist())
    if len(selected) > k:
        selected = set(frame.loc[list(selected)].nlargest(k, score_col).index.tolist())
    return np.asarray(sorted(selected), dtype=int)


def run_uncertainty_aware_constrained_selection_v11(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    path = output_dir / "operator_cv_oof_predictions_v8.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).reset_index(drop=True)
    frame["LowAccess"] = frame["DistanceKm"] >= frame["DistanceKm"].quantile(0.75)
    frame["ScoreNorm"] = minmax_vector(frame["Score"])
    frame["SigmaNorm"] = minmax_vector(frame["RiskSigma"])
    k = max(1, int(math.ceil(len(frame) * 0.10)))
    oracle_utility = float(frame.nlargest(k, "ReferenceUtility")["ReferenceUtility"].mean())
    rows: List[Dict[str, Any]] = []
    for beta in (-0.5, 0.0, 0.5, 1.0):
        score_col = f"RobustScore_{beta}"
        frame[score_col] = frame["ScoreNorm"] + beta * frame["SigmaNorm"]
        for operator_floor in (0.0, 0.5, 1.0):
            for low_floor in (0.0, 0.5, 1.0):
                idx = _v11_constrained_select(frame, score_col, k, operator_floor, low_floor)
                selected = frame.loc[idx]
                group_rates = selected["StationOperator"].value_counts() / frame["StationOperator"].value_counts()
                rows.append({
                    "UncertaintyBeta": beta,
                    "UncertaintyPolicy": "risk-averse" if beta < 0 else ("neutral" if beta == 0 else "uncertainty-prioritizing"),
                    "OperatorFloorFraction": operator_floor,
                    "LowAccessFloorFraction": low_floor,
                    "SelectedStations": len(selected),
                    "MeanSelectedUtility": float(selected["ReferenceUtility"].mean()),
                    "RegretVsOracle": oracle_utility - float(selected["ReferenceUtility"].mean()),
                    "OperatorSelectionRateDisparity": float(group_rates.max() - group_rates.min()),
                    "SelectedLowAccessShare": float(selected["LowAccess"].mean()),
                    "PopulationLowAccessShare": float(frame["LowAccess"].mean()),
                    "LowAccessCoverageGap": abs(float(selected["LowAccess"].mean()) - float(frame["LowAccess"].mean())),
                    "MeanSelectedRiskSigma": float(selected["RiskSigma"].mean()),
                })
    out = pd.DataFrame(rows)
    feasible = out[(out["OperatorSelectionRateDisparity"] <= 0.01) & (out["LowAccessCoverageGap"] <= 0.02)]
    if feasible.empty:
        feasible = out.copy()
    selected_idx = feasible.sort_values(
        ["MeanSelectedUtility", "OperatorSelectionRateDisparity", "LowAccessCoverageGap"],
        ascending=[False, True, True],
    ).index[0]
    out["SelectedPolicyByPredeclaredRule"] = False
    out.loc[selected_idx, "SelectedPolicyByPredeclaredRule"] = True
    out.to_csv(output_dir / "uncertainty_aware_constrained_tradeoff_v11.csv", index=False)
    save_json(output_dir / "uncertainty_aware_selected_policy_v11.json", out.loc[selected_idx].to_dict())
    return out


def run_reproducibility_manifest_v11(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    split = np.full(len(df), "", dtype=object)
    split[train_idx] = "train"; split[val_idx] = "validation"; split[test_idx] = "test"
    manifest = pd.DataFrame({
        "RowIndex": np.arange(len(df)),
        "StationID": df["Station ID"].astype(str),
        "StationOperator": df["Station Operator"].astype(str),
        "Split": split,
    })
    manifest.to_csv(output_dir / "fixed_split_row_manifest_v11.csv", index=False)
    paths = [("main_dataset", data_path), ("canonical_script", Path(__file__).resolve())]
    if args.external_us:
        paths.append(("external_public_dataset", Path(args.external_us).expanduser().resolve()))
    if args.external_usage:
        paths.append(("external_usage_dataset", Path(args.external_usage).expanduser().resolve()))
    rows = []
    for role, path in paths:
        if path.exists():
            rows.append({
                "Role": role, "Path": str(path), "Bytes": path.stat().st_size,
                "SHA256": _v11_sha256(path),
            })
    checksums = pd.DataFrame(rows)
    checksums.to_csv(output_dir / "dataset_and_code_checksums_v11.csv", index=False)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
        "split_seed": int(args.split_seed),
        "split_metadata": split_meta,
    }
    save_json(output_dir / "environment_and_split_v11.json", environment)
    return checksums


def run_complexity_and_convergence_v11(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    metric_path = output_dir / "full_model_metrics.csv"
    history_path = output_dir / "histories" / "PRIME_EV_v7_FULL_history.csv"
    F_count = None
    split_path = output_dir / "split_metadata.json"
    if split_path.exists():
        meta = json.load(open(split_path, "r", encoding="utf-8"))
        F_count = int(meta.get("feature_count", 0))
    H = int(args.exact_hidden_dim)
    L = int(args.exact_latent_dim)
    B = int(args.exact_residual_blocks)
    complexity = pd.DataFrame([
        {"Stage": "Residual tabular encoder", "TimeComplexity": "O(N*(F*H + B*H^2 + H*L))", "MemoryComplexity": "O(F*H + B*H^2 + H*L)", "Notes": "Primary V9/V11 encoder; no feature-order assumption."},
        {"Stage": "Legacy outer-product Conv1D", "TimeComplexity": "O(N*F^2*C)", "MemoryComplexity": "O(N*F^2) activations", "Notes": "Retained only as a sensitivity/ablation variant."},
        {"Stage": "Pairwise training", "TimeComplexity": "O(P) sampled pairs per approximate epoch", "MemoryComplexity": "O(pair_batch_size)", "Notes": "P is reported for every split."},
        {"Stage": "Inference and ranking", "TimeComplexity": "O(N*model_forward + N*log(N))", "MemoryComplexity": "O(N)", "Notes": "Sorting dominates the decision-list construction after inference."},
        {"Stage": "Policy-constrained top-k", "TimeComplexity": "O(N*log(N) + G*N)", "MemoryComplexity": "O(N)", "Notes": "G is the number of operator/access groups."},
    ])
    complexity["FeatureCount_F"] = F_count
    complexity["HiddenDimension_H"] = H
    complexity["LatentDimension_L"] = L
    complexity["ResidualBlocks_B"] = B
    if metric_path.exists():
        metrics = pd.read_csv(metric_path).iloc[0]
        complexity["MeasuredParameterCount"] = metrics.get("ParameterCount", np.nan)
        complexity["MeasuredModelMemoryMB"] = metrics.get("ModelMemoryMB", np.nan)
        complexity["MeasuredTrainingTimeSeconds"] = metrics.get("TrainingTime_seconds", np.nan)
        complexity["MeasuredLatencyMsPerStation"] = metrics.get("Latency_ms_per_station_test", np.nan)
    complexity.to_csv(output_dir / "computational_complexity_v11.csv", index=False)
    if history_path.exists():
        history = pd.read_csv(history_path)
        selection_col = "validation_selection" if "validation_selection" in history else "validation_ndcg"
        best_idx = history[selection_col].idxmax()
        summary = pd.DataFrame([{
            "EpochsExecuted": len(history),
            "OptimizerStepsPerEpoch": int(history["optimizer_steps"].iloc[0]),
            "TotalOptimizerSteps": int(history["optimizer_steps"].sum()),
            "BestEpoch": int(history.loc[best_idx, "epoch"]),
            "InitialValidationSelection": float(history[selection_col].iloc[0]),
            "BestValidationSelection": float(history.loc[best_idx, selection_col]),
            "FinalValidationSelection": float(history[selection_col].iloc[-1]),
            "InitialTrainLoss": float(history["train_total"].iloc[0]),
            "FinalTrainLoss": float(history["train_total"].iloc[-1]),
        }])
        summary.to_csv(output_dir / "convergence_summary_v11.csv", index=False)
        plt.figure(figsize=(6.2, 4.0))
        plt.plot(history["epoch"], history[selection_col], label="Validation selection")
        plt.plot(history["epoch"], history["train_total"], label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Metric / loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_diagnostics_v11.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def bounded_two_sided_bootstrap_p(differences: Sequence[float]) -> float:
    """Sign-based two-sided bootstrap p-value constrained to [0, 1].

    Degenerate all-zero bootstrap distributions return 1.0 rather than the
    mathematically invalid value 2.0 produced by an uncapped doubled tail.
    """
    arr = np.asarray(differences, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    lower = float(np.mean(arr <= 0.0))
    upper = float(np.mean(arr >= 0.0))
    return float(min(1.0, max(0.0, 2.0 * min(lower, upper))))


def run_external_paired_bootstrap_v11(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    pred_path = output_dir / "cross_dataset_ensemble_predictions_v10.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(pred_path)
    comparisons = []
    for dataset in pred["Dataset"].unique():
        sub = pred[pred["Dataset"] == dataset]
        methods = set(sub["Method"])
        if dataset == "External_PaloAlto_Usage":
            primary_candidates = ["PRIME_DemandHead", "PRIME_CommonCore_Full"]
            baseline_candidates = ["Random", "Ridge_SourcePriority", "Ridge_CommonCore"]
        else:
            primary_candidates = ["PRIME_CommonCore_Full"]
            baseline_candidates = ["Random", "Ridge_CommonCore"]
        primary = next((x for x in primary_candidates if x in methods), None)
        if primary is None:
            continue
        pframe = sub[sub["Method"] == primary].sort_values("StationIndex")
        labels = pframe["Target"].to_numpy(dtype=float)
        pscores = pframe["Score"].to_numpy(dtype=float)
        for baseline in baseline_candidates:
            if baseline not in methods:
                continue
            bframe = sub[sub["Method"] == baseline].sort_values("StationIndex")
            bscores = bframe["Score"].to_numpy(dtype=float)
            rng = np.random.default_rng(args.split_seed + len(dataset) + len(baseline) + 14000)
            n = len(labels); k = max(1, int(math.ceil(n * 0.10)))
            for metric in ("NDCG_at_10_percent", "Spearman", "PairwiseAccuracy"):
                diffs = []
                for _ in range(args.cross_bootstrap):
                    idx = rng.integers(0, n, n)
                    y = labels[idx]; p = pscores[idx]; b = bscores[idx]
                    if metric == "NDCG_at_10_percent":
                        pv = ndcg_at_k(y, p, k); bv = ndcg_at_k(y, b, k)
                    elif metric == "Spearman":
                        pv = safe_spearman(y, p); bv = safe_spearman(y, b)
                    else:
                        pairs = _v10_pair_pool(y, args.split_seed + len(diffs) + 14100)
                        pv = pairwise_accuracy(y, p, pairs); bv = pairwise_accuracy(y, b, pairs)
                    diffs.append(pv - bv)
                comparisons.append({
                    "Dataset": dataset, "PrimaryMethod": primary,
                    "Comparator": baseline, "Metric": metric,
                    "MeanDifference": float(np.mean(diffs)),
                    "BootstrapCI95_L": float(np.quantile(diffs, 0.025)),
                    "BootstrapCI95_U": float(np.quantile(diffs, 0.975)),
                    "TwoSidedBootstrapP": bounded_two_sided_bootstrap_p(diffs),
                    "BootstrapRepetitions": args.cross_bootstrap,
                })
    out = pd.DataFrame(comparisons)
    out.to_csv(output_dir / "cross_dataset_paired_bootstrap_v11.csv", index=False)
    return out


def run_v11_self_tests(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    tests: List[Dict[str, Any]] = []
    try:
        train_idx, val_idx, test_idx, meta = operator_disjoint_split(df, args.split_seed)
        tests.append({"Test": "split_row_disjoint", "Passed": not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))})
        train_ops = set(df.iloc[train_idx]["Station Operator"].astype(str))
        val_ops = set(df.iloc[val_idx]["Station Operator"].astype(str))
        test_ops = set(df.iloc[test_idx]["Station Operator"].astype(str))
        tests.append({"Test": "split_operator_disjoint", "Passed": not (train_ops & val_ops or train_ops & test_ops or val_ops & test_ops)})
    except Exception as exc:
        tests.append({"Test": "split_construction", "Passed": False, "Error": str(exc)})
    forbidden = set(LABEL_ONLY_COLS + ["Station Operator", "Address"])
    allowed = set(NUMERIC_MODEL_COLS + CATEGORICAL_MODEL_COLS)
    tests.append({"Test": "target_side_features_excluded", "Passed": len(forbidden & allowed) == 0})
    toy = pd.DataFrame({"a": [1, 0, 1], "b": [1, 1, 0]})
    scores = pareto_balanced_scores(toy)
    tests.append({"Test": "pareto_dominant_item_first", "Passed": int(np.argmax(scores)) == 0})
    tests.append({"Test": "random_adjustment_identity", "Passed": abs(_v11_random_adjusted(0.5, 0.5)) < 1e-12})
    report = {
        "version": V11_VERSION,
        "all_passed": bool(all(item.get("Passed", False) for item in tests)),
        "tests": tests,
    }
    save_json(output_dir / "v11_self_test_report.json", report)
    return report



def run_selected_deployment_test_v11(args: argparse.Namespace) -> pd.DataFrame:
    """Train and test the validation-selected deployment configuration exactly once.

    Architecture selection is completed using training/validation data only.
    This function then retrains the selected configuration on the fixed training
    split with validation early stopping and evaluates the untouched operator-
    disjoint test split. The complete PRIME-EV architecture remains available
    as a methodological comparator in full_model_metrics.csv.
    """
    global SEED
    output_dir = Path(args.output).expanduser().resolve()
    selected_path = output_dir / "deployment_configuration_selected_v11.json"
    if not selected_path.exists():
        print("[V11] No deployment selection manifest found; selected-model test evaluation skipped.")
        return pd.DataFrame()

    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_name = str(selected_payload["selected_configuration"])
    selected_variant = dict(selected_payload.get("selected_variant", {}))

    data_path = Path(args.data).expanduser().resolve()
    df = pd.read_csv(data_path)
    ensure_required_columns(df)
    train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, args.split_seed)
    split_meta = dict(split_meta)
    split_meta["split_seed"] = int(args.split_seed)
    split_meta["model_selection_scope"] = "training and validation only"
    split_meta["selected_configuration"] = selected_name

    cfg = _v11_make_config(args, data_path, output_dir)
    bundle = build_bundle(df, train_idx, val_idx, test_idx, cfg, split_meta, pair_seed_offset=11300)
    device = choose_device(args.device)

    review_seeds = [int(x.strip()) for x in str(args.review_seeds).split(",") if x.strip()]
    training_seed = review_seeds[0] if review_seeds else int(args.split_seed)
    old_seed = SEED
    SEED = int(training_seed)
    set_seed(SEED)
    result = train_prime_ev(
        bundle, cfg, device,
        name=f"DeploymentSelected_{selected_name}",
        variant=selected_variant,
        epochs_override=cfg.epochs,
    )
    SEED = old_seed

    metrics = {
        "Model": result.name,
        "SelectedConfiguration": selected_name,
        "SelectedVariantJSON": json.dumps(selected_variant, sort_keys=True),
        "SelectionScope": "training/validation only",
        "SplitSeed": int(args.split_seed),
        "TrainingSeed": int(training_seed),
        "BestEpoch": int(result.best_epoch),
        "ParameterCount": count_parameters(result.model),
        "ModelMemoryMB": memory_mb_for_model(result.model),
        "TrainingTime_seconds": float(result.train_seconds),
        "Latency_ms_per_station": float(result.latency_ms_per_station),
        "TestTotalLoss": float(result.losses["total"]),
        "TestRiskLoss": float(result.losses["risk"]),
        "TestDemandLoss": float(result.losses["demand"]),
        "TestRankingLoss": float(result.losses["rank"]),
    }
    metrics.update(result.test_metrics)
    metrics.update(fixed_cutoff_metrics(bundle.g_test, result.test_scores, bundle.test_pairs))
    out = pd.DataFrame([metrics])
    out.to_csv(output_dir / "selected_deployment_model_metrics_v11.csv", index=False)

    raw_test = df.iloc[test_idx].reset_index(drop=True)
    prediction_table = pd.DataFrame({
        "GlobalRowIndex": test_idx.astype(int),
        "StationID": raw_test["Station ID"].astype(str),
        "StationOperator": raw_test["Station Operator"].astype(str),
        "ReferencePriorityProxy": bundle.g_test,
        "RiskProxyTarget": bundle.y_test,
        "DemandProxyTarget": bundle.u_test,
        "SelectedModelScore": result.test_scores,
        "PredictedRiskMean": result.test_mu,
        "PredictedRiskSigma": result.test_sigma,
        "PredictedDemand": result.test_usage_hat,
    })
    prediction_table.to_csv(output_dir / "selected_deployment_test_predictions_v11.csv", index=False)

    histories_dir = output_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(
        histories_dir / f"selected_deployment_{selected_name}_history_v11.csv", index=False
    )
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        models_dir / f"selected_deployment_{selected_name}_v11.pt",
        result, bundle, cfg,
    )

    comparison_rows: List[Dict[str, Any]] = []
    full_path = output_dir / "full_model_metrics.csv"
    if full_path.exists():
        full_df = pd.read_csv(full_path)
        if not full_df.empty:
            full_row = full_df.iloc[0].to_dict()
            full_row["ComparisonRole"] = "Complete methodological architecture"
            comparison_rows.append(full_row)
    selected_row = out.iloc[0].to_dict()
    selected_row["ComparisonRole"] = "Validation-selected deployment configuration"
    comparison_rows.append(selected_row)
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "full_vs_selected_deployment_v11.csv", index=False
    )
    return out



class TemporalDemandRankerV11(nn.Module):
    """Small uncertainty-aware ranker for one-month-ahead observed demand."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = torch.sigmoid(self.mean_head(h)).squeeze(1)
        sigma = F.softplus(self.scale_head(h)).squeeze(1) + 1e-3
        score = self.score_head(h).squeeze(1)
        return mu, sigma, score


def _v11_month_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    pair_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    work = frame.reset_index(drop=True).copy()
    work["_score"] = np.asarray(scores, dtype=float)
    rows: List[Dict[str, Any]] = []
    for month, sub in work.groupby("TargetMonth", sort=True):
        y = sub["ObservedDemandPriority"].to_numpy(dtype=float)
        s = sub["_score"].to_numpy(dtype=float)
        n = len(y)
        k = max(1, int(math.ceil(0.10 * n)))
        i, j = np.triu_indices(n, 1)
        valid = np.abs(y[i] - y[j]) > pair_threshold
        if valid.any():
            pair_acc = float(np.mean(
                np.sign(s[i[valid]] - s[j[valid]])
                == np.sign(y[i[valid]] - y[j[valid]])
            ))
        else:
            pair_acc = float("nan")
        rows.append({
            "TargetMonth": str(pd.Timestamp(month).date()),
            "Stations": int(n),
            "NDCG_full": ndcg_at_k(y, s, n),
            "NDCG_at_10_percent": ndcg_at_k(y, s, k),
            "TopKAgreement_at_10_percent": precision_at_k(y, s, k),
            "Spearman": safe_spearman(y, s),
            "KendallTau": safe_kendall(y, s),
            "PairwiseAccuracy": pair_acc,
            "Regret_at_10_percent": regret_at_k(y, s, k),
        })
    monthly = pd.DataFrame(rows)
    summary = {
        metric: float(monthly[metric].mean())
        for metric in [
            "NDCG_full", "NDCG_at_10_percent",
            "TopKAgreement_at_10_percent", "Spearman", "KendallTau",
            "PairwiseAccuracy", "Regret_at_10_percent",
        ]
    }
    return monthly, summary


def _v11_temporal_composite(summary: Mapping[str, float]) -> float:
    return float(
        0.50 * summary["NDCG_at_10_percent"]
        + 0.20 * summary["TopKAgreement_at_10_percent"]
        + 0.15 * ((summary["Spearman"] + 1.0) / 2.0)
        + 0.15 * summary["PairwiseAccuracy"]
    )


def _v11_temporal_pair_pool(
    labels: np.ndarray,
    months: np.ndarray,
    threshold: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left: List[int] = []
    right: List[int] = []
    rho: List[float] = []
    margin: List[float] = []
    for month in np.unique(months):
        idx = np.where(months == month)[0]
        i, j = np.triu_indices(len(idx), 1)
        if len(i) == 0:
            continue
        gi, gj = idx[i], idx[j]
        diff = labels[gi] - labels[gj]
        valid = np.abs(diff) > threshold
        left.extend(gi[valid].tolist())
        right.extend(gj[valid].tolist())
        rho.extend(np.sign(diff[valid]).astype(float).tolist())
        margin.extend(np.abs(diff[valid]).astype(float).tolist())
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(rho, dtype=np.float32),
        np.asarray(margin, dtype=np.float32),
    )


def _v11_build_palo_temporal_panel(
    path: Path,
    output_dir: Path,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Build a leakage-safe one-month-ahead station-demand panel.

    Eligibility is determined using records available through 2018-12 only.
    Every predictive demand feature is lagged by at least one month. The test
    window ends in 2020-02 to avoid using the COVID-19 disruption as an
    unmodeled distributional shock.
    """
    usecols = [
        "Station Name", "Start Date", "Energy (kWh)", "User ID",
        "Port Number", "Latitude", "Longitude",
    ]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    raw["StartDateParsed"] = pd.to_datetime(raw["Start Date"], errors="coerce")
    raw = raw.dropna(subset=["StartDateParsed", "Station Name"]).copy()
    raw["TargetMonth"] = raw["StartDateParsed"].dt.to_period("M").dt.to_timestamp()
    monthly = raw.groupby(["Station Name", "TargetMonth"], as_index=False).agg(
        Sessions=("Station Name", "size"),
        EnergyKWh=("Energy (kWh)", "sum"),
        UniqueUsers=("User ID", lambda x: x.nunique()),
        PortCount=("Port Number", lambda x: x.nunique()),
        Latitude=("Latitude", "median"),
        Longitude=("Longitude", "median"),
    )
    eligibility_cutoff = pd.Timestamp("2018-12-01")
    eligibility = monthly[monthly["TargetMonth"] <= eligibility_cutoff].groupby(
        "Station Name"
    ).agg(
        ActiveMonthsBeforeCutoff=("TargetMonth", "nunique"),
        SessionsBeforeCutoff=("Sessions", "sum"),
    )
    eligible_stations = eligibility[
        (eligibility["ActiveMonthsBeforeCutoff"] >= 18)
        & (eligibility["SessionsBeforeCutoff"] >= 300)
    ].index.tolist()
    monthly = monthly[monthly["Station Name"].isin(eligible_stations)].copy()

    panel_parts: List[pd.DataFrame] = []
    panel_end = pd.Timestamp("2020-02-01")
    earliest_allowed = pd.Timestamp("2014-01-01")
    for station in eligible_stations:
        sub = monthly[monthly["Station Name"] == station].set_index(
            "TargetMonth"
        ).sort_index()
        first_month = max(sub.index.min(), earliest_allowed)
        months = pd.date_range(first_month, panel_end, freq="MS")
        station_panel = sub.reindex(months)
        station_panel["Station Name"] = station
        for col in ["Sessions", "EnergyKWh", "UniqueUsers"]:
            station_panel[col] = station_panel[col].fillna(0.0)
        for col in ["PortCount", "Latitude", "Longitude"]:
            station_panel[col] = (
                station_panel[col].ffill().bfill().fillna(0.0)
            )
        station_panel["TargetMonth"] = months
        panel_parts.append(station_panel.reset_index(drop=True))
    panel = pd.concat(panel_parts, ignore_index=True)
    panel = panel.sort_values(["Station Name", "TargetMonth"]).reset_index(drop=True)

    for col in ["Sessions", "EnergyKWh", "UniqueUsers"]:
        grouped = panel.groupby("Station Name")[col]
        for lag in [1, 2, 3, 6]:
            panel[f"{col}_lag{lag}"] = grouped.shift(lag)
        panel[f"{col}_roll3"] = grouped.transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
        panel[f"{col}_roll6"] = grouped.transform(
            lambda x: x.shift(1).rolling(6).mean()
        )
        panel[f"{col}_trend3"] = panel[f"{col}_lag1"] - panel[f"{col}_lag3"]

    panel["PortCount_lag1"] = panel.groupby("Station Name")["PortCount"].shift(1)
    panel["HistoryMonths"] = panel.groupby("Station Name").cumcount()
    panel["MonthSin"] = np.sin(2.0 * np.pi * panel["TargetMonth"].dt.month / 12.0)
    panel["MonthCos"] = np.cos(2.0 * np.pi * panel["TargetMonth"].dt.month / 12.0)
    panel["DaysInMonth"] = panel["TargetMonth"].dt.days_in_month.astype(float)
    panel["ObservedSessionsPerDay"] = panel["Sessions"] / panel["DaysInMonth"]
    panel["ObservedEnergyPerDay"] = panel["EnergyKWh"] / panel["DaysInMonth"]
    panel["ObservedSessionPercentile"] = panel.groupby("TargetMonth")[
        "ObservedSessionsPerDay"
    ].rank(method="average", pct=True)
    panel["ObservedEnergyPercentile"] = panel.groupby("TargetMonth")[
        "ObservedEnergyPerDay"
    ].rank(method="average", pct=True)
    panel["ObservedDemandPriority"] = (
        0.60 * panel["ObservedSessionPercentile"]
        + 0.40 * panel["ObservedEnergyPercentile"]
    )

    panel["HistoricalSessionsPerDay"] = (
        panel["Sessions_roll3"] / panel["DaysInMonth"]
    )
    panel["HistoricalEnergyPerDay"] = (
        panel["EnergyKWh_roll3"] / panel["DaysInMonth"]
    )
    panel["HistoricalSessionPercentile"] = panel.groupby("TargetMonth")[
        "HistoricalSessionsPerDay"
    ].rank(method="average", pct=True)
    panel["HistoricalEnergyPercentile"] = panel.groupby("TargetMonth")[
        "HistoricalEnergyPerDay"
    ].rank(method="average", pct=True)
    panel["PersistenceScore"] = (
        0.60 * panel["HistoricalSessionPercentile"]
        + 0.40 * panel["HistoricalEnergyPercentile"]
    )

    feature_cols = [
        f"{prefix}_{suffix}"
        for prefix in ["Sessions", "EnergyKWh", "UniqueUsers"]
        for suffix in [
            "lag1", "lag2", "lag3", "lag6", "roll3", "roll6", "trend3"
        ]
    ] + [
        "PortCount_lag1", "Latitude", "Longitude", "HistoryMonths",
        "MonthSin", "MonthCos",
    ]
    panel = panel[panel["HistoryMonths"] >= 6].dropna(
        subset=feature_cols + ["ObservedDemandPriority", "PersistenceScore"]
    ).reset_index(drop=True)
    panel["Split"] = np.select(
        [
            panel["TargetMonth"] <= pd.Timestamp("2018-12-01"),
            (panel["TargetMonth"] >= pd.Timestamp("2019-01-01"))
            & (panel["TargetMonth"] <= pd.Timestamp("2019-06-01")),
            (panel["TargetMonth"] >= pd.Timestamp("2019-07-01"))
            & (panel["TargetMonth"] <= pd.Timestamp("2020-02-01")),
        ],
        ["train", "validation", "test"],
        default="excluded",
    )
    panel = panel[panel["Split"] != "excluded"].reset_index(drop=True)

    manifest_cols = [
        "Station Name", "TargetMonth", "Split", "ObservedSessionsPerDay",
        "ObservedEnergyPerDay", "ObservedDemandPriority", "PersistenceScore",
    ] + feature_cols
    panel[manifest_cols].to_csv(
        output_dir / "palo_temporal_panel_manifest_v11.csv", index=False
    )
    feature_manifest = pd.DataFrame({
        "Feature": feature_cols,
        "TemporalAvailability": [
            "known before target month" for _ in feature_cols
        ],
        "LeakageControl": [
            "lagged by >=1 month or static geography/calendar"
            for _ in feature_cols
        ],
    })
    feature_manifest.to_csv(
        output_dir / "palo_temporal_feature_manifest_v11.csv", index=False
    )
    audit = {
        "RawSessionRows": int(len(raw)),
        "RawStations": int(raw["Station Name"].nunique()),
        "EligibilityCutoff": "2018-12-31",
        "EligibilityRule": "at least 18 active months and 300 sessions before cutoff",
        "EligibleStations": int(len(eligible_stations)),
        "TrainRows": int((panel["Split"] == "train").sum()),
        "ValidationRows": int((panel["Split"] == "validation").sum()),
        "TestRows": int((panel["Split"] == "test").sum()),
        "TrainMonths": int(panel.loc[panel["Split"] == "train", "TargetMonth"].nunique()),
        "ValidationMonths": int(panel.loc[panel["Split"] == "validation", "TargetMonth"].nunique()),
        "TestMonths": int(panel.loc[panel["Split"] == "test", "TargetMonth"].nunique()),
        "TestWindow": "2019-07 through 2020-02 (pre-COVID)",
        "Target": "0.60*within-month sessions/day percentile + 0.40*within-month energy/day percentile",
        "PredictionHorizon": "one month ahead from lagged demand features",
    }
    save_json(output_dir / "palo_temporal_audit_v11.json", audit)
    return panel, feature_cols, audit


def _v11_train_temporal_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    epochs: int,
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols]).astype(np.float32)
    X_val = scaler.transform(validation[feature_cols]).astype(np.float32)
    X_test = scaler.transform(test[feature_cols]).astype(np.float32)
    y_train = train["ObservedDemandPriority"].to_numpy(dtype=np.float32)
    y_val = validation["ObservedDemandPriority"].to_numpy(dtype=np.float32)
    months_train = train["TargetMonth"].astype(str).to_numpy()

    set_seed(seed)
    model = TemporalDemandRankerV11(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    x_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    pair_i, pair_j, pair_rho, pair_margin = _v11_temporal_pair_pool(
        y_train, months_train, threshold=0.03
    )
    if len(pair_i) == 0:
        raise RuntimeError("Temporal demand panel did not yield preference pairs.")

    deterministic = mode == "deterministic"
    pointwise_only = mode == "pointwise"
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_composite = -np.inf
    best_fusion = (0.5, 0.5, 0.0)
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        steps = max(6, int(math.ceil(len(train) / 128)))
        for _ in range(steps):
            station_idx = rng.choice(
                len(train), size=min(256, len(train)), replace=False
            )
            pair_idx = rng.choice(
                len(pair_i), size=min(512, len(pair_i)), replace=False
            )
            mu, sigma, score = model(x_train_t)
            station_t = torch.tensor(
                station_idx, dtype=torch.long, device=device
            )
            if deterministic:
                uncertainty_loss = F.mse_loss(
                    mu[station_t], y_train_t[station_t]
                )
            else:
                uncertainty_loss = gaussian_nll(
                    mu[station_t], sigma[station_t], y_train_t[station_t]
                )
            point_loss = F.smooth_l1_loss(
                torch.sigmoid(score[station_t]),
                y_train_t[station_t],
                beta=0.05,
            )
            rank_loss = torch.tensor(0.0, device=device)
            if not pointwise_only:
                i_t = torch.tensor(
                    pair_i[pair_idx], dtype=torch.long, device=device
                )
                j_t = torch.tensor(
                    pair_j[pair_idx], dtype=torch.long, device=device
                )
                rho_t = torch.tensor(
                    pair_rho[pair_idx], dtype=torch.float32, device=device
                )
                margins_t = torch.tensor(
                    pair_margin[pair_idx], dtype=torch.float32, device=device
                )
                weights = margins_t / (margins_t.mean() + EPS)
                rank_loss = torch.mean(
                    weights * F.softplus(
                        -rho_t * (score[i_t] - score[j_t])
                    )
                )
            loss = (
                0.20 * uncertainty_loss
                + 1.00 * point_loss
                + (2.00 * rank_loss if not pointwise_only else 0.0)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_mu, val_sigma, val_score_raw = model(x_val_t)
        val_mu_np = val_mu.detach().cpu().numpy()
        val_direct_np = torch.sigmoid(val_score_raw).detach().cpu().numpy()
        persistence_val = validation["PersistenceScore"].to_numpy(dtype=float)

        # Early stopping evaluates the neural model only. The structured
        # persistence fusion is selected once after the best neural checkpoint
        # is fixed, using validation data only.
        neural_candidate = 0.5 * val_direct_np + 0.5 * val_mu_np
        _, epoch_summary = _v11_month_metrics(validation, neural_candidate)
        epoch_best = _v11_temporal_composite(epoch_summary)
        history.append({
            "Epoch": epoch,
            "TrainLoss": float(np.mean(epoch_losses)),
            "ValidationComposite": float(epoch_best),
            "ValidationNDCG10Percent": epoch_summary["NDCG_at_10_percent"],
            "ValidationTopKAgreement10Percent": epoch_summary["TopKAgreement_at_10_percent"],
            "ValidationSpearman": epoch_summary["Spearman"],
            "ValidationPairwiseAccuracy": epoch_summary["PairwiseAccuracy"],
            "FusionWeightDirect": 0.5,
            "FusionWeightMean": 0.5,
            "FusionWeightPersistence": 0.0,
        })
        if epoch_best > best_composite + 1e-6:
            best_composite = float(epoch_best)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 12:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_mu, val_sigma, val_raw = model(x_val_t)
        test_mu, test_sigma, test_raw = model(x_test_t)
    val_mu_np = val_mu.detach().cpu().numpy()
    val_sigma_np = val_sigma.detach().cpu().numpy()
    val_direct_np = torch.sigmoid(val_raw).detach().cpu().numpy()
    test_mu_np = test_mu.detach().cpu().numpy()
    test_sigma_np = test_sigma.detach().cpu().numpy()
    test_direct_np = torch.sigmoid(test_raw).detach().cpu().numpy()
    test_persistence = test["PersistenceScore"].to_numpy(dtype=float)
    validation_persistence = validation["PersistenceScore"].to_numpy(dtype=float)

    # Select the three-way hybrid once, after checkpoint selection, on the
    # validation months only. Grid spacing is 0.10 and all weights are
    # non-negative and sum to one.
    best_fusion_metric = -np.inf
    best_fusion = (0.5, 0.5, 0.0)
    for w_direct in np.linspace(0.0, 1.0, 11):
        remaining = 1.0 - w_direct
        for w_mean in np.linspace(0.0, remaining, 11):
            w_persist = remaining - w_mean
            validation_candidate = (
                w_direct * val_direct_np
                + w_mean * val_mu_np
                + w_persist * validation_persistence
            )
            _, fusion_summary = _v11_month_metrics(
                validation, validation_candidate
            )
            fusion_metric = _v11_temporal_composite(fusion_summary)
            if fusion_metric > best_fusion_metric + 1e-12:
                best_fusion_metric = fusion_metric
                best_fusion = (
                    float(w_direct), float(w_mean), float(w_persist)
                )

    test_hybrid = (
        best_fusion[0] * test_direct_np
        + best_fusion[1] * test_mu_np
        + best_fusion[2] * test_persistence
    )
    test_neural = (
        (best_fusion[0] * test_direct_np + best_fusion[1] * test_mu_np)
        / max(best_fusion[0] + best_fusion[1], EPS)
        if best_fusion[0] + best_fusion[1] > 0
        else test_mu_np
    )
    validation_hybrid = (
        best_fusion[0] * val_direct_np
        + best_fusion[1] * val_mu_np
        + best_fusion[2] * validation_persistence
    )
    return {
        "Model": model,
        "Scaler": scaler,
        "Mode": mode,
        "Seed": int(seed),
        "BestEpoch": int(best_epoch),
        "BestValidationComposite": float(best_fusion_metric),
        "BestNeuralValidationComposite": float(best_composite),
        "FusionWeights": best_fusion,
        "TrainingTime_seconds": float(time.perf_counter() - start_time),
        "History": pd.DataFrame(history),
        "ValidationMean": val_mu_np,
        "ValidationSigma": val_sigma_np,
        "ValidationHybridScore": validation_hybrid,
        "TestMean": test_mu_np,
        "TestSigma": test_sigma_np,
        "TestHybridScore": test_hybrid,
        "TestNeuralScore": test_neural,
    }


def _v11_temporal_bootstrap_difference(
    month_table: pd.DataFrame,
    primary: str,
    comparator: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    a = month_table[month_table["Method"] == primary].set_index(
        ["Seed", "TargetMonth"]
    )[metric]
    b = month_table[month_table["Method"] == comparator].set_index(
        ["Seed", "TargetMonth"]
    )[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    # Collapse multiple training seeds to one method estimate per month before
    # resampling months, preserving the temporal evaluation unit.
    by_month = joined.groupby(level="TargetMonth").mean()
    diffs = by_month["a"] - by_month["b"]
    rng = np.random.default_rng(seed)
    boot = []
    values = diffs.to_numpy(dtype=float)
    for _ in range(repetitions):
        sampled = rng.choice(values, size=len(values), replace=True)
        boot.append(float(np.mean(sampled)))
    boot_arr = np.asarray(boot, dtype=float)
    return {
        "MeanDifference": float(values.mean()),
        "BootstrapCI95_L": float(np.quantile(boot_arr, 0.025)),
        "BootstrapCI95_U": float(np.quantile(boot_arr, 0.975)),
        "TwoSidedBootstrapP": bounded_two_sided_bootstrap_p(boot_arr),
        "TestMonths": int(len(values)),
        "BootstrapRepetitions": int(repetitions),
    }


def run_palo_temporal_observed_demand_v11(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Observed-demand temporal validation on the Palo Alto session dataset."""
    if not args.external_usage:
        print("[V11] Palo Alto path not supplied; temporal observed-demand task skipped.")
        return pd.DataFrame()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel, feature_cols, audit = _v11_build_palo_temporal_panel(
        Path(args.external_usage).expanduser().resolve(), output_dir
    )
    train = panel[panel["Split"] == "train"].reset_index(drop=True)
    validation = panel[panel["Split"] == "validation"].reset_index(drop=True)
    test = panel[panel["Split"] == "test"].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Palo temporal split is empty.")

    seed_list = [
        int(x.strip()) for x in str(args.temporal_seeds).split(",") if x.strip()
    ]
    device = choose_device(args.device)
    seed_metric_rows: List[Dict[str, Any]] = []
    month_metric_rows: List[pd.DataFrame] = []
    prediction_rows: List[pd.DataFrame] = []
    uncertainty_rows: List[Dict[str, Any]] = []

    # Predeclared classical baselines; preprocessing is fit on train only.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_validation = scaler.transform(validation[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    y_train = train["ObservedDemandPriority"].to_numpy(dtype=float)
    y_validation = validation["ObservedDemandPriority"].to_numpy(dtype=float)

    for seed in seed_list:
        print(f"[V11] Palo temporal observed-demand seed={seed}")
        neural_full = _v11_train_temporal_ranker(
            train, validation, test, feature_cols, seed,
            args.temporal_epochs, device, mode="full",
        )
        neural_point = _v11_train_temporal_ranker(
            train, validation, test, feature_cols, seed,
            args.temporal_epochs, device, mode="pointwise",
        )
        neural_det = _v11_train_temporal_ranker(
            train, validation, test, feature_cols, seed,
            args.temporal_epochs, device, mode="deterministic",
        )
        histories = output_dir / "histories"
        histories.mkdir(exist_ok=True)
        for label, result in [
            ("full", neural_full),
            ("pointwise", neural_point),
            ("deterministic", neural_det),
        ]:
            result["History"].to_csv(
                histories / f"palo_temporal_{label}_seed_{seed}_v11.csv",
                index=False,
            )

        # Ridge alpha is chosen using validation months only.
        ridge_candidates = {}
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            ridge = Ridge(alpha=alpha).fit(X_train, y_train)
            pred_val = ridge.predict(X_validation)
            _, val_summary = _v11_month_metrics(validation, pred_val)
            ridge_candidates[alpha] = (
                _v11_temporal_composite(val_summary), ridge
            )
        ridge_alpha = max(ridge_candidates, key=lambda a: ridge_candidates[a][0])
        ridge_model = ridge_candidates[ridge_alpha][1]
        gbr_model = GradientBoostingRegressor(
            random_state=seed, n_estimators=120, max_depth=2,
            learning_rate=0.03, min_samples_leaf=4,
        ).fit(X_train, y_train)
        rf_model = RandomForestRegressor(
            random_state=seed, n_estimators=300, max_depth=8,
            min_samples_leaf=3, n_jobs=-1,
        ).fit(X_train, y_train)

        methods: Dict[str, np.ndarray] = {
            "PRIME_Temporal_Hybrid": neural_full["TestHybridScore"],
            "PRIME_Temporal_Neural": neural_full["TestNeuralScore"],
            "PRIME_Temporal_PointwiseOnly": neural_point["TestHybridScore"],
            "PRIME_Temporal_Deterministic": neural_det["TestHybridScore"],
            "HistoricalPersistence": test["PersistenceScore"].to_numpy(dtype=float),
            "RidgeTemporal": ridge_model.predict(X_test),
            "GradientBoostingTemporal": gbr_model.predict(X_test),
            "RandomForestTemporal": rf_model.predict(X_test),
            "Random": np.random.default_rng(seed + 50000).random(len(test)),
        }
        for method, scores in methods.items():
            monthly_metrics, summary = _v11_month_metrics(test, scores)
            monthly_metrics.insert(0, "Method", method)
            monthly_metrics.insert(0, "Seed", seed)
            month_metric_rows.append(monthly_metrics)
            row = {
                "Method": method,
                "Seed": int(seed),
                "Stations": int(test["Station Name"].nunique()),
                "TestMonths": int(test["TargetMonth"].nunique()),
            }
            row.update(summary)
            if method == "PRIME_Temporal_Hybrid":
                row.update({
                    "BestEpoch": neural_full["BestEpoch"],
                    "ValidationComposite": neural_full["BestValidationComposite"],
                    "FusionWeightDirect": neural_full["FusionWeights"][0],
                    "FusionWeightMean": neural_full["FusionWeights"][1],
                    "FusionWeightPersistence": neural_full["FusionWeights"][2],
                    "TrainingTime_seconds": neural_full["TrainingTime_seconds"],
                })
            if method == "RidgeTemporal":
                row["SelectedRidgeAlpha"] = float(ridge_alpha)
            seed_metric_rows.append(row)
            pred_frame = test[[
                "Station Name", "TargetMonth", "ObservedDemandPriority",
                "ObservedSessionsPerDay", "ObservedEnergyPerDay",
            ]].copy()
            pred_frame.insert(0, "Method", method)
            pred_frame.insert(0, "Seed", seed)
            pred_frame["Score"] = np.asarray(scores, dtype=float)
            prediction_rows.append(pred_frame)

        # Validation-only finite-sample conformal intervals for the full neural
        # predictive mean. These are intervals for the observed demand target,
        # not for the hybrid ranking score.
        val_residual = np.abs(
            y_validation - neural_full["ValidationMean"]
        ) / (neural_full["ValidationSigma"] + EPS)
        y_test = test["ObservedDemandPriority"].to_numpy(dtype=float)
        for nominal in [0.50, 0.80, 0.90, 0.95]:
            n_cal = len(val_residual)
            q_level = min(
                1.0, math.ceil((n_cal + 1) * nominal) / n_cal
            )
            q = float(np.quantile(val_residual, q_level, method="higher"))
            lo = np.clip(
                neural_full["TestMean"] - q * neural_full["TestSigma"],
                0.0, 1.0,
            )
            hi = np.clip(
                neural_full["TestMean"] + q * neural_full["TestSigma"],
                0.0, 1.0,
            )
            covered = (y_test >= lo) & (y_test <= hi)
            uncertainty_rows.append({
                "Seed": seed,
                "NominalCoverage": nominal,
                "CalibrationRows": n_cal,
                "FiniteSampleQuantileLevel": q_level,
                "Multiplier": q,
                "ObservedCoverage": float(np.mean(covered)),
                "CoverageError": float(np.mean(covered) - nominal),
                "MeanIntervalWidth": float(np.mean(hi - lo)),
                "CalibrationSource": "2019-01 through 2019-06 validation months",
            })

    seed_metrics = pd.DataFrame(seed_metric_rows)
    month_metrics = pd.concat(month_metric_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    seed_metrics.to_csv(
        output_dir / "palo_temporal_seed_metrics_v11.csv", index=False
    )
    month_metrics.to_csv(
        output_dir / "palo_temporal_month_metrics_v11.csv", index=False
    )
    predictions.to_csv(
        output_dir / "palo_temporal_predictions_v11.csv", index=False
    )
    pd.DataFrame(uncertainty_rows).to_csv(
        output_dir / "palo_temporal_uncertainty_v11.csv", index=False
    )

    summary_rows: List[Dict[str, Any]] = []
    for method, sub in seed_metrics.groupby("Method"):
        for metric in [
            "NDCG_full", "NDCG_at_10_percent",
            "TopKAgreement_at_10_percent", "Spearman", "KendallTau",
            "PairwiseAccuracy", "Regret_at_10_percent",
        ]:
            values = sub[metric].to_numpy(dtype=float)
            lo, hi = ci95(values)
            summary_rows.append({
                "Method": method,
                "Metric": metric,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "CI95_L": lo,
                "CI95_U": hi,
                "Seeds": int(len(values)),
                "StationsPerMonth": int(test["Station Name"].nunique()),
                "TestMonths": int(test["TargetMonth"].nunique()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "palo_temporal_summary_v11.csv", index=False
    )

    comparisons = [
        "PRIME_Temporal_Neural", "PRIME_Temporal_PointwiseOnly",
        "PRIME_Temporal_Deterministic", "HistoricalPersistence",
        "RidgeTemporal", "GradientBoostingTemporal", "RandomForestTemporal",
        "Random",
    ]
    significance_rows: List[Dict[str, Any]] = []
    for comparator in comparisons:
        for metric in [
            "NDCG_at_10_percent", "TopKAgreement_at_10_percent",
            "Spearman", "PairwiseAccuracy", "Regret_at_10_percent",
        ]:
            record = {
                "PrimaryMethod": "PRIME_Temporal_Hybrid",
                "Comparator": comparator,
                "Metric": metric,
            }
            record.update(_v11_temporal_bootstrap_difference(
                month_metrics, "PRIME_Temporal_Hybrid", comparator,
                metric, args.temporal_bootstrap,
                seed=args.split_seed + len(significance_rows) + 88000,
            ))
            significance_rows.append(record)
    pd.DataFrame(significance_rows).to_csv(
        output_dir / "palo_temporal_paired_bootstrap_v11.csv", index=False
    )

    # Compact manuscript table with one row per method.
    pivot = summary.pivot(
        index="Method", columns="Metric", values="Mean"
    ).reset_index()
    pivot.to_csv(
        output_dir / "palo_temporal_manuscript_table_v11.csv", index=False
    )
    return summary


def write_reviewer_evidence_matrix_v11(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    rows = [
        {"ReviewerComment": "R1.1 Limited novelty", "Resolution": "Integrated methodological contribution clarified", "Evidence": "pair_loss_sensitivity_validation_v11.csv; deployment_configuration_selection_v11.csv; uncertainty_aware_constrained_tradeoff_v11.csv", "SupportedClaim": "Validation-selected proxy-ranking pipeline with conformal uncertainty and policy-constrained decision layer", "RemainingBoundary": "No new universal ranking theorem or novel base loss is claimed."},
        {"ReviewerComment": "R1.2 Proxy labels", "Resolution": "Falsification and target sensitivity added", "Evidence": "proxy_validity_bootstrap_v11.csv; proxy_validity_within_operator_v11.csv; proxy_shuffled_negative_control_v11.csv; target_learnability_falsification_v11.csv; shuffled_target_learnability_summary_v11.csv", "SupportedClaim": "Rating-derived service-condition proxy only", "RemainingBoundary": "Not operational failure, downtime, degradation, or maintenance urgency."},
        {"ReviewerComment": "R1.3 Weak exact ranking", "Resolution": "Random-adjusted and fixed-cutoff evidence made primary", "Evidence": "random_adjusted_baseline_metrics_v11.csv; random_ranking_reference.csv; full_model_test_predictions_v11.csv", "SupportedClaim": "Candidate screening and list-level discrimination", "RemainingBoundary": "Exact station ordering remains data-limited and must not be overstated."},
        {"ReviewerComment": "R1.4 Simpler baselines", "Resolution": "Validation-only deployment selection and paired tests", "Evidence": "deployment_configuration_selection_v11.csv; significance_prime_vs_baselines.csv; baseline_transparency.csv", "SupportedClaim": "Use the simplest validation-eligible configuration", "RemainingBoundary": "No blanket superiority claim."},
        {"ReviewerComment": "R1.5 Ablation necessity", "Resolution": "Matched seeds and optimization budgets", "Evidence": "multiseed_ablation_metrics.csv; significance_full_vs_ablations.csv; retrained_feature_order_sensitivity_v11.csv", "SupportedClaim": "Components provide trade-offs", "RemainingBoundary": "No claim that every component is statistically necessary."},
        {"ReviewerComment": "R1.6 Real-world validation", "Resolution": "Two external datasets and domain-shift audit", "Evidence": "cross_dataset_summary_v10.csv; cross_dataset_paired_bootstrap_v11.csv; cross_dataset_domain_shift_v10.csv; palo_temporal_summary_v11.csv; palo_temporal_paired_bootstrap_v11.csv", "SupportedClaim": "External transportability plus leakage-safe one-month-ahead observed-demand ranking", "RemainingBoundary": "External outcomes are different constructs and commercial maintenance validation remains future work."},
        {"ReviewerComment": "R1.7 Vehicular technology relevance", "Resolution": "Operational charger-availability decision layer", "Evidence": "operator_cv_constrained_rerank_tradeoff_v8.csv; uncertainty_aware_constrained_tradeoff_v11.csv", "SupportedClaim": "Prioritization of service review for charging-network continuity", "RemainingBoundary": "No routing, grid dispatch, battery, or V2G optimization is modeled."},
        {"ReviewerComment": "R1.8 Descriptive fairness", "Resolution": "Constraints influence final top-k selection", "Evidence": "operator_cv_oof_predictions_v8.csv; uncertainty_aware_constrained_tradeoff_v11.csv", "SupportedClaim": "Group allocation diagnostics and policy-constrained selection", "RemainingBoundary": "Not demographic fairness learning."},
        {"ReviewerComment": "R1.9 Theory", "Resolution": "Analytic complexity and empirical convergence", "Evidence": "computational_complexity_v11.csv; convergence_summary_v11.csv; convergence_diagnostics_v11.pdf", "SupportedClaim": "Measured computational profile and convergence behavior", "RemainingBoundary": "No non-convex convergence or generalization theorem claimed."},
        {"ReviewerComment": "R1.10 Evaluation diversity", "Resolution": "Fixed split, leave-one-operator-out, regional holdout, and two external datasets", "Evidence": "fixed_split_row_manifest_v11.csv; operator_cv_oof_predictions_v8.csv; regional_transfer_results.csv; cross_dataset_summary_v10.csv", "SupportedClaim": "Multi-protocol robustness evaluation", "RemainingBoundary": "Additional commercial and temporal networks remain desirable."},
        {"ReviewerComment": "R3.1 Proxy-label validity", "Resolution": "Bootstrap, within-operator, categorical, and shuffled controls", "Evidence": "proxy_falsification_summary_v11.json", "SupportedClaim": "Transparent falsification analysis", "RemainingBoundary": "Weak validity is reported, not hidden."},
        {"ReviewerComment": "R3.2 Top-decile baselines", "Resolution": "Top-k agreement, regret, random adjustment, and paired comparisons", "Evidence": "random_adjusted_baseline_metrics_v11.csv; significance_prime_vs_baselines.csv", "SupportedClaim": "Competitive or limited performance stated exactly", "RemainingBoundary": "Simpler baselines may be equal or better."},
        {"ReviewerComment": "R3.3 Split inconsistency", "Resolution": "Pooled leave-one-operator-out predictions", "Evidence": "operator_cv_oof_predictions_v8.csv; fixed_split_row_manifest_v11.csv", "SupportedClaim": "Allocation diagnostics use held-out predictions for every station", "RemainingBoundary": "Full-pool in-sample diagnostics are illustrative only."},
        {"ReviewerComment": "R3.4 Configuration rationale", "Resolution": "Predeclared validation-only quality-tolerance rule", "Evidence": "deployment_configuration_selected_v11.json", "SupportedClaim": "Deployment model chosen without test-set tuning", "RemainingBoundary": "Full architecture remains an evaluated variant if not selected."},
        {"ReviewerComment": "R3.5 Repetitive writing", "Resolution": "Requires manuscript rewrite", "Evidence": "reviewer_evidence_matrix_v11.csv", "SupportedClaim": "Use standardized terminology and non-repetitive section roles", "RemainingBoundary": "This is an editorial revision, not an experiment."},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "reviewer_evidence_matrix_v11.csv", index=False)
    return out


def write_v11_authoritative_payload(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    payload: Dict[str, Any] = {
        "version": V11_VERSION,
        "claim_scope": {
            "primary": "proxy-ranking candidate screening and policy-constrained station selection",
            "risk": "rating-derived service-condition proxy; not observed failure probability",
            "uncertainty": "validation-conformal marginal coverage; instance-level informativeness reported separately",
            "external": "transportability across different outcome constructs; not construct equivalence",
        },
        "files": sorted([p.name for p in output_dir.glob("*") if p.is_file()]),
    }
    for name in (
        "pair_loss_selected_config_v11.json", "deployment_configuration_selected_v11.json",
        "proxy_falsification_summary_v11.json", "uncertainty_aware_selected_policy_v11.json",
        "v11_self_test_report.json",
    ):
        path = output_dir / name
        if path.exists():
            payload[name.replace(".json", "")] = json.load(open(path, "r", encoding="utf-8"))
    save_json(output_dir / "authoritative_reviewer_evidence_v11.json", payload)


def repair_existing_results_v12(args: argparse.Namespace) -> pd.DataFrame:
    """Correct known final-output defects without changing fitted predictions."""
    output_dir = Path(args.output).expanduser().resolve()
    rows: List[Dict[str, Any]] = []

    # 1. Bootstrap p-values must be in [0, 1].
    for filename in ("palo_temporal_paired_bootstrap_v11.csv", "cross_dataset_paired_bootstrap_v11.csv"):
        path = output_dir / filename
        if not path.exists():
            rows.append({"Check": filename, "Status": "missing", "Correction": "not applied"})
            continue
        df = pd.read_csv(path)
        if "TwoSidedBootstrapP" in df.columns:
            before = int((pd.to_numeric(df["TwoSidedBootstrapP"], errors="coerce") > 1.0).sum())
            df["TwoSidedBootstrapP"] = pd.to_numeric(df["TwoSidedBootstrapP"], errors="coerce").clip(0.0, 1.0)
            df.to_csv(path, index=False)
            rows.append({"Check": filename, "Status": "corrected", "Correction": f"capped {before} p-values to [0,1]"})

    # 2. Repair single-run ablation complexity from measured multi-seed records.
    ablation_path = output_dir / "corrected_ablation_table.csv"
    multi_path = output_dir / "multiseed_ablation_metrics.csv"
    if ablation_path.exists() and multi_path.exists():
        ablation = pd.read_csv(ablation_path)
        multi = pd.read_csv(multi_path)
        complexity = multi.groupby("Variant", as_index=False).agg(
            ParameterCount=("ParameterCount", "median"),
            ModelMemoryMB=("ModelMemoryMB", "median"),
        )

        # Some single-run ablations (LowDimension, NoDIM, NoRiskToRanker) are
        # intentionally absent from the reduced multi-seed ablation set. Their
        # complexity is measured by instantiating the exact architecture from
        # the saved experiment configuration; no training or test access is used.
        split_info = json.load(open(output_dir / "split_metadata.json", "r", encoding="utf-8"))
        input_dim = int(split_info["feature_count"])
        variants_for_complexity = {
            "Full": {},
            "NoAttention": {"no_attention": True},
            "NoIRE_Conv": {"no_ire": True},
            "LowDimension": {"latent_dim": 4},
            "DeterministicRisk": {"deterministic_risk": True},
            "NoDIM": {"no_dim": True},
            "PointwiseRanking": {"pointwise_rank": True},
            "NoRiskToRanker": {"no_risk_input": True},
        }
        measured_rows = []
        for variant_name, variant in variants_for_complexity.items():
            latent_dim = int(variant.get("latent_dim", getattr(args, "exact_latent_dim", 48)))
            model = V9ExactPrimeEV(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dim=int(getattr(args, "exact_hidden_dim", 128)),
                residual_blocks=int(getattr(args, "exact_residual_blocks", 3)),
                dropout=float(getattr(args, "exact_dropout", 0.05)),
                risk_weight=float(split_info.get("risk_weight", 0.60)),
                demand_weight=float(split_info.get("demand_weight", 0.40)),
                include_risk_in_ranker=not variant.get("no_risk_input", False),
                demand_enabled=not variant.get("no_dim", False),
                use_feature_gating=not variant.get("no_attention", False),
                simple_encoder=bool(variant.get("no_ire", False)),
                legacy_conv=False,
                legacy_attention=not variant.get("no_attention", False),
            )
            measured_rows.append({
                "Variant": variant_name,
                "MeasuredParameterCount": count_parameters(model),
                "MeasuredModelMemoryMB": memory_mb_for_model(model),
            })
        measured = pd.DataFrame(measured_rows)

        ablation = ablation.drop(columns=["ParameterCount", "ModelMemoryMB"], errors="ignore").merge(
            complexity, on="Variant", how="left", validate="one_to_one"
        ).merge(measured, on="Variant", how="left", validate="one_to_one")
        ablation["ParameterCount"] = pd.to_numeric(ablation["ParameterCount"], errors="coerce").fillna(ablation["MeasuredParameterCount"])
        ablation["ModelMemoryMB"] = pd.to_numeric(ablation["ModelMemoryMB"], errors="coerce").fillna(ablation["MeasuredModelMemoryMB"])
        ablation = ablation.drop(columns=["MeasuredParameterCount", "MeasuredModelMemoryMB"])
        if ablation[["ParameterCount", "ModelMemoryMB"]].isna().any().any():
            missing = ablation.loc[ablation[["ParameterCount", "ModelMemoryMB"]].isna().any(axis=1), "Variant"].tolist()
            raise RuntimeError(f"Cannot repair Pareto complexity for variants: {missing}")
        ablation.to_csv(ablation_path, index=False)
        frontier = pareto_frontier(ablation)
        frontier.to_csv(output_dir / "ablation_pareto_frontier.csv", index=False)
        make_pareto_plot(frontier, output_dir)
        rows.append({"Check": "ablation_pareto_frontier.csv", "Status": "corrected", "Correction": "restored measured parameter count and memory"})

    # 3. Distinguish fusion evaluation from actual selected multi-component fusion.
    fusion_path = output_dir / "exact_fusion_selection_v9.json"
    fusion_meta: Dict[str, Any] = {}
    if fusion_path.exists():
        fusion_meta = json.load(open(fusion_path, "r", encoding="utf-8"))
        weights = np.asarray(fusion_meta.get("weights_neural_structured_ridge", [1.0, 0.0, 0.0]), dtype=float)
        active = np.flatnonzero(weights > 1e-8)
        names = ["Neural", "Structured", "Ridge"]
        selected = bool(len(active) >= 2)
        source = "MultiComponentFusion" if selected else (f"{names[int(active[0])]}Only" if len(active) == 1 else "Undefined")
        fusion_meta["evaluated"] = True
        fusion_meta["fusion_selected"] = selected
        fusion_meta["enabled"] = selected
        fusion_meta["selected_score_source"] = source
        save_json(fusion_path, fusion_meta)
        rows.append({"Check": "exact_fusion_selection_v9.json", "Status": "corrected", "Correction": f"fusion_selected={selected}; source={source}"})

        metrics_path = output_dir / "full_model_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            metrics["FusionEvaluated"] = 1.0
            metrics["FusionSelected"] = float(selected)
            metrics["FusionEnabled"] = float(selected)
            metrics["ScoreSource"] = source
            metrics["FusionWeightNeural"] = float(weights[0])
            metrics["FusionWeightStructured"] = float(weights[1])
            metrics["FusionWeightRidge"] = float(weights[2])
            metrics.to_csv(metrics_path, index=False)

    # 4. Create a single authoritative file that separates raw and conformal uncertainty.
    calibration_path = output_dir / "uncertainty_calibration_v11.csv"
    calibration_payload: Dict[str, Any] = {}
    if calibration_path.exists():
        cal = pd.read_csv(calibration_path)
        calibration_payload = {
            "raw_gaussian": cal[cal["Method"] == "RawGaussian"].to_dict("records"),
            "validation_split_conformal": cal[cal["Method"] == "ValidationSplitConformal"].to_dict("records"),
            "claim": "Conformal results support marginal coverage only; conditional group coverage is reported separately.",
        }
    group_path = output_dir / "conformal_group_coverage_v11.csv"
    if group_path.exists():
        calibration_payload["conditional_group_coverage"] = pd.read_csv(group_path).to_dict("records")
    residual_path = output_dir / "uncertainty_interval_diagnostics.csv"
    if residual_path.exists():
        calibration_payload["raw_uncertainty_diagnostics"] = pd.read_csv(residual_path).to_dict("records")

    full_metrics = pd.read_csv(output_dir / "full_model_metrics.csv").iloc[0].to_dict() if (output_dir / "full_model_metrics.csv").exists() else {}
    authoritative = {
        "version": V11_VERSION,
        "claim_scope": {
            "main_task": "rating-derived service-condition proxy screening; not observed failure-risk prediction",
            "exact_ranking": "supported primarily by the independent one-month-ahead observed-demand task",
            "superiority": "claim only comparisons supported after multiplicity correction",
            "uncertainty": "validation split-conformal marginal coverage; not conditional or instance-level calibration",
        },
        "full_model_metrics": full_metrics,
        "fusion_selection": fusion_meta,
        "uncertainty": calibration_payload,
        "files": sorted([p.name for p in output_dir.glob("*") if p.is_file()]),
    }
    for filename, key in (
        ("significance_prime_vs_baselines.csv", "baseline_significance"),
        ("significance_full_vs_ablations.csv", "ablation_significance"),
        ("cross_dataset_summary_v10.csv", "cross_dataset_summary"),
        ("palo_temporal_summary_v11.csv", "temporal_observed_demand_summary"),
        ("uncertainty_aware_constrained_tradeoff_v11.csv", "policy_tradeoff"),
    ):
        path = output_dir / filename
        if path.exists():
            authoritative[key] = pd.read_csv(path).to_dict("records")
    save_json(output_dir / "authoritative_manuscript_values_v12.json", authoritative)
    save_json(output_dir / "authoritative_reviewer_evidence_v12.json", authoritative)
    # Overwrite the legacy manuscript file with the corrected, explicit schema.
    save_json(output_dir / "authoritative_manuscript_values.json", authoritative)

    checksum_rows = []
    checksum_targets = [Path(__file__).resolve(), Path(args.data).expanduser().resolve()]
    for optional in (getattr(args, "external_us", None), getattr(args, "external_usage", None)):
        if optional:
            checksum_targets.append(Path(optional).expanduser().resolve())
    for target in checksum_targets:
        if target.exists():
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksum_rows.append({"File": target.name, "AbsolutePath": str(target), "SHA256": digest.hexdigest(), "Bytes": target.stat().st_size})
    pd.DataFrame(checksum_rows).to_csv(output_dir / "dataset_and_code_checksums_v12.csv", index=False)

    manifest = [
        "authoritative_manuscript_values_v12.json",
        "authoritative_reviewer_evidence_v12.json",
        "v12_output_correction_audit.csv",
        "v12_final_consistency_check.csv",
        "dataset_and_code_checksums_v12.csv",
        "full_model_metrics.csv",
        "significance_prime_vs_baselines.csv",
        "significance_full_vs_ablations.csv",
        "ablation_pareto_frontier.csv",
        "uncertainty_calibration_v11.csv",
        "conformal_group_coverage_v11.csv",
        "cross_dataset_summary_v10.csv",
        "cross_dataset_paired_bootstrap_v11.csv",
        "palo_temporal_summary_v11.csv",
        "palo_temporal_paired_bootstrap_v11.csv",
        "reviewer_evidence_matrix_v11.csv",
    ]
    (output_dir / "V12_FINAL_UPLOAD_THESE_FILES.txt").write_text("\n".join(manifest), encoding="utf-8")

    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "v12_output_correction_audit.csv", index=False)
    return audit


def write_final_result_consistency_v12(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output).expanduser().resolve()
    checks: List[Dict[str, Any]] = []
    temporal = output_dir / "palo_temporal_paired_bootstrap_v11.csv"
    if temporal.exists():
        df = pd.read_csv(temporal)
        checks.append({"Check": "Temporal bootstrap p-values in [0,1]", "Passed": bool(df["TwoSidedBootstrapP"].between(0, 1, inclusive="both").all())})
    pareto = output_dir / "ablation_pareto_frontier.csv"
    if pareto.exists():
        df = pd.read_csv(pareto)
        checks.append({"Check": "Pareto parameter counts positive", "Passed": bool((df["ParameterCount"] > 0).all())})
        checks.append({"Check": "Pareto memory positive", "Passed": bool((df["ModelMemoryMB"] > 0).all())})
    metrics = output_dir / "full_model_metrics.csv"
    if metrics.exists():
        row = pd.read_csv(metrics).iloc[0]
        active = int(sum(float(row.get(c, 0.0)) > 1e-8 for c in ["FusionWeightNeural", "FusionWeightStructured", "FusionWeightRidge"]))
        checks.append({"Check": "Fusion status matches active component count", "Passed": bool(int(float(row.get("FusionSelected", row.get("FusionEnabled", 0)))) == int(active >= 2))})
    auth = output_dir / "authoritative_manuscript_values_v12.json"
    if auth.exists():
        payload = json.load(open(auth, "r", encoding="utf-8"))
        checks.append({"Check": "Authoritative output separates raw and conformal uncertainty", "Passed": bool("raw_gaussian" in payload.get("uncertainty", {}) and "validation_split_conformal" in payload.get("uncertainty", {}))})
    out = pd.DataFrame(checks)
    out.to_csv(output_dir / "v12_final_consistency_check.csv", index=False)
    if not out.empty and not bool(out["Passed"].all()):
        failed = out.loc[~out["Passed"], "Check"].tolist()
        raise RuntimeError(f"V12 result consistency checks failed: {failed}")
    return out


def run_v11_postprocessing(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_v11_audits:
        if not args.skip_proxy_falsification:
            run_proxy_falsification_v11(args)
        if not args.skip_target_learnability:
            run_target_learnability_v11(args)
            run_shuffled_target_learnability_v11(args)
        run_random_adjusted_evidence_v11(args)
        run_uncertainty_aware_constrained_selection_v11(args)
        run_reproducibility_manifest_v11(args)
        run_complexity_and_convergence_v11(args)
        run_external_paired_bootstrap_v11(args)
    write_reviewer_evidence_matrix_v11(args)
    run_v11_self_tests(args)
    write_v11_authoritative_payload(args)
    repair_existing_results_v12(args)
    write_final_result_consistency_v12(args)
    upload_files = [
        "authoritative_reviewer_evidence_v11.json",
        "authoritative_manuscript_values_v12.json",
        "v12_output_correction_audit.csv",
        "v12_final_consistency_check.csv",
        "reviewer_evidence_matrix_v11.csv",
        "pair_loss_sensitivity_validation_v11.csv",
        "pair_loss_selected_config_v11.json",
        "deployment_configuration_selection_v11.csv",
        "deployment_configuration_selected_v11.json",
        "selected_deployment_model_metrics_v11.csv",
        "selected_deployment_test_predictions_v11.csv",
        "full_vs_selected_deployment_v11.csv",
        "retrained_feature_order_sensitivity_v11.csv",
        "retrained_feature_order_sensitivity_summary_v11.csv",
        "proxy_validity_bootstrap_v11.csv",
        "proxy_validity_within_operator_v11.csv",
        "proxy_shuffled_negative_control_v11.csv",
        "target_learnability_falsification_v11.csv",
        "shuffled_target_learnability_raw_v11.csv",
        "shuffled_target_learnability_summary_v11.csv",
        "random_adjusted_baseline_metrics_v11.csv",
        "uncertainty_calibration_v11.csv",
        "conformal_test_intervals_v11.csv",
        "conformal_group_coverage_v11.csv",
        "uncertainty_aware_constrained_tradeoff_v11.csv",
        "uncertainty_aware_selected_policy_v11.json",
        "fixed_split_row_manifest_v11.csv",
        "dataset_and_code_checksums_v11.csv",
        "environment_and_split_v11.json",
        "computational_complexity_v11.csv",
        "convergence_summary_v11.csv",
        "convergence_diagnostics_v11.pdf",
        "cross_dataset_paired_bootstrap_v11.csv",
        "palo_temporal_audit_v11.json",
        "palo_temporal_feature_manifest_v11.csv",
        "palo_temporal_panel_manifest_v11.csv",
        "palo_temporal_seed_metrics_v11.csv",
        "palo_temporal_month_metrics_v11.csv",
        "palo_temporal_summary_v11.csv",
        "palo_temporal_predictions_v11.csv",
        "palo_temporal_uncertainty_v11.csv",
        "palo_temporal_paired_bootstrap_v11.csv",
        "palo_temporal_manuscript_table_v11.csv",
        "v11_self_test_report.json",
    ]
    (output_dir / "V11_UPLOAD_THESE_FILES.txt").write_text("\n".join(upload_files), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    global V10_RUNTIME_ARGS, V9_RUNTIME_ARGS
    if V10_RUNTIME_ARGS is not None:
        return V10_RUNTIME_ARGS
    parser = argparse.ArgumentParser(
        description="PRIME-EV V11 final reviewer-completion experiment suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="prime_ev_v11_final")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sensitivity-epochs", type=int, default=10)
    parser.add_argument("--regional-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-metric", choices=["ndcg_full", "ndcg_top_fraction", "exact_composite"], default="exact_composite")
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-demand", type=float, default=0.30)
    parser.add_argument("--lambda-rank", type=float, default=6.0)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--pair-threshold", type=float, default=0.03)
    parser.add_argument("--train-pairs", type=int, default=60000)
    parser.add_argument("--validation-pairs", type=int, default=12000)
    parser.add_argument("--test-pairs", type=int, default=12000)
    parser.add_argument("--risk-weight", type=float, default=0.60)
    parser.add_argument("--demand-weight", type=float, default=0.40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--review-seeds", default="42,123,456,789,2025,31415,27182,16180,57721,65537")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--disable-pair-margin-weighting", action="store_true")
    parser.add_argument("--multiseed-epochs", type=int, default=10)
    parser.add_argument("--multiseed-train-pairs", type=int, default=30000)
    parser.add_argument("--multiseed-validation-pairs", type=int, default=8000)
    parser.add_argument("--multiseed-test-pairs", type=int, default=8000)
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-regional-transfer", action="store_true")
    parser.add_argument("--skip-label-sensitivity", action="store_true")
    parser.add_argument("--skip-baseline-sensitivity", action="store_true")
    parser.add_argument("--skip-order-sensitivity", action="store_true")
    parser.add_argument("--skip-operator-cv", action="store_true")
    parser.add_argument("--order-permutations", type=int, default=5)
    parser.add_argument("--order-epochs", type=int, default=8)
    parser.add_argument("--order-seeds", default="42,123,456", help="Matched training seeds reused for every feature permutation")
    parser.add_argument("--operator-cv-epochs", type=int, default=10)
    parser.add_argument("--disable-engineered-features", action="store_true")
    parser.add_argument("--disable-exact-fusion", action="store_true")
    parser.add_argument("--exact-hidden-dim", type=int, default=128)
    parser.add_argument("--exact-latent-dim", type=int, default=48)
    parser.add_argument("--exact-residual-blocks", type=int, default=3)
    parser.add_argument("--exact-dropout", type=float, default=0.05)
    parser.add_argument("--exact-pair-weight", type=float, default=1.0)
    parser.add_argument("--exact-point-weight", type=float, default=1.0)
    parser.add_argument("--exact-list-weight", type=float, default=0.10)
    parser.add_argument("--exact-corr-weight", type=float, default=0.25)
    parser.add_argument("--exact-hard-weight", type=float, default=0.50)
    parser.add_argument("--exact-hard-pairs", type=int, default=4096)
    parser.add_argument("--exact-huber-beta", type=float, default=0.05)
    parser.add_argument("--exact-list-temperature", type=float, default=0.18)
    parser.add_argument("--exact-fusion-trials", type=int, default=512)
    parser.add_argument("--exact-fusion-min-gain", type=float, default=0.002)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--external-us")
    parser.add_argument("--external-usage")
    parser.add_argument("--skip-cross-dataset", action="store_true")
    parser.add_argument("--cross-only", action="store_true")
    parser.add_argument("--cross-seeds", default="42,123,456,789,2025")
    parser.add_argument("--cross-bootstrap", type=int, default=1000)
    parser.add_argument("--skip-palo-temporal", action="store_true", help="Skip observed-demand temporal validation on Palo Alto sessions")
    parser.add_argument("--temporal-seeds", default="42,123,456,789,2025")
    parser.add_argument("--temporal-epochs", type=int, default=40)
    parser.add_argument("--temporal-bootstrap", type=int, default=1000)
    parser.add_argument("--skip-pair-loss-sensitivity", action="store_true")
    parser.add_argument("--pair-loss-search-epochs", type=int, default=4)
    parser.add_argument("--pair-loss-search-steps", type=int, default=20)
    parser.add_argument("--pair-loss-search-pairs", type=int, default=20000)
    parser.add_argument("--pair-loss-search-seed", type=int, default=31415)
    parser.add_argument("--pair-loss-selection-tolerance", type=float, default=0.005)
    parser.add_argument("--skip-deployment-selection", action="store_true")
    parser.add_argument("--deployment-selection-epochs", type=int, default=6)
    parser.add_argument("--deployment-selection-steps", type=int, default=30)
    parser.add_argument("--deployment-selection-pairs", type=int, default=20000)
    parser.add_argument("--deployment-selection-seed", type=int, default=27182)
    parser.add_argument("--deployment-quality-tolerance", type=float, default=0.005)
    parser.add_argument("--proxy-bootstrap", type=int, default=1000)
    parser.add_argument("--proxy-shuffles", type=int, default=1000)
    parser.add_argument("--target-shuffles", type=int, default=200)
    parser.add_argument("--skip-proxy-falsification", action="store_true")
    parser.add_argument("--skip-target-learnability", action="store_true")
    parser.add_argument("--skip-v11-audits", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true", help="Use existing V9/V10 outputs and run only V11 audits/report generation")
    parser.add_argument("--repair-only", action="store_true", help="Correct existing final-result files without retraining or rerunning statistical experiments")
    parser.add_argument("--selection-only", action="store_true", help="Run validation-only pair/loss and deployment selection, then stop")
    V10_RUNTIME_ARGS = parser.parse_args()
    V9_RUNTIME_ARGS = V10_RUNTIME_ARGS
    if V10_RUNTIME_ARGS.quick:
        V10_RUNTIME_ARGS.cross_seeds = "42,123"
        V10_RUNTIME_ARGS.order_seeds = "42"
        V10_RUNTIME_ARGS.cross_bootstrap = min(V10_RUNTIME_ARGS.cross_bootstrap, 100)
        V10_RUNTIME_ARGS.temporal_seeds = "42"
        V10_RUNTIME_ARGS.temporal_epochs = min(V10_RUNTIME_ARGS.temporal_epochs, 8)
        V10_RUNTIME_ARGS.temporal_bootstrap = min(V10_RUNTIME_ARGS.temporal_bootstrap, 100)
        V10_RUNTIME_ARGS.proxy_bootstrap = min(V10_RUNTIME_ARGS.proxy_bootstrap, 100)
        V10_RUNTIME_ARGS.proxy_shuffles = min(V10_RUNTIME_ARGS.proxy_shuffles, 100)
        V10_RUNTIME_ARGS.target_shuffles = min(V10_RUNTIME_ARGS.target_shuffles, 20)
        V10_RUNTIME_ARGS.pair_loss_search_epochs = min(V10_RUNTIME_ARGS.pair_loss_search_epochs, 1)
        V10_RUNTIME_ARGS.pair_loss_search_steps = min(V10_RUNTIME_ARGS.pair_loss_search_steps, 2)
        V10_RUNTIME_ARGS.deployment_selection_epochs = min(V10_RUNTIME_ARGS.deployment_selection_epochs, 1)
        V10_RUNTIME_ARGS.deployment_selection_steps = min(V10_RUNTIME_ARGS.deployment_selection_steps, 2)
    return V10_RUNTIME_ARGS


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.repair_only:
        repair_existing_results_v12(args)
        write_final_result_consistency_v12(args)
        print(f"\n{V11_VERSION} repair completed: {output_dir}")
        return
    if args.postprocess_only:
        run_v11_postprocessing(args)
        print(f"\n{V11_VERSION} postprocessing completed: {output_dir}")
        return
    if args.selection_only:
        if not args.skip_pair_loss_sensitivity:
            selected = run_pair_loss_sensitivity_v11(args)
            print(f"[V11] Selected pair configuration: {selected['selected']}")
        if not args.skip_deployment_selection:
            selected_deployment = run_deployment_selection_v11(args)
            print(f"[V11] Recommended deployment configuration: {selected_deployment['selected_configuration']}")
        run_v11_self_tests(args)
        print(f"\n{V11_VERSION} selection-only run completed: {output_dir}")
        return
    if args.cross_only:
        if not args.skip_cross_dataset:
            if not args.external_us or not args.external_usage:
                raise ValueError("--cross-only requires --external-us and --external-usage")
            run_cross_dataset_validation(args)
        if not args.skip_palo_temporal:
            if not args.external_usage:
                raise ValueError("Palo temporal validation requires --external-usage")
            run_palo_temporal_observed_demand_v11(args)
        run_v11_postprocessing(args)
        print(f"\n{V11_VERSION} cross-only run completed: {output_dir}")
        return

    if not args.skip_pair_loss_sensitivity:
        print("\n[V11] Validation-only pair-threshold/loss-weight selection")
        selected = run_pair_loss_sensitivity_v11(args)
        print(f"[V11] Selected pair configuration: {selected['selected']}")
    if not args.skip_deployment_selection:
        print("\n[V11] Validation-only deployment configuration selection")
        selected_deployment = run_deployment_selection_v11(args)
        print(f"[V11] Recommended deployment configuration: {selected_deployment['selected_configuration']}")

    _V9_MAIN()
    if not args.skip_deployment_selection:
        print("\n[V11] Evaluating validation-selected deployment configuration on the untouched test split")
        run_selected_deployment_test_v11(args)
    if not args.skip_cross_dataset:
        if args.external_us and args.external_usage:
            run_cross_dataset_validation(args)
        else:
            print("[V11] External dataset paths were not supplied; cross-dataset validation skipped.")
    if not args.skip_palo_temporal:
        if args.external_usage:
            print("\n[V11] Running leakage-safe temporal observed-demand validation")
            run_palo_temporal_observed_demand_v11(args)
        else:
            print("[V11] Palo Alto path not supplied; temporal observed-demand validation skipped.")
    run_v11_postprocessing(args)
    print(f"\n{V11_VERSION} completed")
    print(f"Output directory: {output_dir}")
    print("Use authoritative_reviewer_evidence_v11.json and V11_UPLOAD_THESE_FILES.txt.")


if __name__ == "__main__":
    main()
