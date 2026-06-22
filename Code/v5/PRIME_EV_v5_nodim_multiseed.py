#!/usr/bin/env python3
"""
PRIME-EV reviewer-ready end-to-end experiment script
=====================================================

This single file reruns PRIME-EV with:
  1. explicit target and preference-label construction;
  2. operator-disjoint train/validation/test splits;
  3. supervised preference pairs with reported pair counts;
  4. leakage checks and label-weight sensitivity;
  5. trained PRIME-EV regional transfer tests;
  6. operator-level, geographic, and accessibility equity metrics;
  7. corrected SSI, DeltaPred, normalized latency, and composite score;
  8. transparent MCDM, multi-objective, machine-learning, Pareto, and random baselines;
  9. MCDM and multi-objective weight sensitivity;

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
    def __init__(self, latent_dim: int, include_risk: bool = True):
        super().__init__()
        self.include_risk = include_risk
        input_dim = latent_dim + 2 + (1 if include_risk else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor, mu: torch.Tensor, cost: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        pieces = [z]
        if self.include_risk:
            pieces.append(mu.unsqueeze(1))
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
        self.ranker = PriorityUtilityNetwork(latent_dim, include_risk=include_risk_in_ranker)

    def forward(self, x: torch.Tensor, cost: torch.Tensor, distance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        mu, sigma = self.risk(z)
        usage_hat = self.demand(z, mu, distance)
        score = self.ranker(z, mu, cost, distance)
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
    variant = dict(variant or {})
    set_seed(SEED + int(variant.get("seed_offset", 0)))
    latent_dim = int(variant.get("latent_dim", config.latent_dim))
    model = PrimeEV(
        input_dim=bundle.X_train.shape[1],
        latent_dim=latent_dim,
        use_attention=not variant.get("no_attention", False),
        use_conv=not variant.get("no_ire", False),
        include_risk_in_ranker=not variant.get("no_risk_input", False),
    ).to(device)

    tensors = tensors_for_bundle(bundle, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epochs = int(epochs_override or config.epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_ndcg = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        rng_epoch = np.random.default_rng(SEED + epoch + int(variant.get("seed_offset", 0)) * 1000)
        n_train = len(bundle.X_train)
        station_count = min(config.station_batch_size, n_train)
        station_idx_np = rng_epoch.choice(n_train, size=station_count, replace=False)

        pair_total = len(bundle.train_pairs[0])
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

        mu_all, sigma_all, usage_all, score_all = model(
            tensors["X_train"][union_idx],
            tensors["cost_train"][union_idx],
            tensors["dist_train"][union_idx],
        )
        mu = mu_all[station_local]
        sigma = sigma_all[station_local]
        usage_hat = usage_all[station_local]
        y_target = tensors["y_train"][torch.tensor(station_idx_np, dtype=torch.long, device=device)]
        u_target = tensors["u_train"][torch.tensor(station_idx_np, dtype=torch.long, device=device)]
        g_target = tensors["g_train"][torch.tensor(station_idx_np, dtype=torch.long, device=device)]

        if variant.get("deterministic_risk", False):
            risk_loss = F.mse_loss(mu, y_target)
        else:
            risk_loss = gaussian_nll(mu, sigma, y_target)

        demand_loss = torch.tensor(0.0, device=device)
        if not variant.get("no_dim", False):
            demand_loss = F.mse_loss(usage_hat, u_target)

        if variant.get("pointwise_rank", False):
            rank_loss = F.mse_loss(torch.sigmoid(score_all[station_local]), g_target)
        else:
            rank_loss = torch.mean(
                F.softplus(-pair_rho * (score_all[pair_i_local] - score_all[pair_j_local]))
            )

        loss = (
            config.lambda_risk * risk_loss
            + config.lambda_demand * demand_loss
            + config.lambda_rank * rank_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mu, val_sigma, val_usage_hat, val_score = model(
                tensors["X_val"], tensors["cost_val"], tensors["dist_val"]
            )
            val_score_np = val_score.detach().cpu().numpy()
            if config.validation_metric == "ndcg_top_fraction":
                val_k = max(1, int(math.ceil(len(bundle.g_val) * config.top_fraction)))
            else:
                val_k = len(bundle.g_val)
            val_ndcg = ndcg_at_k(bundle.g_val, val_score_np, val_k)
            if variant.get("deterministic_risk", False):
                val_risk = F.mse_loss(val_mu, tensors["y_val"])
            else:
                val_risk = gaussian_nll(val_mu, val_sigma, tensors["y_val"])
            val_demand = torch.tensor(0.0, device=device)
            if not variant.get("no_dim", False):
                val_demand = F.mse_loss(val_usage_hat, tensors["u_val"])
            if variant.get("pointwise_rank", False):
                val_rank = F.mse_loss(torch.sigmoid(val_score), tensors["g_val"])
            else:
                val_rank = pairwise_logistic_loss(val_score, tensors["val_pairs"])
            val_total = (
                config.lambda_risk * val_risk
                + config.lambda_demand * val_demand
                + config.lambda_rank * val_rank
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_total": float(loss.item()),
                "train_risk": float(risk_loss.item()),
                "train_demand": float(demand_loss.item()),
                "train_rank": float(rank_loss.item()),
                "validation_total": float(val_total.item()),
                "validation_ndcg": float(val_ndcg),
            }
        )

        if val_ndcg > best_ndcg + 1e-6:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{name}] epoch {epoch:03d}/{epochs} "
                f"train={loss.item():.5f} val={val_total.item():.5f} "
                f"val_NDCG={val_ndcg:.5f}"
            )

        if epochs_without_improvement >= config.patience:
            print(f"[{name}] early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    mu, sigma, usage_hat, score, latency = infer_model(
        model, bundle.X_test, bundle.cost_test, bundle.dist_test, device,
        batch_size=config.eval_batch_size
    )
    ranking = ranking_metrics(bundle.g_test, score, bundle.test_pairs, config.top_fraction)
    risk = risk_metrics(bundle.y_test, mu, sigma)
    losses = evaluate_losses(model, tensors, "test", variant, config)
    metrics = dict(ranking)
    metrics.update(risk)
    metrics["Demand_MSE"] = float(mean_squared_error(bundle.u_test, usage_hat))
    metrics["SSI"] = system_stress_index(score)

    return ModelResult(
        name=name,
        model=model,
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
    ranks = criteria.rank(ascending=False, method="average")
    frontier = 1.0 - (ranks.min(axis=1) - 1.0) / max(1, len(ranks))
    balance = 1.0 - (ranks.mean(axis=1) - 1.0) / max(1, len(ranks))
    return (0.45 * frontier + 0.55 * balance).to_numpy(dtype=float)


def tune_regressor(
    model_class: Any,
    grid: Mapping[str, Sequence[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fixed: Optional[Mapping[str, Any]] = None,
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
        metric = ndcg_at_k(y_val, pred, len(y_val))
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
            "Family": "Pareto surrogate",
            "Inputs": ", ".join(criteria_test.columns),
            "Objective": "Balanced criterion-rank aggregation",
            "Hyperparameters": "0.45 frontier rank + 0.55 mean rank",
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
        )
        scores[method_name] = model.predict(bundle.X_test)
        transparency_rows.append(
            {
                "Method": method_name,
                "Family": "Machine-learning ranker",
                "Inputs": "Same leakage-controlled inference feature matrix as PRIME-EV",
                "Objective": "Regression to formula-derived intervention-priority proxy",
                "Hyperparameters": json.dumps(params),
                "Tuning": f"Grid search selected by validation NDCG={val_score:.5f}",
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
        "NoAttention": {"no_attention": True, "seed_offset": 1},
        "NoIRE_Conv": {"no_ire": True, "seed_offset": 2},
        "LowDimension": {"latent_dim": 4, "seed_offset": 3},
        "DeterministicRisk": {"deterministic_risk": True, "seed_offset": 4},
        "NoDIM": {"no_dim": True, "seed_offset": 5},
        "PointwiseRanking": {"pointwise_rank": True, "seed_offset": 6},
        "NoRiskToRanker": {"no_risk_input": True, "seed_offset": 7},
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
            variant={"seed_offset": 20 + idx},
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
            variant={"seed_offset": 40 + fold},
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
        out[f"MAP@{k}"] = average_precision_at_k(labels, scores, kk)
        out[f"Recall@{k}"] = precision_at_k(labels, scores, kk)
        out[f"HitRate@{k}"] = float(precision_at_k(labels, scores, kk) > 0.0)
        out[f"TopKOverlap@{k}"] = precision_at_k(labels, scores, kk)
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
    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=250, random_state=SEED).fit(bundle.X_train, y_train)
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
    rng = np.random.default_rng(SEED + 900)
    rows = []
    for r in range(repeats):
        X = bundle.X_test[:, rng.permutation(bundle.X_test.shape[1])]
        _, _, _, scores, _ = infer_model(result.model, X, bundle.cost_test, bundle.dist_test, device, repeats=1, batch_size=config.eval_batch_size)
        rows.append({"Permutation": r, **fixed_cutoff_metrics(bundle.g_test, scores, bundle.test_pairs)})
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "conv_feature_permutation_sensitivity.csv", index=False)
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
        "NoIRE_Conv": {"no_ire": True, "seed_offset": 2},
        "DeterministicRisk": {"deterministic_risk": True, "seed_offset": 4},
        "NoDIM": {"no_dim": True, "seed_offset": 5},
        "PointwiseRanking": {"pointwise_rank": True, "seed_offset": 6},
        # Kept for supplementary diagnostics only. Do not place every row in the main paper table.
        "NoAttention": {"no_attention": True, "seed_offset": 1},
    }
    for seed in seed_list:
        print(f"\n[V5] Multi-seed evaluation seed={seed}")
        SEED = seed
        set_seed(seed)
        train_idx, val_idx, test_idx, split_meta = operator_disjoint_split(df, seed)
        cfg = copy.deepcopy(config)
        cfg.epochs = min(cfg.epochs, args.multiseed_epochs)
        cfg.patience = min(cfg.patience, max(3, args.multiseed_epochs // 2))
        cfg.batch_pairs_train = min(cfg.batch_pairs_train, args.multiseed_train_pairs)
        cfg.batch_pairs_val = min(cfg.batch_pairs_val, args.multiseed_validation_pairs)
        cfg.batch_pairs_test = min(cfg.batch_pairs_test, args.multiseed_test_pairs)
        b = build_bundle(df, train_idx, val_idx, test_idx, cfg, split_meta)
        full = train_prime_ev(b, cfg, device, name=f"Full_seed_{seed}", epochs_override=cfg.epochs)
        save_model_checkpoint(output_dir / "models" / f"PRIME_EV_v5_seed_{seed}.pt", full, b, cfg)
        base, trans, scores = evaluate_baselines_v5(b, cfg, full)
        fixed = baseline_fixed_table(b, scores)
        for _, r in fixed.iterrows():
            rec = dict(r); rec["Seed"] = seed; baseline_rows.append(rec)
        for vname, var in variants.items():
            res = full if vname == "Full" else train_prime_ev(b, cfg, device, name=f"{vname}_seed_{seed}", variant=var, epochs_override=cfg.epochs)
            rec = {"Seed": seed, "Variant": vname, "ParameterCount": count_parameters(res.model), "ModelMemoryMB": memory_mb_for_model(res.model), "TrainingTime_seconds": res.train_seconds, "Latency_ms_per_station": res.latency_ms_per_station}
            rec.update(fixed_cutoff_metrics(b.g_test, res.test_scores, b.test_pairs))
            ablation_rows.append(rec)
    SEED = old_seed
    baselines = pd.DataFrame(baseline_rows)
    ablations = pd.DataFrame(ablation_rows)
    baselines.to_csv(output_dir / "multiseed_baseline_fixed_metrics.csv", index=False)
    ablations.to_csv(output_dir / "multiseed_ablation_metrics.csv", index=False)
    return baselines, ablations


def pareto_frontier(ablation: pd.DataFrame) -> pd.DataFrame:
    if ablation.empty:
        return ablation
    work = ablation.copy()
    work["ParameterCount"] = work.get("ParameterCount", pd.Series(np.zeros(len(work)))).fillna(0)
    work["ParetoEfficient"] = True
    for i, row in work.iterrows():
        dominates = (
            (work["NDCG_full"] >= row["NDCG_full"]) &
            (work["GeographicEquityScore"] >= row.get("GeographicEquityScore", 0)) &
            (work["Latency_ms_per_station"] <= row["Latency_ms_per_station"]) &
            (work["ParameterCount"] <= row["ParameterCount"])
        )
        strict = (
            (work["NDCG_full"] > row["NDCG_full"]) |
            (work["Latency_ms_per_station"] < row["Latency_ms_per_station"]) |
            (work["ParameterCount"] < row["ParameterCount"])
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
    av = ablation_summary[(ablation_summary.get("Variant", pd.Series(dtype=str)).isin(["Full", "NoIRE_Conv", "DeterministicRisk", "NoDIM", "PointwiseRanking"])) & (ablation_summary.get("Metric", pd.Series(dtype=str)).isin(["NDCG_full", "PairwiseAccuracy", "TrainingTime_seconds", "Latency_ms_per_station", "ParameterCount", "ModelMemoryMB"]))]
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
    (output_dir / "v5_compact_reviewer_table.tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRIME-EV V5 reviewer-ready experiment script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="prime_ev_v5_results")
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
    parser.add_argument("--multiseed-epochs", type=int, default=8)
    parser.add_argument("--multiseed-train-pairs", type=int, default=20000)
    parser.add_argument("--multiseed-validation-pairs", type=int, default=5000)
    parser.add_argument("--multiseed-test-pairs", type=int, default=5000)
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-regional-transfer", action="store_true")
    parser.add_argument("--skip-label-sensitivity", action="store_true")
    parser.add_argument("--skip-baseline-sensitivity", action="store_true")
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
        torch_threads=max(1, args.torch_threads),
    )
    if args.quick:
        config.epochs = 3; config.sensitivity_epochs = 2; config.regional_epochs = 2
        config.batch_pairs_train = min(config.batch_pairs_train, 3000)
        config.batch_pairs_val = min(config.batch_pairs_val, 1000)
        config.batch_pairs_test = min(config.batch_pairs_test, 1000)
        config.run_ablations = False; config.run_regional_transfer = False; config.run_label_sensitivity = False; config.run_baseline_sensitivity = False
        args.review_seeds = "42,123"; args.multiseed_epochs = 2; args.multiseed_train_pairs = 3000; args.multiseed_validation_pairs = 1000; args.multiseed_test_pairs = 1000
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
    train_idx, val_idx, test_idx, split_metadata = operator_disjoint_split(df, SEED)
    bundle = build_bundle(df, train_idx, val_idx, test_idx, config, split_metadata)
    save_json(output_dir / "split_metadata.json", bundle.split_metadata)

    pair_summary = pd.DataFrame([
        {"Quantity": "Risk target y_i", "Source": "Reviews (Rating)", "Formula": "1 - training-scaled rating", "Interpretation": "Rating-derived risk proxy; not observed failure, complaint, or downtime", "Train": len(bundle.train_idx), "Validation": len(bundle.val_idx), "Test": len(bundle.test_idx), "TrainPairs": "--", "ValidationPairs": "--", "TestPairs": "--"},
        {"Quantity": "Demand target u_i", "Source": "Usage Stats (avg users/day)", "Formula": "training-scaled usage", "Interpretation": "Auxiliary demand target; excluded from inference inputs", "Train": len(bundle.train_idx), "Validation": len(bundle.val_idx), "Test": len(bundle.test_idx), "TrainPairs": "--", "ValidationPairs": "--", "TestPairs": "--"},
        {"Quantity": "Preference proxy g_i", "Source": "y_i and u_i", "Formula": f"{config.risk_weight:.2f}y_i + {config.demand_weight:.2f}u_i", "Interpretation": "Formula-derived intervention-priority proxy", "Train": len(bundle.train_idx), "Validation": len(bundle.val_idx), "Test": len(bundle.test_idx), "TrainPairs": "--", "ValidationPairs": "--", "TestPairs": "--"},
        {"Quantity": "Preference pair p_ij", "Source": "g_i", "Formula": f"sign(g_i-g_j), |g_i-g_j|>{config.pair_threshold}", "Interpretation": "Supervised pairwise preference", "Train": "--", "Validation": "--", "Test": "--", "TrainPairs": len(bundle.train_pairs[0]), "ValidationPairs": len(bundle.val_pairs[0]), "TestPairs": len(bundle.test_pairs[0])},
    ])
    pair_summary.to_csv(output_dir / "pair_and_split_summary.csv", index=False)

    print("[V5] Training full PRIME-EV model")
    full = train_prime_ev(bundle, config, device, name="PRIME-EV-Full")
    full.history.to_csv(output_dir / "histories" / "PRIME_EV_v5_FULL_history.csv", index=False)
    save_model_checkpoint(output_dir / "models" / "PRIME_EV_v5_FULL.pt", full, bundle, config)
    all_mu, all_sigma, all_usage, all_scores, all_latency = infer_all_candidates(full.model, bundle, config, device)
    full_row = {**full.test_metrics, **fixed_cutoff_metrics(bundle.g_test, full.test_scores, bundle.test_pairs), "Model": full.name, "BestEpoch": full.best_epoch, "ParameterCount": count_parameters(full.model), "ModelMemoryMB": memory_mb_for_model(full.model), "TrainingTime_seconds": full.train_seconds, "Latency_ms_per_station_test": full.latency_ms_per_station, "Latency_ms_per_station_all_candidates": all_latency, "TestTotalLoss": full.losses["total"], "TestRiskLoss": full.losses["risk"], "TestDemandLoss": full.losses["demand"], "TestRankingLoss": full.losses["rank"]}
    pd.DataFrame([full_row]).to_csv(output_dir / "full_model_metrics.csv", index=False)

    g_summary = plot_g_distribution(bundle, output_dir)
    unc = uncertainty_diagnostics(bundle.y_test, full.test_mu, full.test_sigma, output_dir)
    bounded_risk_baselines(bundle).to_csv(output_dir / "bounded_risk_formulation_comparison.csv", index=False)
    leakage_correlation_table(bundle).to_csv(output_dir / "leakage_feature_correlations.csv", index=False)
    fair = fairness_metrics(df, all_scores, all_mu, bundle.preprocessor, config.top_fraction)
    pd.DataFrame([{**fair, "MetricScope": "all candidate stations, top 10 percent selected", "GES_equation": "1 - mean(geographic selection disparity, accessibility selection disparity, low-access coverage gap)", "GES_note": "GES can remain high when low-access coverage is zero if the other disparity components are small; report low-access coverage separately."}]).to_csv(output_dir / "fairness_and_accessibility.csv", index=False)
    group_selection_diagnostics(df, all_scores, all_mu, config.top_fraction, output_dir)
    feature_counterfactual_tests(bundle, full, config, device, output_dir)
    feature_permutation_sensitivity(bundle, full, config, device, output_dir, repeats=5 if args.quick else 10)

    print("[V5] Evaluating baselines")
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
        ablation["ParameterCount"] = np.nan
        ablation["ModelMemoryMB"] = np.nan
        pfront = pareto_frontier(ablation)
        pfront.to_csv(output_dir / "ablation_pareto_frontier.csv", index=False)
        make_pareto_plot(pfront, output_dir)
    ablation.to_csv(output_dir / "corrected_ablation_table.csv", index=False)
    label_sens = run_label_weight_sensitivity(bundle, config, device) if config.run_label_sensitivity else pd.DataFrame()
    label_sens.to_csv(output_dir / "label_weight_sensitivity.csv", index=False)
    regional = run_regional_transfer(df, config, device) if config.run_regional_transfer else pd.DataFrame()
    regional.to_csv(output_dir / "regional_transfer_results.csv", index=False)

    if not args.skip_multiseed:
        multi, multi_ablation = run_multiseed(df, args, config, output_dir, device)
        multi_summary = summarize_by_group(multi, "Method", output_dir / "multiseed_baseline_summary.csv")
        ablation_summary = summarize_by_group(multi_ablation, "Variant", output_dir / "multiseed_ablation_summary.csv")
        significance_table(multi, "Method", "PRIME-EV", ["Random", "VIKOR", "Pareto_Balanced", "RidgeRanker", "TwoStage_RiskUsage"], "NDCG@100", output_dir / "significance_prime_vs_baselines.csv")
        significance_table(multi_ablation, "Variant", "Full", ["NoIRE_Conv", "PointwiseRanking", "NoAttention", "DeterministicRisk", "NoDIM"], "NDCG_full", output_dir / "significance_full_vs_ablations.csv")
    else:
        multi_summary = pd.DataFrame(); ablation_summary = pd.DataFrame()

    write_compact_latex(output_dir, baseline_fixed, multi_summary, ablation_summary, g_summary, unc)
    save_json(output_dir / "authoritative_manuscript_values.json", {"split": bundle.split_metadata, "full_model_metrics": full_row, "g_distribution": g_summary, "uncertainty_diagnostics": unc, "fairness": fair})
    print("\nPRIME-EV V5 reviewer-ready run completed")
    print(f"Output directory: {output_dir}")
    print("Use v5_compact_reviewer_table.tex and authoritative_manuscript_values.json in the manuscript.")


if __name__ == "__main__":
    main()
