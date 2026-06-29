#!/usr/bin/env python3
"""
W11 statistical reporting generator for PRIME-EV.

Purpose:
  Generate the exact values requested in Reviewer W11 without retraining:
  1. confidence intervals for PRIME-EV and all baselines;
  2. exact paired p-values;
  3. Holm-corrected p-values;
  4. Cohen's dz effect sizes;
  5. LaTeX-ready tables for manuscript Table 5.

How to use:
  Put this file inside your prime_ev_v6_results folder, then run:

      python w11_generate_statistics.py

Expected input files in the same folder:
  Preferred:
      multiseed_baseline_fixed_metrics.csv

  Also accepted:
      baseline_results.csv
      significance_prime_vs_baselines.csv

Important:
  This script does NOT train models and does NOT create new experimental results.
  It only summarizes existing seed-level CSV results.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = [
    "NDCG_full",
    "NDCG_at_10_percent",
    "Precision_at_10_percent",
    "Spearman",
    "PairwiseAccuracy",
]

METRIC_LABELS = {
    "NDCG_full": "NDCG",
    "NDCG_at_10_percent": "NDCG@10\\%",
    "Precision_at_10_percent": "P@10\\%",
    "Spearman": "Spearman",
    "PairwiseAccuracy": "Pairwise Acc.",
}

METHOD_RENAME = {
    "PRIME-EV": "PRIME-EV",
    "AHP_WeightedSum": "WS",
    "WeightedSum": "WS",
    "WS": "WS",
    "TOPSIS": "TOPSIS",
    "VIKOR": "VIKOR",
    "MultiObjective_WeightedSum": "MO Weighted Sum",
    "MO_WeightedSum": "MO Weighted Sum",
    "Pareto_Balanced": "Pareto Balanced",
    "GradientBoostedRanker": "GB Ranker",
    "GBRanker": "GB Ranker",
    "RandomForestRanker": "RF Ranker",
    "RFRanker": "RF Ranker",
    "RidgeRanker": "Ridge Ranker",
    "ShallowTreeRanker": "Tree Ranker",
    "TreeRanker": "Tree Ranker",
    "TwoStage_RiskUsage": "Risk+Usage Ranker",
    "Risk+Usage Ranker": "Risk+Usage Ranker",
    "RiskUsageRanker": "Risk+Usage Ranker",
    "Random": "Random",
    "Oracle_Label_UpperBound": "Oracle Label Upper Bound",
}

METHOD_ORDER = [
    "PRIME-EV",
    "WS",
    "TOPSIS",
    "VIKOR",
    "MO Weighted Sum",
    "Pareto Balanced",
    "GB Ranker",
    "RF Ranker",
    "Ridge Ranker",
    "Tree Ranker",
    "Risk+Usage Ranker",
    "Random",
]


def find_first_existing(patterns: List[str], base: Path) -> Optional[Path]:
    for pat in patterns:
        matches = sorted(base.glob(pat))
        if matches:
            return matches[0]
    return None


def normalize_method_name(name: str) -> str:
    name = str(name).strip()
    return METHOD_RENAME.get(name, name)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "NDCG": "NDCG_full",
        "NDCG@100": "NDCG_full",
        "NDCG@10pct": "NDCG_at_10_percent",
        "NDCG@10%": "NDCG_at_10_percent",
        "P@10pct": "Precision_at_10_percent",
        "P@10%": "Precision_at_10_percent",
        "Precision@10%": "Precision_at_10_percent",
        "Pairwise Acc.": "PairwiseAccuracy",
        "Pairwise Accuracy": "PairwiseAccuracy",
        "Kendall": "KendallTau",
    }

    out = df.rename(columns={c: rename.get(c, c) for c in df.columns}).copy()

    # Some existing CSVs contain both names such as NDCG and NDCG@100.
    # After renaming, pandas creates duplicate columns, and pd.to_numeric fails.
    # This block collapses duplicate column names by taking the first non-null value.
    if out.columns.duplicated().any():
        collapsed = {}
        for col in dict.fromkeys(out.columns):
            same = out.loc[:, out.columns == col]
            if same.shape[1] == 1:
                collapsed[col] = same.iloc[:, 0]
            else:
                collapsed[col] = same.bfill(axis=1).iloc[:, 0]
        out = pd.DataFrame(collapsed)

    if "Method" in out.columns:
        out["Method"] = out["Method"].map(normalize_method_name)

    return out


def add_seed_column_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Seed" in out.columns:
        return out
    if "seed" in out.columns:
        out = out.rename(columns={"seed": "Seed"})
        return out

    # If each method appears five times in order, infer seed index.
    if "Method" in out.columns and out.groupby("Method").size().max() > 1:
        out["Seed"] = out.groupby("Method").cumcount() + 1
    else:
        out["Seed"] = 1
    return out


def load_metric_table(base: Path) -> pd.DataFrame:
    path = find_first_existing(
        [
            "multiseed_baseline_fixed_metrics.csv",
            "multiseed_baseline_metrics.csv",
            "multiseed_baseline_results.csv",
            "baseline_results.csv",
            "baseline_results*.csv",
        ],
        base,
    )
    if path is None:
        raise FileNotFoundError(
            "Could not find a baseline metrics CSV. Place multiseed_baseline_fixed_metrics.csv "
            "or baseline_results.csv in this folder."
        )

    df = pd.read_csv(path)
    df = normalize_metric_columns(df)
    df = add_seed_column_if_missing(df)

    if "Method" not in df.columns:
        raise ValueError(f"{path.name} has no Method column.")

    keep = ["Seed", "Method"] + [m for m in PRIMARY_METRICS if m in df.columns]
    df = df[keep].copy()

    for col in keep:
        if col not in {"Seed", "Method"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Method"] = df["Method"].map(normalize_method_name)
    df = df.dropna(subset=["Method"])

    print(f"Loaded metric table: {path.name}")
    print(f"Rows: {len(df)}, Methods: {sorted(df['Method'].unique())}")
    return df


def ci95(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    else:
        half = 0.0
    return mean, std, mean - half, mean + half


def summarize_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("Method", sort=False):
        for metric in PRIMARY_METRICS:
            if metric not in g.columns:
                continue
            mean, std, lo, hi = ci95(g[metric].to_numpy())
            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "MetricLabel": METRIC_LABELS.get(metric, metric),
                    "NSeeds": int(g["Seed"].nunique()),
                    "Mean": mean,
                    "Std": std,
                    "CI95_Lower": lo,
                    "CI95_Upper": hi,
                }
            )
    out = pd.DataFrame(rows)
    out["MethodOrder"] = out["Method"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
    out["MetricOrder"] = out["Metric"].apply(lambda x: PRIMARY_METRICS.index(x) if x in PRIMARY_METRICS else 999)
    return out.sort_values(["MethodOrder", "MetricOrder"]).drop(columns=["MethodOrder", "MetricOrder"])


def holm_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def paired_stats(df: pd.DataFrame, control: str = "PRIME-EV") -> pd.DataFrame:
    rows = []
    if control not in set(df["Method"]):
        raise ValueError(f"Control method {control!r} not found in Method column.")

    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue

        p_rows = []
        methods = [m for m in METHOD_ORDER if m in set(df["Method"]) and m != control]
        methods += sorted([m for m in set(df["Method"]) if m not in METHOD_ORDER and m != control])

        for comp in methods:
            wide = df[df["Method"].isin([control, comp])].pivot_table(
                index="Seed",
                columns="Method",
                values=metric,
                aggfunc="mean",
            ).dropna()

            if control not in wide.columns or comp not in wide.columns or len(wide) < 2:
                continue

            diff = wide[control].to_numpy(dtype=float) - wide[comp].to_numpy(dtype=float)
            n = len(diff)
            mean_diff = float(np.mean(diff))
            sd_diff = float(np.std(diff, ddof=1))
            if sd_diff == 0:
                t_stat = 0.0
                p_val = 1.0
                dz = 0.0
            else:
                t_stat, p_val = stats.ttest_rel(wide[control], wide[comp])
                dz = mean_diff / sd_diff

            p_rows.append(
                {
                    "Control": control,
                    "Comparator": comp,
                    "Metric": metric,
                    "MetricLabel": METRIC_LABELS.get(metric, metric),
                    "MeanDifference": mean_diff,
                    "PairedT": float(t_stat),
                    "PairedP": float(p_val),
                    "CohenDz": float(dz),
                    "NSeeds": int(n),
                }
            )

        if p_rows:
            adjusted = holm_adjust([r["PairedP"] for r in p_rows])
            for r, hp in zip(p_rows, adjusted):
                r["HolmP"] = hp
                rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ComparatorOrder"] = out["Comparator"].apply(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
    out["MetricOrder"] = out["Metric"].apply(lambda x: PRIMARY_METRICS.index(x) if x in PRIMARY_METRICS else 999)
    return out.sort_values(["MetricOrder", "ComparatorOrder"]).drop(columns=["MetricOrder", "ComparatorOrder"])


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def fmt_ci(lo: float, hi: float, digits: int = 4) -> str:
    return f"$[{lo:.{digits}f},{hi:.{digits}f}]$"


def make_table5a_latex(summary: pd.DataFrame, out_path: Path) -> str:
    metrics = PRIMARY_METRICS
    rows = []
    methods = [m for m in METHOD_ORDER if m in set(summary["Method"])]
    for method in methods:
        cells = [method]
        for metric in metrics:
            r = summary[(summary["Method"] == method) & (summary["Metric"] == metric)]
            if r.empty:
                cells.append("--")
            else:
                rr = r.iloc[0]
                cells.append(fmt_pm(rr["Mean"], rr["Std"]))
        rows.append(" & ".join(cells) + r"\\")

    latex = r"""\begin{subtable}[t]{\textwidth}
\centering
\caption{Comparison with deployable baselines. Values are mean $\pm$ standard deviation over five independent seeds.}
\label{tab:baseline_comparison_main}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{NDCG} & \textbf{NDCG@10\%} & \textbf{P@10\%} & \textbf{Spearman} & \textbf{Pairwise Acc.}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{subtable}
"""
    out_path.write_text(latex, encoding="utf-8")
    return latex


def make_table5b_stats_latex(stats_df: pd.DataFrame, out_path: Path) -> str:
    if stats_df.empty:
        latex = "% No paired statistics could be generated.\n"
        out_path.write_text(latex, encoding="utf-8")
        return latex

    rows = []
    for _, r in stats_df.iterrows():
        rows.append(
            f"{r['Comparator']} & {r['MetricLabel']} & "
            f"${r['MeanDifference']:.4f}$ & ${r['PairedP']:.4f}$ & "
            f"${r['HolmP']:.4f}$ & ${r['CohenDz']:.3f}$\\\\"
        )

    latex = r"""\begin{subtable}[t]{\textwidth}
\centering
\caption{Paired PRIME-EV versus baseline statistical comparisons. Positive mean differences favor PRIME-EV.}
\label{tab:baseline_statistical_tests}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Comparator} & \textbf{Metric} & \textbf{Mean Diff.} & \textbf{Paired $p$} & \textbf{Holm $p$} & \textbf{Cohen's $d_z$}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{subtable}
"""
    out_path.write_text(latex, encoding="utf-8")
    return latex


def make_ci_table_latex(summary: pd.DataFrame, out_path: Path) -> str:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['Method']} & {r['MetricLabel']} & "
            f"${r['Mean']:.4f} \\pm {r['Std']:.4f}$ & "
            f"$[{r['CI95_Lower']:.4f},{r['CI95_Upper']:.4f}]$\\\\"
        )

    latex = r"""\begin{subtable}[t]{\textwidth}
\centering
\caption{Baseline metric confidence intervals computed across independent seeds.}
\label{tab:baseline_metric_confidence_intervals}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Method} & \textbf{Metric} & \textbf{Mean $\pm$ Std.} & \textbf{95\% CI}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{subtable}
"""
    out_path.write_text(latex, encoding="utf-8")
    return latex


def main() -> None:
    base = Path(".").resolve()

    df = load_metric_table(base)
    summary = summarize_with_ci(df)
    stat_tests = paired_stats(df, control="PRIME-EV")

    summary_path = base / "w11_baseline_metric_ci.csv"
    stats_path = base / "w11_prime_vs_baseline_stats_all_metrics.csv"
    summary.to_csv(summary_path, index=False)
    stat_tests.to_csv(stats_path, index=False)

    make_table5a_latex(summary, base / "w11_table5a_baseline_comparison.tex")
    make_table5b_stats_latex(stat_tests, base / "w11_table5b_statistical_tests.tex")
    make_ci_table_latex(summary, base / "w11_table5c_baseline_ci.tex")

    print("\nSaved:")
    print(f"  {summary_path.name}")
    print(f"  {stats_path.name}")
    print("  w11_table5a_baseline_comparison.tex")
    print("  w11_table5b_statistical_tests.tex")
    print("  w11_table5c_baseline_ci.tex")

    if df.groupby("Method")["Seed"].nunique().max() < 2:
        print("\nWARNING:")
        print("  The input file appears to contain only one row per method.")
        print("  Confidence intervals and paired tests require seed-level rows.")
        print("  Use multiseed_baseline_fixed_metrics.csv if available.")


if __name__ == "__main__":
    main()
