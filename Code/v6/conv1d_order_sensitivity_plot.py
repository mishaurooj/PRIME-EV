# save as conv1d_order_sensitivity_plot.py

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PRIME_EV_v6 as pe


def permute_bundle_features(bundle, permutation):
    b = copy.deepcopy(bundle)
    b.X_train = b.X_train[:, permutation]
    b.X_val = b.X_val[:, permutation]
    b.X_test = b.X_test[:, permutation]
    b.preprocessor.feature_names = [b.preprocessor.feature_names[i] for i in permutation]
    return b


def run_order_sensitivity(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    pe.SEED = args.seed
    pe.set_seed(args.seed)

    df = pd.read_csv(args.data)
    pe.ensure_required_columns(df)

    config = pe.ExperimentConfig(
        data_path=args.data,
        output_dir=str(out),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        latent_dim=args.latent_dim,
        batch_pairs_train=args.train_pairs,
        batch_pairs_val=args.val_pairs,
        batch_pairs_test=args.test_pairs,
        pair_threshold=args.pair_threshold,
        lambda_risk=args.lambda_risk,
        lambda_demand=args.lambda_demand,
        lambda_rank=args.lambda_rank,
        torch_threads=args.torch_threads,
        device=args.device,
    )

    import torch
    torch.set_num_threads(args.torch_threads)
    device = pe.choose_device(args.device)

    train_idx, val_idx, test_idx, split_meta = pe.operator_disjoint_split(df, args.seed)
    base_bundle = pe.build_bundle(df, train_idx, val_idx, test_idx, config, split_meta)

    n_features = base_bundle.X_train.shape[1]
    rng = np.random.default_rng(args.seed + 999)

    methods = {
        "Conv1D+Attention": {},
        "Conv1D+MeanPool": {"no_attention": True, "seed_offset": 10},
        "MLP_NoConv": {"no_ire": True, "seed_offset": 20},
        "PointwiseRanking": {"pointwise_rank": True, "seed_offset": 30},
    }

    rows = []

    for perm_id in range(args.permutations + 1):
        if perm_id == 0:
            permutation = np.arange(n_features)
            perm_name = "Original"
        else:
            permutation = rng.permutation(n_features)
            perm_name = f"Perm{perm_id}"

        bundle = permute_bundle_features(base_bundle, permutation)

        for method_name, variant in methods.items():
            print(f"\nTraining {method_name} under {perm_name}")

            variant = dict(variant)
            variant["seed_offset"] = variant.get("seed_offset", 0) + perm_id * 100

            result = pe.train_prime_ev(
                bundle=bundle,
                config=config,
                device=device,
                name=f"{method_name}_{perm_name}",
                variant=variant,
                epochs_override=args.epochs,
            )

            fixed = pe.fixed_cutoff_metrics(
                bundle.g_test,
                result.test_scores,
                bundle.test_pairs,
            )

            rows.append({
                "Permutation": perm_id,
                "PermutationName": perm_name,
                "Method": method_name,
                "NDCG_full": fixed["NDCG_full"],
                "NDCG_at_10_percent": fixed["NDCG@10pct"],
                "P_at_10_percent": fixed["P@10pct"],
                "PairwiseAccuracy": fixed["PairwiseAccuracy"],
                "Spearman": fixed["Spearman"],
                "KendallTau": fixed["KendallTau"],
                "Latency_ms_per_station": result.latency_ms_per_station,
                "Train_seconds": result.train_seconds,
                "Best_epoch": result.best_epoch,
                "Params": pe.count_parameters(result.model),
            })

    results = pd.DataFrame(rows)
    results.to_csv(out / "conv1d_feature_order_sensitivity.csv", index=False)

    plot_metrics = [
        ("NDCG_full", "Full-list NDCG"),
        ("NDCG_at_10_percent", "NDCG@10%"),
        ("P_at_10_percent", "P@10%"),
        ("PairwiseAccuracy", "Pairwise accuracy"),
        ("Latency_ms_per_station", "Latency"),
    ]

    fig, axes = plt.subplots(
        1,
        len(plot_metrics),
        figsize=(4.2 * len(plot_metrics), 4.2),
        dpi=900,
        constrained_layout=True,
    )

    if len(plot_metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, plot_metrics):
        for method_name in methods:
            sub = results[results["Method"] == method_name].sort_values("Permutation")
            ax.plot(
                sub["Permutation"],
                sub[metric],
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=method_name,
            )

        ax.set_title(title, fontsize=14, fontname="Times New Roman")
        ax.set_xlabel("Feature order", fontsize=12, fontname="Times New Roman")
        ax.set_ylabel("Value", fontsize=12, fontname="Times New Roman")
        ax.grid(True, linestyle=":", linewidth=0.8)
        ax.set_xticks(range(args.permutations + 1))
        ax.set_xticklabels(
            ["Orig"] + [f"P{i}" for i in range(1, args.permutations + 1)],
            rotation=0,
            fontsize=10,
            fontname="Times New Roman",
        )

    axes[-1].legend(
        loc="best",
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(
        "Feature-Order Sensitivity of PRIME-EV Encoders",
        fontsize=18,
        fontweight="bold",
        fontname="Times New Roman",
    )

    fig.savefig(out / "conv1d_feature_order_sensitivity_1xn.png", dpi=900, bbox_inches="tight")
    fig.savefig(out / "conv1d_feature_order_sensitivity_1xn.pdf", dpi=900, bbox_inches="tight")
    plt.close(fig)

    summary = (
        results
        .groupby("Method")
        .agg({
            "NDCG_full": ["mean", "std", "min", "max"],
            "NDCG_at_10_percent": ["mean", "std", "min", "max"],
            "P_at_10_percent": ["mean", "std", "min", "max"],
            "PairwiseAccuracy": ["mean", "std", "min", "max"],
            "Latency_ms_per_station": ["mean", "std", "min", "max"],
        })
    )
    summary.to_csv(out / "conv1d_feature_order_sensitivity_summary.csv")

    print("\nSaved:")
    print(out / "conv1d_feature_order_sensitivity.csv")
    print(out / "conv1d_feature_order_sensitivity_summary.csv")
    print(out / "conv1d_feature_order_sensitivity_1xn.png")
    print(out / "conv1d_feature_order_sensitivity_1xn.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="conv1d_order_sensitivity_results")
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-pairs", type=int, default=20000)
    parser.add_argument("--val-pairs", type=int, default=5000)
    parser.add_argument("--test-pairs", type=int, default=5000)
    parser.add_argument("--pair-threshold", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--lambda-demand", type=float, default=0.2)
    parser.add_argument("--lambda-rank", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    run_order_sensitivity(args)