"""EDA + statistical validation (plain script — runs headless, no notebook needed).

Produces:
  * data/processed/eda_target_by_feature.png  — univariate signal
  * data/processed/eda_calibration.png        — reliability (calibration) curve
  * console: default rates, correlations, bootstrap AUC CI

Run:  python notebooks/01_eda_and_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from credit_risk.config import PROJECT_ROOT, get_config
from credit_risk.models import registry, train

OUT = PROJECT_ROOT / "data" / "processed"


def load():
    cfg = get_config()
    train_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "train.parquet")
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")
    return cfg, train_df, test_df


def target_by_feature(cfg, df):
    numeric = list(cfg.data.numeric_features)
    target = cfg.data.target_column
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), numeric, strict=False):
        for label, sub in df.groupby(target):
            ax.hist(sub[col], bins=30, alpha=0.5, label=f"default={label}", density=True)
        ax.set_title(col)
        ax.legend(fontsize=7)
    fig.suptitle("Feature distributions by default flag (train)")
    fig.tight_layout()
    fig.savefig(OUT / "eda_target_by_feature.png", dpi=110)
    plt.close(fig)


def calibration(cfg, test_df):
    target = cfg.data.target_column
    feat = list(cfg.data.numeric_features) + list(cfg.data.categorical_features)
    pipeline = registry.load_production_model()
    scores = pipeline.predict_proba(test_df[feat])[:, 1]
    frac_pos, mean_pred = calibration_curve(test_df[target], scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "s-", label="model")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Fraction defaulted")
    ax.set_title("Calibration (reliability) curve — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "eda_calibration.png", dpi=110)
    plt.close(fig)
    return scores


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, train_df, test_df = load()
    target = cfg.data.target_column
    print(f"Train: {len(train_df)} rows | default rate = {train_df[target].mean():.3f}")
    print(f"Test : {len(test_df)} rows | default rate = {test_df[target].mean():.3f}")

    numeric = list(cfg.data.numeric_features)
    print("\nCorrelation of numeric features with default_flag:")
    corr = train_df[numeric + [target]].corr(numeric_only=True)[target].drop(target).sort_values(
        key=np.abs, ascending=False
    )
    print(corr.round(3).to_string())

    target_by_feature(cfg, train_df)
    scores = calibration(cfg, test_df)

    # Bootstrap AUC CI (statistical validation)
    point, lo, hi = train.bootstrap_auc_ci(test_df[target].to_numpy(), scores, n_boot=300)
    print(f"\nROC-AUC = {point:.4f}  (95% CI {lo:.4f}, {hi:.4f})")
    print(f"Plots written to {OUT}")


if __name__ == "__main__":  # pragma: no cover
    main()
