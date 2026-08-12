"""Generate polished README figures into docs/images/ (committed assets).

Produces:
  * docs/images/results_overview.png  — score dist + ROC + threshold/cost + calibration
  * docs/images/shap_summary.png      — copy of the training-time global SHAP plot

Run after training:  python docs/make_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from credit_risk.config import PROJECT_ROOT, get_config
from credit_risk.models import registry

IMG = ROOT / "docs" / "images"


def _load():
    cfg = get_config()
    feat = list(cfg.data.numeric_features) + list(cfg.data.categorical_features)
    test = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")
    pipe = registry.load_production_model()
    scores = pipe.predict_proba(test[feat])[:, 1]
    y = test[cfg.data.target_column].to_numpy()
    curve = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "threshold_curve.csv")
    return cfg, y, scores, curve


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    cfg, y, scores, curve = _load()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) PD score distribution by outcome
    ax = axes[0, 0]
    ax.hist(scores[y == 0], bins=40, alpha=0.6, label="repaid", density=True)
    ax.hist(scores[y == 1], bins=40, alpha=0.6, label="defaulted", density=True)
    ax.set_title("Score distribution by outcome (test set)")
    ax.set_xlabel("Predicted probability of default")
    ax.legend()

    # (b) ROC curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")

    # (c) Threshold vs approval rate + costs (cost-optimal threshold marked)
    ax = axes[1, 0]
    ax.plot(curve["threshold"], curve["approval_rate"], label="approval rate")
    ax.plot(curve["threshold"], curve["realised_loss"] / curve["realised_loss"].max(),
            label="realised loss (norm.)")
    ax.plot(curve["threshold"], curve["opportunity_cost"] / curve["opportunity_cost"].max(),
            label="opportunity cost (norm.)")
    import json
    metrics = json.loads((PROJECT_ROOT / "data" / "processed" / "model_metrics.json").read_text())
    opt = metrics["optimal_threshold"]
    ax.axvline(opt, color="k", ls=":", lw=1, label=f"optimum ≈ {opt:.1%}")
    ax.set_title("Approval / cost trade-off vs threshold")
    ax.set_xlabel("Decision threshold (approve if PD < threshold)")
    ax.legend(fontsize=8)

    # (d) Calibration curve
    ax = axes[1, 1]
    frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "s-", label="model")
    ax.set_title("Calibration (reliability) curve")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default fraction")
    ax.legend()

    fig.suptitle("Credit-Risk PD Model — test-set evaluation", fontsize=14)
    fig.tight_layout()
    out = IMG / "results_overview.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    shap_src = PROJECT_ROOT / "data" / "processed" / "shap_summary.png"
    if shap_src.exists():
        shutil.copy(shap_src, IMG / "shap_summary.png")
        print("copied shap_summary.png")


if __name__ == "__main__":  # pragma: no cover
    main()
