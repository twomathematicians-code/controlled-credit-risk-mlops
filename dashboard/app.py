"""Streamlit business dashboard for the credit-risk PD model.

Run:  streamlit run dashboard/app.py

Tabs:
  * Portfolio       — headline KPIs + score distribution
  * Threshold & Cost — approval/loss tradeoff + interactive threshold simulator
  * Drift           — PSI/KS report vs the frozen reference (with a drift simulator)
  * Explainability  — global SHAP summary
  * Score applicant — single-request scorer with reason codes

Streamlit Community Cloud ready
-------------------------------
If the locally-trained artifacts are absent (e.g. a fresh checkout on Community
Cloud), ``load_artifacts`` transparently trains a fast lightweight model in
memory and caches it — so the dashboard works out-of-the-box with no setup.
Nothing on disk is required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for cloud
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk import business  # noqa: E402
from credit_risk.config import PROJECT_ROOT, get_config  # noqa: E402
from credit_risk.models import registry  # noqa: E402
from credit_risk.monitoring import drift  # noqa: E402

_ARTIFACTS = {
    PROJECT_ROOT / "data" / "processed" / "test.parquet",
    PROJECT_ROOT / "data" / "processed" / "model_metrics.json",
    PROJECT_ROOT / "data" / "drift_reference" / "reference.parquet",
}


def _threshold(metrics: dict) -> float:
    return float(metrics.get("optimal_threshold", get_config().serving.default_threshold))


def _try_load_registry_model():
    """Return the Production pipeline if a registry is available, else None."""
    try:
        return registry.load_production_model()
    except Exception:
        return None


def _demo_bootstrap():
    """Train a fast lightweight model in memory (Community Cloud / fresh checkout).

    Returns the same bundle as the real path: a fitted pipeline, test split,
    scores, metrics dict, threshold curve, reference features, and a SHAP figure.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from credit_risk.data import huggingface, ingestion, synthetic
    from credit_risk.features.engineering import build_pipeline
    from credit_risk.models import train as trainmod

    cfg = get_config()
    feat = list(cfg.data.numeric_features) + list(cfg.data.categorical_features)
    target = cfg.data.target_column

    # Cloud demo: real Home Credit data (small sample). Offline fallback: synthetic.
    try:
        raw = huggingface.load(sample_size=8000, seed=42)
        data_source = "Home Credit (demo sample)"
    except Exception:
        raw = synthetic.generate(n_samples=8000, seed=42)
        data_source = "synthetic (offline)"
    df = ingestion.validate_schema(raw)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(train_df[feat], train_df[target])

    scores = pipeline.predict_proba(test_df[feat])[:, 1]
    y = test_df[target].to_numpy()
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    point, lo, hi = trainmod.bootstrap_auc_ci(y, scores, n_boot=120, seed=42)
    opt_t, opt_m = business.optimal_threshold(y, scores)
    curve = business.threshold_curve(y, scores)
    metrics = {
        "champion": f"logreg ({data_source})",
        "test_roc_auc": auc,
        "test_average_precision": ap,
        "test_roc_auc_ci": [lo, hi],
        "optimal_threshold": opt_t,
        "optimal_threshold_metrics": opt_m,
    }
    reference = train_df[feat].copy()
    shap_fig = _build_shap_figure(pipeline, reference, test_df[feat])
    return cfg, feat, test_df, pipeline, scores, y, metrics, curve, reference, "demo", shap_fig


def _build_shap_figure(pipeline, reference, X_eval):
    """Best-effort global SHAP summary as an in-memory matplotlib figure."""
    try:
        import shap

        from credit_risk.models.explainability import Explainer

        explainer = Explainer(pipeline, background=reference)
        explanation = explainer.explain(X_eval.iloc[: min(400, len(X_eval))])
        shap.summary_plot(
            explanation.values,
            features=None,
            feature_names=explanation.feature_names,
            show=False,
            max_display=12,
        )
        fig = plt.gcf()
        return fig
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading model & data…")
def load_artifacts():
    """Load trained artifacts. Falls back to a cached in-memory demo if absent."""
    cfg = get_config()
    feat = list(cfg.data.numeric_features) + list(cfg.data.categorical_features)

    real_available = all(p.exists() for p in _ARTIFACTS) and _try_load_registry_model() is not None
    if not real_available:
        return _demo_bootstrap()

    pipeline = registry.load_production_model()
    test = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")
    scores = pipeline.predict_proba(test[feat])[:, 1]
    y = test[cfg.data.target_column].to_numpy()
    metrics = json.loads((PROJECT_ROOT / "data" / "processed" / "model_metrics.json").read_text())
    curve = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "threshold_curve.csv")
    version = registry.production_version() or "local"
    reference = pd.read_parquet(PROJECT_ROOT / "data" / "drift_reference" / "reference.parquet")
    reference = reference.drop(columns=[cfg.data.target_column], errors="ignore")[feat]
    shap_fig = _build_shap_figure(pipeline, reference, test[feat])
    return cfg, feat, test, pipeline, scores, y, metrics, curve, reference, version, shap_fig


def _applicant_form(cfg, pipeline, reference, threshold, sample_df):
    """Data-driven single-applicant scorer built from the live feature schema."""
    st.subheader("Score a single applicant")
    num_cols = list(cfg.data.numeric_features)
    cat_cols = list(cfg.data.categorical_features)
    with st.form("applicant"):
        values: dict = {}
        cols = st.columns(3)
        for i, col in enumerate(num_cols):
            s = pd.to_numeric(sample_df[col], errors="coerce").dropna()
            med = float(s.median()) if len(s) else 0.0
            mn, mx = float(s.min()) if len(s) else 0.0, float(s.max()) if len(s) else 1.0
            step = (mx - mn) / 100.0 or 1.0
            with cols[i % 3]:
                values[col] = st.number_input(col, min_value=mn, max_value=mx, value=med, step=step, key=f"n_{col}")
        cols2 = st.columns(3)
        for i, col in enumerate(cat_cols):
            levels = sorted(sample_df[col].astype(str).unique().tolist())
            with cols2[i % 3]:
                values[col] = st.selectbox(col, levels, key=f"c_{col}")
        submitted = st.form_submit_button("Score")
    if submitted:
        row = pd.DataFrame([values])
        pd_score = float(pipeline.predict_proba(row)[0, 1])
        decision = "DECLINE" if pd_score >= threshold else "APPROVE"
        col_a, col_b = st.columns(2)
        col_a.metric("Probability of default", f"{pd_score:.1%}")
        col_b.metric(f"Decision @ threshold {threshold:.2f}", decision)
        try:
            from credit_risk.models.explainability import Explainer, top_reasons

            expl = Explainer(pipeline, background=reference)
            ex = expl.explain(row)
            st.caption("Top contributors")
            st.dataframe(pd.DataFrame(top_reasons(ex, 0, k=5)))
        except Exception as exc:  # pragma: no cover
            st.caption(f"Reason codes unavailable: {exc}")


def main():  # pragma: no cover - UI entry point
    st.set_page_config(page_title="Credit Risk PD — Dashboard", layout="wide")
    st.title("Credit Risk — Probability of Default")
    st.caption("Business presentation layer for the controlled PD scoring system.")

    try:
        (cfg, feat, test, pipeline, scores, y, metrics, curve,
         reference, version, shap_fig) = load_artifacts()
    except Exception as exc:
        st.error(f"Could not load model/data: {exc}")
        return

    threshold = _threshold(metrics)
    is_demo = version == "demo"
    if is_demo:
        st.info(
            "Demo mode: no trained artifacts found, so a fast in-memory model was trained "
            "for this preview. Run `make data && make train` locally for the full model."
        )

    # ---- Header KPIs ----
    m1, m2, m3, m4 = st.columns(4)
    auc = metrics.get("test_roc_auc", float("nan"))
    ci = metrics.get("test_roc_auc_ci", [float("nan"), float("nan")])
    opt_m = metrics.get("optimal_threshold_metrics", {})
    m1.metric("Test ROC-AUC", f"{auc:.3f}", f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    m2.metric("Champion model", metrics.get("champion", "?"), f"v{version}")
    m3.metric("Cost-optimal threshold", f"{threshold:.2%}")
    m4.metric("Approval rate @ opt", f"{opt_m.get('approval_rate', 0):.1%}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Portfolio", "Threshold & Cost", "Drift", "Explainability", "Score applicant"]
    )

    with tab1:
        st.subheader("Score distribution (test set)")
        st.area_chart(pd.DataFrame({"PD": scores})["PD"].clip(0, 1))
        st.caption(f"Realised default rate in test set: **{y.mean():.2%}**")

    with tab2:
        st.subheader("Approval rate vs realised loss")
        st.line_chart(curve.set_index("threshold")[["approval_rate", "realised_loss", "opportunity_cost"]])
        st.caption(f"Cost-optimal cutoff minimises realised loss + opportunity cost: **{threshold:.2%}**")

        st.markdown("---")
        st.subheader("Threshold simulator")
        t = st.slider("Decision threshold (approve if PD < threshold)", 0.01, 0.99, threshold, 0.01)
        sim = business.portfolio_metrics(y, scores, t)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Approval rate", f"{sim['approval_rate']:.1%}")
        s2.metric("Expected loss (forward)", f"{sim['expected_loss_total']:,.0f}")
        s3.metric("Realised loss (FN)", f"{sim['realised_loss']:,.0f}")
        s4.metric("Opportunity cost (FP)", f"{sim['opportunity_cost']:,.0f}")

    with tab3:
        st.subheader("Drift vs frozen reference")
        simulate = st.checkbox("Simulate drift (shift credit utilisation upward)")
        current = test[feat].copy()
        if simulate:
            current["credit_card_balance"] = current["credit_card_balance"] * 1.6
        # In-memory drift report (no file dependency).
        numeric_cols = list(cfg.data.numeric_features)
        categorical_cols = list(cfg.data.categorical_features)
        feature_report = drift.feature_drift_report(
            reference, current, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )
        status = drift.overall_status(feature_report)
        st.metric("Overall status", status)
        rows = [{"feature": name, **info} for name, info in feature_report.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab4:
        st.subheader("Global SHAP summary")
        if shap_fig is not None:
            st.pyplot(shap_fig)
            st.caption("Top features driving probability-of-default (mean |SHAP|).")
        else:
            st.info("SHAP summary unavailable in this environment.")

    with tab5:
        _applicant_form(cfg, pipeline, reference, threshold, test)


if __name__ == "__main__":  # pragma: no cover
    main()
