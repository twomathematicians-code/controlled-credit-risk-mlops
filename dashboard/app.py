"""Streamlit business dashboard for the credit-risk PD model.

Run:  streamlit run dashboard/app.py

Tabs:
  * Portfolio      — headline KPIs + score distribution
  * Threshold & Cost — approval/loss tradeoff + interactive threshold simulator
  * Drift          — PSI/KS report vs the frozen reference (with a drift simulator)
  * Explainability — global SHAP summary
  * Score applicant — single-request scorer with reason codes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk import business  # noqa: E402
from credit_risk.config import PROJECT_ROOT, get_config  # noqa: E402
from credit_risk.models import registry  # noqa: E402
from credit_risk.monitoring import drift  # noqa: E402


@st.cache_resource
def load_artifacts():
    cfg = get_config()
    feat = list(cfg.data.numeric_features) + list(cfg.data.categorical_features)
    test = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")
    pipeline = registry.load_production_model()
    scores = pipeline.predict_proba(test[feat])[:, 1]
    y = test[cfg.data.target_column].to_numpy()
    metrics = json.loads((PROJECT_ROOT / "data" / "processed" / "model_metrics.json").read_text())
    curve = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "threshold_curve.csv")
    version = registry.production_version()
    return cfg, feat, test, pipeline, scores, y, metrics, curve, version


def _applicant_form(cfg, pipeline):
    st.subheader("Score a single applicant")
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 90, 40)
        income = c2.number_input("Annual income", 8000, 400000, 60000, step=1000)
        months = c3.number_input("Months employed", 0, 360, 48)
        c4, c5, c6 = st.columns(3)
        open_acc = c4.number_input("Open accounts", 0, 30, 4)
        inquiries = c5.number_input("Inquiries (12m)", 0, 20, 1)
        missed = c6.number_input("Missed payments (12m)", 0, 12, 0)
        c7, c8 = st.columns(2)
        debt = c7.number_input("Total debt", 0, 500000, 12000, step=1000)
        limit = c8.number_input("Credit limit", 0, 200000, 18000, step=1000)
        balance = c7.number_input("Card balance", 0, 200000, 4000, step=500)
        emp = c2.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed", "Retired"])
        home = c4.selectbox("Home ownership", ["Rent", "Mortgage", "Own", "Other"])
        purpose = c5.selectbox("Loan purpose", [
            "debt_consolidation", "home_improvement", "major_purchase", "credit_card", "other"])
        region = c6.selectbox("Region", ["Capital", "North", "South", "West", "East"])
        submitted = st.form_submit_button("Score")
    if submitted:
        row = pd.DataFrame([{
            "age": int(age), "annual_income": float(income), "months_employed": int(months),
            "num_open_accounts": int(open_acc), "num_credit_inquiries_12m": int(inquiries),
            "total_debt": float(debt), "credit_limit": float(limit),
            "missed_payments_12m": int(missed), "credit_card_balance": float(balance),
            "employment_status": emp, "home_ownership": home, "loan_purpose": purpose, "region": region,
        }])
        pd_score = float(pipeline.predict_proba(row)[0, 1])
        threshold = _threshold(metrics_holder["metrics"])
        decision = "DECLINE" if pd_score >= threshold else "APPROVE"
        col_a, col_b = st.columns(2)
        col_a.metric("Probability of default", f"{pd_score:.1%}")
        col_b.metric(f"Decision @ threshold {threshold:.2f}", decision)
        try:
            from credit_risk.models.explainability import Explainer, top_reasons
            ref = pd.read_parquet(PROJECT_ROOT / "data" / "drift_reference" / "reference.parquet")
            ref = ref.drop(columns=[cfg.data.target_column], errors="ignore")
            expl = Explainer(pipeline, background=ref)
            ex = expl.explain(row)
            st.caption("Top contributors")
            st.dataframe(pd.DataFrame(top_reasons(ex, 0, k=5)))
        except Exception as exc:  # pragma: no cover
            st.caption(f"Reason codes unavailable: {exc}")


def _threshold(metrics):
    return float(metrics.get("optimal_threshold", get_config().serving.default_threshold))


metrics_holder = {"metrics": {}}


def main():  # pragma: no cover - UI entry point
    st.set_page_config(page_title="Credit Risk PD — Dashboard", layout="wide")
    st.title("Credit Risk — Probability of Default")
    st.caption("Business presentation layer for the controlled PD scoring system.")

    try:
        cfg, feat, test, pipeline, scores, y, metrics, curve, version = load_artifacts()
        metrics_holder["metrics"] = metrics
    except Exception as exc:
        st.error(f"Could not load model/data: {exc}")
        st.info("Run `make data && make train` first to generate the data and train the model.")
        return

    # ---- Header KPIs ----
    m1, m2, m3, m4 = st.columns(4)
    auc = metrics.get("test_roc_auc", float("nan"))
    ci = metrics.get("test_roc_auc_ci", [float("nan"), float("nan")])
    opt = _threshold(metrics)
    opt_m = metrics.get("optimal_threshold_metrics", {})
    m1.metric("Test ROC-AUC", f"{auc:.3f}", f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    m2.metric("Champion model", metrics.get("champion", "?"), f"Production v{version}")
    m3.metric("Cost-optimal threshold", f"{opt:.2%}")
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
        st.caption(f"Cost-optimal cutoff minimises realised loss + opportunity cost: **{opt:.2%}**")

        st.markdown("---")
        st.subheader("Threshold simulator")
        t = st.slider("Decision threshold (approve if PD < threshold)", 0.01, 0.99, opt, 0.01)
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
        report = drift.evaluate(current)
        st.metric("Overall status", report["overall_status"])
        rows = []
        for name, info in report["features"].items():
            rows.append({"feature": name, **info})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab4:
        st.subheader("Global SHAP summary")
        png = PROJECT_ROOT / "data" / "processed" / "shap_summary.png"
        if png.exists():
            st.image(str(png), use_container_width=True)
            st.caption("Top features driving probability-of-default (mean |SHAP|).")
        else:
            st.info("No SHAP summary found. It is produced during training.")

    with tab5:
        _applicant_form(cfg, pipeline)


if __name__ == "__main__":  # pragma: no cover
    main()
