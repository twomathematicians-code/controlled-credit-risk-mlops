# Streamlit dashboard

Business presentation layer for the controlled credit-risk PD system.

```
streamlit run dashboard/app.py
```

## Deploy to Streamlit Community Cloud

The dashboard is **self-contained and auto-bootstrapping**: if no trained model or
data is present (i.e. a fresh clone on Community Cloud), it trains a fast
lightweight logistic-regression model in memory on first load and caches it — so
it runs out-of-the-box with zero setup. (Locally, when `make data && make train`
has been run, it serves the real registered model instead.)

Steps:

1. Push this repo to GitHub (it already lives at
   `twomathematicians-code/controlled-credit-risk-mlops`).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **New app** → select the repo + the `main` branch.
4. Configure:
   - **Main file path:** `dashboard/app.py`
   - **Requirements file:** `dashboard/requirements.txt` *(slim — builds faster)*
   - Python version: **3.11**
5. **Deploy.** First load trains the demo model (~10–20s) and is then cached.

> The slim `dashboard/requirements.txt` omits xgboost (the cloud demo uses a
> logistic regression; SHAP explainability falls back gracefully). The full
> training stack (incl. xgboost) lives in the repo-root `requirements.txt`.

### Tabs

| Tab | What it shows |
|---|---|
| Portfolio | Score distribution + realised default rate |
| Threshold & Cost | Approval/loss trade-off + interactive threshold simulator |
| Drift | PSI/KS drift report vs frozen reference (with a drift simulator) |
| Explainability | Global SHAP summary |
| Score applicant | Single-request scorer with SHAP reason codes |
