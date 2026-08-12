# Azure Deployment Guide

This folder shows how to take the local MLOps pipeline and run it on **Azure**,
orchestrated and governed. The guide is designed to be useful **without an Azure
subscription**: every script runs locally first, and the Azure pieces are shown
as the production target.

> See `architecture.md` for the full reference architecture and control model.

## 1. Prerequisites (Azure side)

```bash
# One-time setup (requires Contributor on the resource group)
az group create --name rg-credit-risk --location westeurope
az ml workspace create --name ml-credit-risk -g rg-credit-risk
az storage account create --name strcreditrisk -g rg-credit-risk --sku Standard_LRS
az ml compute create --name cpu-cluster --type amlcompute --size Standard_DS3_v2 \
  --min-instances 0 --max-instances 4 -g rg-credit-risk -w ml-credit-risk
```

## 2. Register the data as an AML data asset

In production the synthetic generator is replaced by curated data landed by Azure
Data Factory into ADLS Gen2. Register it once:

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

client = MLClient(DefaultAzureCredential(), "<sub>", "rg-credit-risk", "ml-credit-risk")
client.data.create_or_update(Data(
    name="credit_applicants",
    version="1",
    description="Curated credit applicant features (raw scoring view)",
    path="abfss://curated@strcreditrisk.dfs.core.windows.net/applicants/",
    type=AssetTypes.URI_FOLDER,
))
```

## 3. Submit the training job

`azureml_train.py` is the AML-shaped entry point — identical to
`python -m credit_risk.models.train`, but it auto-detects the AML context
(workspace-provided MLflow URI, mounted dataset) so the same script runs locally
*and* on an AML compute cluster. Validate it locally first:

```bash
python deployment/azure/azureml_train.py
```

To submit it as a governed AML **command job** (versioned dataset input, named
environment from `environment.yml`, CPU cluster, MLflow logging), wrap it with a
small `azure-ai-ml` v2 `command(...)` call — the canonical pattern shown in the
[AML SDK v2 docs](https://learn.microsoft.com/azure/machine-learning/how-to-train-models).
Run it on the cluster:

```bash
az ml job create --file job.yml --web
```

## 4. Promote a model to Production

Training auto-promotes the champion to the `Production` alias in the registry
(see `credit_risk.models.registry`). On Azure this is the AML model registry.
Approval can be gated behind an Azure Pipeline / GitHub Actions environment
with a human reviewer before the alias flip — a common model-risk control.

## 5. Deploy to a managed online endpoint

```bash
az ml online-endpoint create --name credit-risk-pd -g rg-credit-risk -w ml-credit-risk
az ml online-deployment create --endpoint credit-risk-pd --name blue \
  --model "credit_risk_pd_model@Production" \
  --code-path ./src --deployment-script deployment/azure/score_init.py \
  --environment azureml://registries/.../environments/credit-risk-py311/versions/1 \
  --instance-type Standard_DS2_v2 --instance-count 2
```

The endpoint serves the same FastAPI app (`credit_risk.serving.app`). Use
**blue/green** to shift traffic after smoke tests; roll back instantly by
re-pointing the alias.

## 6. Monitoring in Azure

- **Drift / performance**: a timer-triggered Azure Function (or ADF) calls
  `monitoring.drift.evaluate` / `monitoring.performance.evaluate` on a rolling
  batch and writes results to Application Insights. Breaches raise alerts that
  can trigger the retrain ADF pipeline.
- **Audit**: the FastAPI app streams `audit.jsonl` to **Log Analytics** via the
  stdout/ContainerApps log driver; retention + query via Kusto.

## 7. Business layer

Deploy the Streamlit dashboard to **Azure Container Apps** (Entra-authenticated)
so business users see portfolio KPIs and the threshold simulator without
touching the model directly.

## What is intentionally Azure-agnostic

All model logic, feature engineering, monitoring, audit and business-impact code
is **pure Python** and runs anywhere. Only the thin `azureml_train.py` adapter
(+ `environment.yml`) is Azure-specific. This keeps the system portable and the
Azure surface auditable.
