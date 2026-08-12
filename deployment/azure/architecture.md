# Azure Reference Architecture

This document describes how the repo maps onto Azure services for a **controlled,
auditable** production deployment. No Azure subscription is required to run the
repo locally — this is the target-state blueprint.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        ADS[(Azure Data Lake\nGen2 / Blob)] --> ADF[Azure Data Factory\norchestration]
    end
    ADF -->|curated parquet| ADS
    ADF -->|trigger| AML[Azure ML\nworkspace]

    subgraph AML["Azure ML (training & registry)"]
        AML --> CC[(CPU compute cluster)]
        CC -->|logs models metrics| MLF[(MLflow tracking)]
        MLF --> REG[(Model registry\nversioned + aliased)]
    end

    REG -->|Production version| EP[Managed online endpoint\nblue/green]
    EP --> WEB[Streamlit / FastAPI\nbusiness layer]

    subgraph Govern["Governance & monitoring"]
        EP -->|prediction log| LA[(Log Analytics)]
        EP -->|metrics| AM[Application Insights\n+ alerts]
        MON[(Azure Monitor\ndrift / performance)] --> AM
    end

    ID[(Microsoft Entra ID\nmanaged identities)] -.-> AML
    ID -.-> EP
    ID -.-> ADS
```

## Component mapping (repo → Azure)

| Repo component | Azure service | Notes |
|---|---|---|
| `data/` (raw/processed) | ADLS Gen2 / Blob | Versioned containers; ADF lands curated parquet. |
| `data/synthetic.py` + `ingestion.py` | ADF pipelines + AML data assets | In a real build, replace synthetic with a registered dataset. |
| `models/train.py` | AML **command job** (`azureml_train.py`) | Reproducible, compute-isolated, MLflow-logged. |
| MLflow registry | AML **model registry** | Versions + aliases (Production) = governed promotion. |
| `serving/app.py` | AML **managed online endpoint** | Blue/green, key vault auth, autoscale. |
| `monitoring/drift.py` + `performance.py` | Azure Monitor + Functions timer | PSI/KS + realised metrics → alerts. |
| `monitoring/audit.py` | Log Analytics / Event Hub | Append-only audit trail, retention policy. |
| `dashboard/app.py` | Streamlit on Container Apps | Reads model + metrics; Entra-authenticated. |

## Control & governance (why this is "well-controlled")

- **Identity**: every compute identity is a Managed Identity via Entra ID — no
  secrets in code. Storage/endpoint access is RBAC-scoped (Reader on data,
  ACR pull, endpoint contributor).
- **Reproducibility**: training jobs are pinned by git SHA + AML environment
  version + dataset version. Every model carries run ID, params, metrics.
- **Promotion gate**: only the registry alias `Production` is served. Promotion
  is an explicit, logged action (see `registry.promote_version_to_production`).
- **Monitoring**: drift (PSI/KS) and performance breaches raise Azure Monitor
  alerts that can trigger a retrain ADF pipeline or page an on-call.
- **Audit**: each scoring decision is written with model name + version, feature
  hash, PD, decision and reason codes — the evidence trail for model risk.
- **Data residency / privacy**: ADLS encryption-at-rest + customer-managed keys;
  PII column masking in ADF; network isolation via private endpoints.

## Cost posture

- CPU cluster autoscales to zero between jobs.
- Managed endpoint uses a small SKU with autoscale; blue/green only for promotion windows.
- MLflow on AML is serverless (no self-hosted tracking DB to run).

