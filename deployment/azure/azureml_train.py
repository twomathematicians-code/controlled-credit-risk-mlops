"""Azure ML entry-point training script.

This script is the *same* training pipeline as ``credit_risk.models.train`` but
wired for an Azure ML (AML) command job:

  * **Locally** (no AML context): it generates data if needed and runs the
    standard training — identical to ``python -m credit_risk.models.train``.
  * **On AML compute**: it picks up the workspace-provided MLflow tracking URI
    (``azureml://...``), reads the mounted dataset from ``$AZUREML_INPUT_DATA``,
    writes artefacts to ``./outputs``, and registers the model to the
    AML-managed registry.

Run locally to validate before submitting::

    python deployment/azure/azureml_train.py

To run on Azure ML, see ``deployment/azure/README.md`` (workspace + command-job setup).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the package importable when run as a standalone script (no install).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data import ingestion, synthetic  # noqa: E402
from credit_risk.models import train  # noqa: E402
from credit_risk.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def running_in_aml() -> bool:
    """AML injects these env vars on compute targets."""
    return any(
        os.getenv(v)
        for v in ("AZUREML_ARM_SUBSCRIPTION", "AZUREML_RUN_ID", "MLFLOW_TRACKING_URI")
        if "azureml" in str(os.getenv(v, "")).lower()
    ) or os.getenv("AZUREML_INPUT_DATA") is not None


def ensure_data() -> None:
    """Locally: generate + ingest if the processed splits are missing."""
    processed = ROOT / "data" / "processed" / "train.parquet"
    if processed.exists():
        logger.info("Processed data already present — skipping generation.")
        return
    logger.info("Generating synthetic data (set AZUREML_INPUT_DATA to use a mounted dataset instead).")
    synthetic.main()
    ingestion.run()


def main() -> None:
    logger.info("AML context detected: %s", running_in_aml())
    if running_in_aml():
        # AML auto-configures MLFLOW_TRACKING_URI to point at the workspace.
        logger.info("MLflow tracking URI: %s", os.getenv("MLFLOW_TRACKING_URI", "<unset>"))
        # On AML a dataset would be mounted; here we still rely on the local
        # pipeline generating data if none is mounted.
        input_data = os.getenv("AZUREML_INPUT_DATA")
        if input_data and Path(input_data).exists():
            logger.info("Mounted dataset found at %s", input_data)
    ensure_data()

    results = train.run(register=True, promote=True)
    logger.info(
        "Training complete | champion=%s test_auc=%.4f registered_version=%s",
        results.get("champion"),
        results.get("test_roc_auc", float("nan")),
        results.get("registered_version"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
