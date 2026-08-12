"""Submit the training job to Azure ML (SDK v2).

This file is illustrative — it requires ``azure-ai-ml`` and an authenticated
workspace. It is NOT run by CI. It shows how the same ``azureml_train.py`` is
packaged as a governed, reproducible AML command job with a versioned dataset,
a compute cluster, named environment, and MLflow logging.

    pip install azure-ai-ml azure-identity
    python deployment/azure/submit_job.py
"""
from __future__ import annotations

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

SUBSCRIPTION = "<subscription-id>"
RESOURCE_GROUP = "<resource-group>"
WORKSPACE = "<workspace-name>"
COMPUTE = "cpu-cluster"
DATA_ASSET = "credit_applicants:1"  # registered tabular dataset in the workspace


def build_environment(client: MLClient) -> str:
    env = Environment(
        name="credit-risk-py311",
        description="Python 3.11 env for controlled credit-risk training",
        conda_file="deployment/azure/environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    client.environments.create_or_update(env)
    return f"{env.name}@latest"


def submit() -> None:
    client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    env_name = build_environment(client)

    job = command(
        code="./",  # upload the whole repo as the job context
        command=(
            "python deployment/azure/azureml_train.py "
            "--inputs.data ${{inputs.data}}"
        ),
        inputs={"data": Input(type="uri_folder", path=DATA_ASSET)},
        environment=env_name,
        compute=COMPUTE,
        experiment_name="credit_risk_pd",
        display_name="credit-risk-pd-train",
    )
    returned = client.jobs.create_or_update(job)
    print(f"Submitted job: {returned.name}")
    print(f"Studio: {returned.studio_url}")


if __name__ == "__main__":  # pragma: no cover
    submit()
