from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "prateek",
    "retries": 1,                         # A lightweight retry for robustness
    "retry_delay": timedelta(minutes=1),   # Backoff before trying again
}


# -------------------------------------------------------------------
# DAG Definition
# -------------------------------------------------------------------
# This DAG:
#   1. Starts (EmptyOperator)
#   2. Launches a Docker container ("helical-runner")
#   3. The container runs `run_helical.py` on the mounted dataset
#   4. Produces embeddings + metrics
#
# Notes:
#   • schedule="@once" → run only when manually triggered
#   • catchup=False → avoid backfilling old runs
#   • tags help organize DAGs in the UI
# -------------------------------------------------------------------
with DAG(
    dag_id="helical_model_workflow",
    description="Run a Helical model container on mounted data",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),   # Airflow requires a start_date in the past
    schedule="@once",                  # Run only once unless manually triggered
    catchup=False,                     # Don’t try to run missing historical dates
    tags=["helical", "ml", "docker"],
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------------------------------------------------------------
    # DockerOperator: Launch the Helical model container
    #
    # This container runs:
    #     python run_helical.py
    #
    # And it receives:
    #   - Mounted dataset directory
    #   - Environment variable DATA_DIR=/data
    #
    # The container produces:
    #   - printed logs (visible in Airflow)
    #   - Prometheus metrics written under /tmp/metrics
    #
    # ---------------------------------------------------------------
    run_helical_model = DockerOperator(
        task_id="run_helical_model",
        image="helical-runner:latest",            # Built from your Dockerfile
        api_version="auto",
        auto_remove=True,                        # Delete container after run
        command="python run_helical.py",         # Override CMD if needed
        docker_url="unix://var/run/docker.sock", # Required for DockerOperator
        network_mode="bridge",                   # Default Docker network
        mounts=[
            Mount(
                source="/Users/prateekkesarwani/Desktop/helical-airflow-challenge/data",
                target="/data",                  # Container path
                type="bind",
                read_only=True,
            )
        ],
        mount_tmp_dir=False,                    # Avoid ephemeral temp folder
        environment={"DATA_DIR": "/data"},      # Passed to Python script
    )

    end = EmptyOperator(task_id="end")

    # ---------------------------------------------------------------
    # DAG dependencies:  start → container → end
    # ---------------------------------------------------------------
    start >> run_helical_model >> end
