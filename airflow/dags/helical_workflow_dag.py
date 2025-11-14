from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


default_args = {
    "owner": "prateek",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="helical_model_workflow",
    description="Run a Helical model container on mounted data",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@once",
    catchup=False,
    tags=["helical", "ml", "docker"],
) as dag:

    start = EmptyOperator(task_id="start")

    run_helical_model = DockerOperator(
        task_id="run_helical_model",
        image="helical-runner:latest",
        api_version="auto",
        auto_remove=True,
        command="python run_helical.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(
                source="/Users/prateekkesarwani/Desktop/helical-airflow-challenge/data",
                target="/data",
                type="bind",
                read_only=True,
            )
        ],
        mount_tmp_dir=False,
        environment={"DATA_DIR": "/data"},
    )

    end = EmptyOperator(task_id="end")

    start >> run_helical_model >> end
