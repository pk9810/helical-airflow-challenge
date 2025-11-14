🚀 Helical Airflow Challenge
Containerized Workflow Orchestration with Airflow, Docker, Prometheus, Grafana & cAdvisor

This project implements a containerized workflow orchestration system designed for the Helical Technical Challenge.
It features:

Apache Airflow running with the Celery Executor

Dockerized Helical model execution inside an Airflow DAG

Structured data mounting (/opt/data)

Prometheus-based observability pipeline

StatsD → statsd-exporter → Prometheus → Grafana

Container-level resource monitoring via cAdvisor

Fully repeatable local environment setup script (Conda + Helical installation)

📁 Project Structure
helical-airflow-challenge/
│
├── airflow/
│   ├── dags/
│   │   └── helical_model_workflow.py
│   ├── logs/
│   └── config/
│
├── data/                       # Mounted into the model container
│   └── sample.h5ad
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│
├── setup_helical_env.sh        # One-click local setup & Helical installation
├── docker-compose.yml
└── README.md

⚙️ 1. Environment Setup (One-Click Script)

Run this script on any machine for the first-time setup:

chmod +x setup_helical_env.sh
./setup_helical_env.sh

The script automatically:

✔ Detects OS
✔ Installs Miniconda (if missing)
✔ Creates Conda env helical-package
✔ Installs Helical (PyPI + GitHub latest)
✔ Ensures Python 3.11.13
✔ Installs optional extensions
✔ Automatically activates the environment in new terminals

🐳 2. Start Full Docker Orchestration

Ensure Docker is installed.

Start everything:
docker compose up -d --build

Stop everything (including volumes):
docker compose down -v

📦 3. Airflow Architecture

This setup includes:

Component	Purpose
Airflow Webserver	UI & DAG management
Airflow Scheduler	Orchestrates DAG tasks
Airflow Worker (Celery)	Executes tasks
Postgres	Airflow metadata DB
Redis	Celery broker
Docker provider	Allows Airflow to execute Helical container

All tasks share a mounted folder:

host: ./data  →  container: /opt/data

🧬 4. The Helical Model DAG

A sample DAG is included at:

airflow/dags/helical_model_workflow.py


It performs:

start – empty task

run_helical_model – runs a Docker container

end – empty task

The Docker task mounts:

/opt/data → /opt/data  (inside container)


You can swap this with any Helical model:

image="helicalai/helical:latest"
command="python3 examples/run_model.py --input /opt/data/sample.h5ad"

📊 5. Observability Pipeline

This project includes full metrics stack:

Airflow → StatsD → statsd-exporter → Prometheus → Grafana → Dashboards
                           ↑
                cAdvisor → Prometheus

Prometheus Targets
Target	Purpose
statsd-exporter:9102	Airflow metrics
cadvisor:8080	Container CPU / Memory metrics
prometheus:9090	Self-metrics
Access URLs
Service	URL
Airflow UI	http://localhost:8080

Prometheus	http://localhost:9090

Grafana	http://localhost:3000

cAdvisor	http://localhost:8081
📈 6. Grafana Dashboards

Grafana automatically loads "Heical – Airflow & Containers" dashboard.

Panels available:

Airflow Scheduler Heartbeat

DAG Runs Count / Success / Duration

Task Duration (p95)

Worker CPU / Memory (via cAdvisor)

Per-container resource usage

Login credentials:

Username: admin
Password: admin

📑 7. Metrics Configuration
StatsD in Airflow
AIRFLOW__METRICS__STATSD_ENABLED: "True"
AIRFLOW__METRICS__STATSD_HOST: "statsd-exporter"
AIRFLOW__METRICS__STATSD_PORT: "9125"
AIRFLOW__METRICS__STATSD_PREFIX: "airflow"
AIRFLOW__METRICS__STATSD_ALLOW_LIST: "*"

Prometheus scrapes statsd-exporter:
- job_name: "airflow"
  static_configs:
    - targets: ["statsd-exporter:9102"]

🐳 8. Docker Compose Overview

Key services included:

Service	Description
airflow-webserver	Main UI
airflow-scheduler	DAG scheduling
airflow-worker	Task workers
statsd-exporter	Metric bridge
prometheus	Metrics storage
grafana	Visualization
cadvisor	Container monitoring
redis	Celery broker
postgres	Metadata DB
🚦 9. Triggering a DAG

Visit:

➡ http://localhost:8080

Enable DAG → Click "Play" → Trigger DAG.

Prometheus & Grafana will show metrics once the DAG runs.

🧪 10. Verification Commands
Check exporter metrics:
curl http://localhost:9102/metrics | grep airflow

Check cAdvisor:
curl http://localhost:8081/metrics | head

Check Prometheus UI:
open http://localhost:9090/targets

Check Prometheus queries:

statsd_airflow_scheduler_heartbeat

statsd_airflow_dagrun_success_total

container_cpu_usage_seconds_total

🛠️ 11. Common Issues & Fixes
❌ No Airflow metrics in Prometheus

✔ Ensure Airflow sends StatsD → port 9125, not 8125.

❌ Grafana shows no data

✔ Trigger a DAG so metrics begin flowing.

❌ Docker provider missing

✔ Ensure _PIP_ADDITIONAL_REQUIREMENTS includes:

apache-airflow-providers-docker
statsd

❌ Permission denied reading data files

✔ Ensure ./data is readable by Docker.

📌 12. Future Improvements

Add MLflow tracking for model metadata

Run Airflow with LocalKubernetesExecutor

Add Loki + Promtail for central log aggregation

Add model validation tasks in DAG