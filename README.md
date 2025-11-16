# Helical Airflow Challenge -- Workflow Orchestration + Observability

This project implements a containerised machine-learning workflow using:

-   Apache Airflow for orchestration
-   Docker for isolated model execution
-   Prometheus + Grafana for observability
-   A custom Helical Runner container that executes `run_helical.py`
-   Shared data volume for reading `.h5ad` datasets

Everything runs fully locally using docker-compose.

## 🚀 What This System Does

This workflow automates end-to-end execution of a Helical ML pipeline
using Airflow + Docker + Prometheus + Grafana.

### 1️⃣ User triggers a DAG in Airflow UI

Go to: <http://localhost:8080>

**Login credentials**

    Username: admin
    Password: admin

Then:

-   Open `helical_model_workflow`
-   Click ▶ Trigger DAG

### 2️⃣ Airflow schedules the task

Airflow internal flow:

    Webserver → Scheduler → Worker

Worker executes the DockerOperator task.

### 3️⃣ DockerOperator starts the Helical Runner container

The operator launches the container:

-   Image: `helical-runner:latest`
-   Mounts dataset folder:


    ./data → /data (inside container)

Runs:

    python run_helical.py

### 4️⃣ run_helical.py performs the ML pipeline

Inside the container:

-   Loads `.h5ad` dataset(s) from `/data`
-   If empty → loads Scanpy PBMC3K dataset
-   Shrinks dataset for fast CPU execution
-   Runs scGPT embeddings (Helical SDK)
-   Optionally trains a tiny classifier
-   Writes metrics → `/tmp/metrics`
-   Logs visible inside Airflow UI

### 5️⃣ Observability Metrics Flow

    Airflow
       ↓
    StatsD
       ↓
    StatsD Exporter
       ↓
    Prometheus
       ↓
    Grafana

Prometheus scrapes:

-   Airflow task duration & failures\
-   cAdvisor container CPU/Mem\
-   StatsD metrics\
-   Prometheus internal metrics

Grafana datasource:

    URL: http://prometheus:9090

------------------------------------------------------------------------

# ▶ How To Run Everything (Copy-Paste Friendly)

Follow these steps exactly in order.

------------------------------------------------------------------------

## ✅ 1️⃣ (Optional) Install local Conda environment

    chmod +x setup_helical_env.sh
    ./setup_helical_env.sh

## ✅ 2️⃣ Initialise Airflow database (first time only)

🔥 IMPORTANT --- MUST RUN BEFORE ANYTHING ELSE

    docker compose up airflow-init

This sets up:

-   Airflow metadata DB\
-   Admin user\
-   Initial migrations

Stop it (Ctrl + C) after completion.

------------------------------------------------------------------------

## ✅ 3️⃣ Start full system (Airflow + Prometheus + Grafana)

    docker compose up -d

### Services Available

  Service           URL
  ----------------- -------------------------------
  Airflow Web UI    http://localhost:8080
  
  Prometheus        http://localhost:9090
  
  Grafana           http://localhost:3000
  
  StatsD Exporter   http://localhost:9102/metrics
  
  cAdvisor          http://localhost:8081
  

------------------------------------------------------------------------

## ✅ 4️⃣ Build the Helical Runner container

    docker build -t helical-runner:latest ./helical-container

This image contains:

-   Helical SDK\
-   scGPT deps\
-   Scanpy + AnnData\
-   `run_helical.py`

------------------------------------------------------------------------

## ✅ 5️⃣ Add datasets (optional)

Place `.h5ad` files in:

    ./data/

These map to `/data` inside the container.

If none provided → auto-loads PBMC3K dataset.

------------------------------------------------------------------------

## ✅ 6️⃣ Trigger the Helical DAG in Airflow

1.  Visit: http://localhost:8080\
2.  Unpause: `helical_model_workflow`\
3.  Click "Trigger DAG"\
4.  View logs:\
    `run_helical_model → View Log`

------------------------------------------------------------------------

# 📊 Observability Setup

## Prometheus

Scrapes:

-   StatsD Exporter (`airflow_*`)
-   cAdvisor (container CPU/Mem)
-   Prometheus internal metrics

Config file:

    monitoring/prometheus.yml

## Grafana

Add datasource:

    URL: http://prometheus:9090

Dashboards can show:

-   DAG run durations\
-   Task success/failure counts\
-   Container CPU/memory usage\
-   Airflow scheduler uptime\
-   Worker activity & queue depth
