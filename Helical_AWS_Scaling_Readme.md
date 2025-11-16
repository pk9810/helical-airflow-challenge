# Scaling Guide for Helical Airflow Workflow on AWS

This document explains how to scale the local **Airflow + Docker +
helical-runner + Prometheus/Grafana** workflow into a production-grade,
secure, autoscaling system on **AWS**.

------------------------------------------------------------------------

## 1. High-Level Architecture Mapping

  Local Component            AWS Equivalent (Scalable)
  -------------------------- ---------------------------------
  Docker Compose + Airflow   Amazon EKS (Kubernetes)
  Local Datasets             Amazon S3
  Local DAG Folder           Amazon EFS or Git-Sync
  Local PostgreSQL           Amazon RDS PostgreSQL
  Local Redis                Amazon ElastiCache Redis
  Local Prometheus           Amazon Managed Prometheus (AMP)
  Local Grafana              Amazon Managed Grafana (AMG)
  Local Logs                 CloudWatch Logs + S3

This gives you: - Multi-AZ redundancy\
- Autoscaling\
- Managed databases\
- Durable storage\
- Encrypted and audited infrastructure

------------------------------------------------------------------------

## 2. Scaling Airflow & Orchestration on AWS

### 2.1 Airflow on Amazon EKS

Instead of DockerOperator → local Docker, use **KubernetesPodOperator**:

-   Airflow webserver, scheduler, workers run as EKS deployments
-   Celery executor with autoscaling workers
-   helical-runner jobs become isolated pods
-   EKS cluster-autoscaler automatically adjusts node count

### Benefits

-   Better isolation per run\
-   Scales to thousands of model executions\
-   No dependency on local docker.sock

------------------------------------------------------------------------

## 3. Data Persistence with S3 & EFS

### 3.1 S3 for all workflow data

Store: - Inputs (`.h5ad` datasets) - Outputs (embeddings, metrics) -
Long‑term logs

Enable: - **S3 Versioning**\
- **Lifecycle policies** → Glacier\
- **Server-side encryption with KMS**

### 3.2 EFS (optional)

Use EFS if: - You want shared storage for DAG files\
- Workers must share temporary datasets

------------------------------------------------------------------------

## 4. Reliable Metadata & Queue Layer

### 4.1 RDS PostgreSQL (Airflow metadata)

-   Multi‑AZ failover\
-   Automated backups\
-   Point‑in‑time restore\
-   KMS encryption

### 4.2 ElastiCache Redis (Celery backend)

-   Managed in‑memory queue\
-   Private subnet only\
-   Auth tokens stored in Secrets Manager

------------------------------------------------------------------------

## 5. Observability with AMP + AMG

### 5.1 Metrics Flow

Airflow → StatsD Exporter → AMP → Grafana dashboards

You get: - DAG success/failure rates\
- Task duration distributions\
- Worker autoscaling trends\
- helical-runner performance

### 5.2 Benefits

-   Fully managed, highly available
-   No need to maintain Prometheus servers

------------------------------------------------------------------------

## 6. Security, Encryption & Key Rotation

### 6.1 IRSA (IAM Roles for Service Accounts)

Each pod gets its own IAM role: - AirflowRole → S3 log access\
- HelicalRunnerRole → dataset bucket access

No node credentials exposed.

### 6.2 AWS KMS

Encrypt: - S3 buckets\
- RDS\
- EFS\
- Secrets Manager

Turn on **automatic key rotation**.

### 6.3 Secrets Manager

Secure storage for: - DB password\
- Redis token\
- API keys

Credentials can rotate automatically.

### 6.4 Network Security

-   All pods, RDS, Redis in private subnets\
-   Only ALB is public\
-   Restrictive Security Groups\
-   VPC Endpoints to avoid public traffic

------------------------------------------------------------------------

## 7. Logging, Audit & Retention

### 7.1 CloudWatch Logs

Every Airflow component logs to CloudWatch: - Searchable\
- Set retention (7--180 days)

### 7.2 Long-Term Archival on S3

Airflow/Helical logs stored as:

    s3://helical-workflows-logs/<dag>/<task>/<run_id>/

Lifecycle → Glacier after 60/90 days.

### 7.3 Audit Trails

-   CloudTrail → all API events\
-   KMS logs → key usage\
-   IAM Access Analyzer → access findings

------------------------------------------------------------------------

**Author:** Prateek Kesarwani\
**Project:** Helical Airflow Challenge (Production Scaling Guide)
