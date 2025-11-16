FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    pkg-config \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY run_helical.py .

ENV DATA_DIR=/data

CMD ["python", "run_helical.py"]
