#!/usr/bin/env bash
set -e

# ------------------------------------------------------
# Helical Conda Environment Setup Script
# ------------------------------------------------------
# This script installs miniconda (if missing), creates a
# dedicated conda environment for Helical model workflows,
# installs all required dependencies (Airflow, Scanpy, ML
# libs, Prometheus, Docker provider, etc.), and ensures
# the env auto-activates in new terminals.
#
# Designed to work cleanly on both macOS + Linux.
# ------------------------------------------------------

ENV_NAME="helical-package"
PY_VERSION="3.11.13"
ACTIVATE_LINE="conda activate ${ENV_NAME}"

echo "--------------------------------------------"
echo " Setting up Helical Conda Environment"
echo "--------------------------------------------"

# ------------------------------------------------------
# 1. Detect OS (macOS vs Linux)
# ------------------------------------------------------
# uname -s gives Darwin on macOS, Linux on Ubuntu/Debian/etc.
# We branch installation logic to make setup smooth on both.
OS=$(uname -s)

install_conda_mac() {
    echo "[+] Checking for Conda..."

    # If conda is not found, we install Miniconda.
    if ! command -v conda &>/dev/null; then

        # If the directory already exists, reuse it.
        if [ -d "$HOME/miniconda" ]; then
            echo "[+] Miniconda directory already exists → using existing installation."
            eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
            return
        fi

        echo "[+] Installing Miniconda for macOS..."
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
        bash miniconda.sh -b -p "$HOME/miniconda"
        rm miniconda.sh

        # Load conda into the current shell session
        eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
    else
        echo "[+] Conda already installed."
    fi
}

install_conda_linux() {
    echo "[+] Checking for Conda..."

    if ! command -v conda &>/dev/null; then

        if [ -d "$HOME/miniconda" ]; then
            echo "[+] Miniconda directory already exists → using existing installation."
            eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
            return
        fi

        echo "[+] Installing Miniconda for Linux..."
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
        bash miniconda.sh -b -p "$HOME/miniconda"
        rm miniconda.sh

        eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
    else
        echo "[+] Conda already installed."
    fi
}

echo "[+] OS detected: $OS"

# OS branching for installation
case "$OS" in
    Darwin*) install_conda_mac ;;
    Linux*)  install_conda_linux ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# ------------------------------------------------------
# 2. Initialize Conda for the current shell session
# ------------------------------------------------------
# This makes `conda activate` work immediately without restarting.
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "[-] conda command not found even after installation. Something is wrong."
    exit 1
fi

# ------------------------------------------------------
# 3. Accept Anaconda Terms (non-interactive mode)
# ------------------------------------------------------
# We suppress errors because older conda versions may not require/understand this step.
echo "[+] Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r     || true

# ------------------------------------------------------
# 4. Create the Conda Environment (if missing)
# ------------------------------------------------------
if conda info --envs | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "[+] Conda environment '${ENV_NAME}' already exists — skipping creation."
else
    echo "[+] Creating conda environment '${ENV_NAME}' with Python ${PY_VERSION}..."
    conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
fi

echo "[+] Activating environment..."
conda activate "${ENV_NAME}"

# ------------------------------------------------------
# 5. Install Helical packages
# ------------------------------------------------------
# We try both:
# - Stable release from PyPI
# - Latest development version from GitHub
# If GitHub install fails (common on macOS arm64 due to bitsandbytes),
# we gracefully fall back to PyPI.
echo "[+] Installing Helical core packages..."
pip install --upgrade pip wheel setuptools

echo "[+] Installing stable Helical from PyPI..."
pip install helical || true

echo "[+] Attempting to install latest Helical from GitHub..."
pip install --upgrade "git+https://github.com/helicalAI/helical.git" || {
  echo "[-] Latest GitHub build failed (likely bitsandbytes unavailable on macOS arm64)."
  echo "    Keeping stable PyPI version instead."
}

echo "[+] Installing optional extras (mamba-ssm) if supported..."
pip install "helical[mamba-ssm]" || true

# ------------------------------------------------------
# 6. Install workflow, ML, and Airflow dependencies
# ------------------------------------------------------
# These power:
# - Airflow DAG execution
# - AnnData processing
# - High-dimensional biology workflows
# - Prometheus metrics for observability
echo "[+] Installing workflow / ML / Airflow dependencies..."

pip install \
    "apache-airflow==2.10.2" \
    apache-airflow-providers-docker \
    scanpy \
    anndata \
    h5py \
    numpy \
    scipy \
    pandas \
    matplotlib \
    requests \
    tqdm \
    prometheus-client

# ------------------------------------------------------
# 7. Enable automatic activation in all new terminals
# ------------------------------------------------------
# Appends "conda activate <env>" to .bashrc/.zshrc if missing.
echo "[+] Enabling automatic activation..."

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rc" ]; then
        if ! grep -Fxq "$ACTIVATE_LINE" "$rc"; then
            echo "$ACTIVATE_LINE" >> "$rc"
            echo "  Added auto-activation to $rc"
        else
            echo "  Auto-activation line already present in $rc"
        fi
    fi
done

# ------------------------------------------------------
# Final message
# ------------------------------------------------------
echo ""
echo "=============================================="
echo " Helical environment setup complete! 🎉"
echo ""
echo " Next steps:"
echo "   • Open a NEW terminal → env auto-activates"
echo "   • Or manually run: conda activate ${ENV_NAME}"
echo ""
echo " Installed:"
echo "   ✔ Helical (PyPI + GitHub attempt)"
echo "   ✔ Airflow + Docker Provider"
echo "   ✔ Scanpy / AnnData / h5py"
echo "   ✔ Prometheus Client"
echo "   ✔ Optional mamba-ssm dependencies"
echo "=============================================="
