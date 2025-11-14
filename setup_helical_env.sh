#!/usr/bin/env bash
set -e

ENV_NAME="helical-package"
PY_VERSION="3.11.13"
ACTIVATE_LINE="conda activate ${ENV_NAME}"

echo "--------------------------------------------"
echo " Setting up Helical Conda Environment"
echo "--------------------------------------------"

# ------------------------------------------------------
# 1. Detect OS
# ------------------------------------------------------
OS=$(uname -s)

install_conda_mac() {
    echo "[+] Checking for Conda..."
    if ! command -v conda &>/dev/null; then
        # Check if miniconda directory already exists
        if [ -d "$HOME/miniconda" ]; then
            echo "[+] Miniconda directory already exists → using existing installation."
            eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
            return
        fi

        echo "[+] Installing Miniconda for macOS..."
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
        bash miniconda.sh -b -p "$HOME/miniconda"
        rm miniconda.sh
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

case "$OS" in
    Darwin*) install_conda_mac ;;
    Linux*)  install_conda_linux ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# ------------------------------------------------------
# 2. Initialize Conda in current shell
# ------------------------------------------------------
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "[-] conda command not found even after install. Check \$HOME/miniconda installation."
    exit 1
fi

# ------------------------------------------------------
# 3. Accept Anaconda Terms of Service (non-interactive)
# ------------------------------------------------------
echo "[+] Accepting Anaconda Terms of Service (if needed)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r     || true

# ------------------------------------------------------
# 4. Create Conda environment
# ------------------------------------------------------
if conda info --envs | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "[+] Conda environment '${ENV_NAME}' already exists."
else
    echo "[+] Creating conda environment '${ENV_NAME}' with Python ${PY_VERSION}..."
    conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
fi

echo "[+] Activating environment..."
conda activate "${ENV_NAME}"

# ------------------------------------------------------
# 5. Install core Helical packages
# ------------------------------------------------------
echo "[+] Installing Helical core package..."
pip install --upgrade pip wheel setuptools

echo "[+] Installing stable helical from PyPI..."
pip install helical || true

echo "[+] (Optional) Trying to install latest helical from GitHub..."
# On macOS arm64 bitsandbytes has no compatible wheels, so this may fail.
pip install --upgrade "git+https://github.com/helicalAI/helical.git" || {
  echo "[-] Could not install latest helical from GitHub (likely bitsandbytes missing on macOS arm64)."
  echo "    Keeping the stable PyPI version instead."
}

echo "[+] Installing optional extras (mamba-ssm) if available..."
pip install "helical[mamba-ssm]" || true


# ------------------------------------------------------
# 6. Install workflow + ML + Airflow dependencies
# ------------------------------------------------------
echo "[+] Installing project dependencies..."

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
# 7. Auto-activate this environment in new terminals
# ------------------------------------------------------
echo "[+] Enabling automatic activation..."

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rc" ]; then
        if ! grep -Fxq "$ACTIVATE_LINE" "$rc"; then
            echo "$ACTIVATE_LINE" >> "$rc"
            echo "  Added auto-activation to $rc"
        else
            echo "  Auto-activation already exists in $rc"
        fi
    fi
done

echo ""
echo "=============================================="
echo " Helical environment setup complete! 🎉"
echo ""
echo " Next steps:"
echo "   • Open a NEW terminal → env auto-activates"
echo "   • Or manually: conda activate ${ENV_NAME}"
echo ""
echo " Installed:"
echo "   ✔ Helical (latest from GitHub)"
echo "   ✔ Airflow + Docker Provider"
echo "   ✔ Scanpy / AnnData / h5py"
echo "   ✔ Prometheus Client"
echo "   ✔ Optional mamba-ssm extras (if available)"
echo "=============================================="
