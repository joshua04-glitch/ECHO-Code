#!/bin/bash
# ============================================================================
# GraphEcho Replication - Environment Setup for Stanford Farmshare / Jupyter
# ============================================================================
# This script sets up a Miniconda environment with all required dependencies.
# Run this ONCE before starting any training or notebook work.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# After setup, activate with:
#   conda activate graphecho
# ============================================================================

set -e

# ---------- Detect where we are ----------
if [[ -d "$HOME" ]]; then
    INSTALL_DIR="$HOME/miniconda3"
else
    INSTALL_DIR="$(pwd)/miniconda3"
fi

# ---------- Install Miniconda if not present ----------
if ! command -v conda &> /dev/null && [ ! -f "$INSTALL_DIR/bin/conda" ]; then
    echo ">>> Installing Miniconda to $INSTALL_DIR ..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"
    rm /tmp/miniconda.sh
    echo ">>> Miniconda installed."
fi

# Make sure conda is on PATH
export PATH="$INSTALL_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"

# ---------- Create the environment ----------
ENV_NAME="graphecho"

if conda env list | grep -q "$ENV_NAME"; then
    echo ">>> Environment '$ENV_NAME' already exists. Updating..."
    conda activate "$ENV_NAME"
else
    echo ">>> Creating conda environment '$ENV_NAME' (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
    conda activate "$ENV_NAME"
fi

# ---------- Core dependencies ----------
echo ">>> Installing PyTorch (CUDA 11.8)..."
# Farmshare typically has CUDA 11.x; adjust the cudatoolkit version if needed.
# If you only have CPU, replace the line below with:
#   conda install pytorch torchvision cpuonly -c pytorch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ">>> Installing scientific stack..."
pip install \
    numpy \
    scipy \
    scikit-learn \
    scikit-image \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    Pillow \
    opencv-python-headless \
    nibabel \
    SimpleITK \
    einops \
    tensorboard

echo ">>> Installing graph / GNN dependencies..."
pip install \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster

echo ">>> Installing Jupyter kernel..."
pip install ipykernel ipywidgets
python -m ipykernel install --user --name "$ENV_NAME" --display-name "GraphEcho (PyTorch)"

echo ">>> Installing dataset utilities..."
pip install \
    kaggle \
    pydicom \
    h5py

# ---------- Verify ----------
echo ""
echo "============================================================"
echo "  Setup complete!  Activate with:  conda activate $ENV_NAME"
echo "============================================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
"
echo ""
echo "Next steps:"
echo "  1. Download the CardiacUDA dataset from Kaggle"
echo "  2. Place/unzip it under ./data/CardiacUDA/"
echo "  3. Open the Jupyter notebook:  graphecho_replication.ipynb"
echo "  4. Select the 'GraphEcho (PyTorch)' kernel"
echo ""
