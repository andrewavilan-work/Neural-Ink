#!/bin/bash
# =============================================================================
# Neural-Ink V5 — AMD MI300X Setup Script
# AMD Developer Cloud | ROCm 6.2 | Ubuntu 22.04
# =============================================================================
# Usage:
#   chmod +x setup_mi300x.sh && ./setup_mi300x.sh
# =============================================================================

set -e  # Exit if any command fails

echo "=============================================="
echo "  Neural-Ink V5 — AMD MI300X Environment Setup"
echo "=============================================="

# --- 1. AMD ENVIRONMENT VARIABLES ---
export HSA_OVERRIDE_GFX_VERSION=9.4.2    # MI300X GFX ID
export PYTORCH_ROCM_ARCH=gfx942          # MI300X Architecture for compilation
export HIP_VISIBLE_DEVICES=0             # Use first GPU
export ROCM_PATH=/opt/rocm               # Standard path on AMD Developer Cloud
export PATH=$ROCM_PATH/bin:$PATH

echo "[1/6] AMD Environment variables configured"
echo "  HSA_OVERRIDE_GFX_VERSION = $HSA_OVERRIDE_GFX_VERSION"
echo "  PYTORCH_ROCM_ARCH        = $PYTORCH_ROCM_ARCH"

# --- 2. VERIFY ROCm ---
echo "[2/6] Verifying ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showmeminfo vram | grep -E "GPU|VRAM"
else
    echo "[WARN] rocm-smi not found — please verify ROCm installation"
fi

# --- 3. INSTALL PYTORCH ROCm ---
echo "[3/6] Installing PyTorch with ROCm 6.2 support..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2 \
    --quiet

# Verify that PyTorch detects the AMD GPU
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM: {mem:.0f} GB')
"

# --- 4. PYTHON DEPENDENCIES ---
echo "[4/6] Installing project dependencies..."
pip install \
    onnx \
    onnxruntime \
    onnxconverter-common \
    opencv-python-headless \
    pillow \
    numpy \
    --quiet

echo "  Dependencies installed successfully"

# --- 5. VERIFY DATASET ---
echo "[5/6] Verifying dataset structure..."
DATASET_DIR="NeuralDataset_V5"
REQUIRED=("rgb" "depth" "bg_rgb" "bg_depth" "motion")
ALL_OK=true

for folder in "${REQUIRED[@]}"; do
    if [ -d "$DATASET_DIR/$folder" ]; then
        count=$(ls "$DATASET_DIR/$folder" | wc -l)
        echo "  [OK] $DATASET_DIR/$folder — $count files"
    else
        echo "  [ERROR] Missing folder: $DATASET_DIR/$folder"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "[ERROR] Dataset incomplete. Please upload the dataset from your local machine:"
    echo "  scp -r ./NeuralDataset_V5 user@mi300x-instance:~/neural-ink/"
    echo ""
fi

# --- 6. VERIFY PRE-TRAINED CHECKPOINT ---
echo "[6/6] Checking for transfer learning checkpoint..."
if [ -f "neural_ink_v5_gold.pth" ]; then
    size=$(du -sh neural_ink_v5_gold.pth | cut -f1)
    echo "  [OK] neural_ink_v5_gold.pth found ($size)"
    echo "  Will use as base for Transfer Learning."
else
    echo "  [WARN] neural_ink_v5_gold.pth not found"
    echo "  Training will start from scratch (requires more time)."
fi

# --- PERSIST VARIABLES IN .bashrc ---
cat >> ~/.bashrc << 'EOF'
# Neural-Ink AMD MI300X
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export PYTORCH_ROCM_ARCH=gfx942
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
EOF

echo ""
echo "=============================================="
echo "  Setup Complete. Start training with:"
echo ""
echo "  nohup python train_mi300x_hackathon.py > training.log 2>&1 &"
echo "  tail -f training.log"
echo ""
echo "  Monitor GPU utilization:"
echo "  watch -n 2 rocm-smi"
echo "=============================================="
