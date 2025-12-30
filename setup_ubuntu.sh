#!/bin/bash

echo "=========================================="
echo "      RL Environment Setup (Ubuntu)"
echo "=========================================="

# 1. 检查 Conda
if ! command -v conda &> /dev/null; then
    echo "Error: Conda could not be found. Please install Miniconda or Anaconda."
    exit 1
fi

# 初始化 Conda Hook
eval "$(conda shell.bash hook)"

# 2. 创建环境
echo "Creating Conda environment 'RL' with Python 3.12..."
conda create -n RL python=3.12 -y

# 3. 激活环境
echo "Activating environment 'RL'..."
conda activate RL

# 4. 安装 PyTorch
# (PyTorch CUDA 12.4 版本兼容 CUDA 12.6 驱动)
echo "Installing PyTorch (CUDA 12.x support)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. 安装依赖
echo "Installing Ray, PettingZoo, and other libs..."
pip install "ray[rllib]" pettingzoo gymnasium matplotlib numpy pandas pygame

# 6. 检查配置文件
if [ ! -f env_config.py ]; then
    echo ""
    echo "[WARNING] 'env_config.py' not found in current directory!"
    echo "Please ensure your config file is present before running scripts."
    echo ""
fi

echo "=========================================="
echo "          Setup Complete!"
echo "=========================================="
echo "To run the training:"
echo "   conda activate RL"
echo "   python train_mappo.py"
echo ""
echo "To run the visualization:"
echo "   conda activate RL"
echo "   python run.py"
echo "=========================================="