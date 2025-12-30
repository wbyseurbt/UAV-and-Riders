@echo off
echo ==========================================
echo       RL Environment Setup (Windows)
echo ==========================================

REM 1. 检查 Conda 是否安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b
)

REM 2. 创建 Conda 环境 (Python 3.12)
echo Creating Conda environment 'RL' with Python 3.12...
call conda create -n RL python=3.12 -y

REM 3. 激活环境
echo Activating environment...
call conda activate RL

REM 4. 安装 PyTorch
REM (PyTorch CUDA 12.4 版本兼容 CUDA 12.6 驱动)
echo Installing PyTorch (CUDA 12.x support)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM 5. 安装 RL 相关依赖
echo Installing Ray, PettingZoo, and other libs...
pip install "ray[rllib]" pettingzoo gymnasium matplotlib numpy pandas pygame

REM 6. 检查配置文件是否存在 (仅提示)
if not exist env_config.py (
    echo.
    echo [WARNING] 'env_config.py' not found in current directory!
    echo Please make sure you have copied your config file here.
    echo.
)

echo ==========================================
echo          Setup Complete!
echo ==========================================
echo To run the training:
echo    conda activate RL
echo    python train_mappo.py
echo.
echo To run the visualization:
echo    conda activate RL
echo    python run.py
echo ==========================================
pause