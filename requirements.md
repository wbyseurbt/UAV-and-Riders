# Project Dependencies

This project requires the following Python packages. You can install them using `pip`:

```bash
pip install gymnasium>=0.29.0 numpy>=1.24.0 "stable-baselines3[extra]>=2.2.1" matplotlib>=3.7.0 tensorboard>=2.14.0 torch>=2.0.0
```

## Core Libraries

*   **gymnasium** (>=0.29.0): The standard API for reinforcement learning environments.
*   **numpy** (>=1.24.0): Fundamental package for scientific computing with Python.
*   **stable-baselines3[extra]** (>=2.2.1): Reliable implementations of reinforcement learning algorithms (PPO, A2C, etc.).
*   **matplotlib** (>=3.7.0): Library for creating static, animated, and interactive visualizations.
*   **tensorboard** (>=2.14.0): Visualization toolkit for machine learning experimentation.
*   **torch** (>=2.0.0): The PyTorch machine learning framework.

## Installation Note

If you encounter issues with `numpy` 2.x compatibility (e.g. `AttributeError: _ARRAY_API not found`), consider pinning numpy to version `<2`:

```bash
pip install "numpy<2"
```
