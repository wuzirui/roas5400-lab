# ROAS5400 Student Bundle

This bundle contains:
- : student starter code (tests + demos + unimplemented )
- : assignment writeup and compiled 
- : train/val/test data

# Environment Setup

This project uses **Conda + pip** for environment management.

## 1. Prerequisites

- Conda (Miniconda or Anaconda)
- Python 3.10+
- A working `pip` inside your Conda environment

## 2. Create and activate a Conda environment

```bash
conda create -n roas5400 python=3.10 -y
conda activate roas5400
```

## 3. Install project dependencies

From this directory (`roas5400-lab/`), run:

```bash
pip install -e .
```

This installs the dependencies declared in `pyproject.toml` (including PyTorch, torchvision, OpenCV, NumPy, SciPy, and testing utilities).

## 4. Run tests

```bash
pytest
```

## Notes

- If your machine is Apple Silicon or Linux/Windows, pip will resolve platform-appropriate wheels.
- If PyTorch installation fails on your network, install PyTorch first from the official index, then run `pip install -e .` again.

