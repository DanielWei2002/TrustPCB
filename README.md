# TrustPCB

TrustPCB is a research project for reliability-aware PCB defect detection using deep learning and automated optical inspection.

## Project Aim

The project investigates a model-agnostic reliability framework for lightweight PCB defect detection by combining:

- calibrated confidence
- class consistency
- localisation stability
- composite prediction-risk scoring
- selective human-review referral

## Dataset

The project uses the DsPCBSD+ PCB surface-defect dataset.

Raw and processed datasets are not stored in this GitHub repository.

## Project Structure

- `data/` - dataset files and data splits
- `notebooks/` - Jupyter notebooks for analysis and experiments
- `src/` - reusable TrustPCB source code
- `configs/` - model and experiment configuration files
- `scripts/` - training, evaluation and utility scripts
- `experiments/` - experiment definitions and records
- `runs/` - model training outputs
- `outputs/` - generated figures, tables and predictions
- `docs/` - project and research documentation

## Environment

Main development environment:

- Python 3.11
- PyTorch
- CUDA
- Ultralytics YOLO
- DICC HPC

## Status

Research and development in progress.
