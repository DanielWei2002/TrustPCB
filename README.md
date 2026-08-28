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

This project uses the **DsPCBSD+ PCB surface-defect dataset** introduced by Lv et al. (2024).

**Reference**  
Lv, S., Ouyang, B., Deng, Z., et al. (2024). *A dataset for deep learning based detection of printed circuit board surface defect*. Scientific Data, 11, 811.  
DOI: https://doi.org/10.1038/s41597-024-03656-8  
Figshare dataset: https://doi.org/10.6084/m9.figshare.24970329

The DsPCBSD+ dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.

Raw and processed dataset files are not stored in this GitHub repository. Some generated training visualisations, validation examples and model-output figures in this repository are derived from DsPCBSD+ for research and evaluation purposes and should be interpreted with attribution to the original dataset authors.

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
