# Dynamic Model Growth for LLMs (CSCE-4110-Transformer)

PyTorch implementation of dynamic model growth for cost-efficient Large Language Model training. Explores progressive width and depth expansion to minimize computational cost (TFLOPs) without sacrificing validation accuracy.

This repository contains the codebase for our project on cost-efficient Transformer training via dynamic architecture expansion.

## Team
* **Person 1:** Data Pipeline & Baselines
* **John Greig:** Model Growth Architecture
* **Person 3:** Evaluation & Benchmarking 

## Repository Structure
* `model_growth.py`: Core PyTorch engine for progressive dimensionality growth and redundant layer stacking.
* `train.py`: (WIP) Integration loop handling the expansion trigger timings.
* `data/`: (Ignored) Local directory for the English text corpus.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Johngg2642/CSCE-4110-Transformer.git
   cd CSCE-4110-Transformer
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install requirements: `pip install -r requirements.txt`
4. Run architecture tests: `python model_growth.py`

Expected output ends with:
```
SUCCESS: 256-dim/4-layer model expanded to 512-dim/8-layer model with no shape errors.
```
