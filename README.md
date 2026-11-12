# Dynamic Model Growth for LLMs (CSCE-4110-Transformer)

PyTorch implementation of dynamic model growth for cost-efficient Large Language Model training. Explores progressive width and depth expansion to minimize computational cost (TFLOPs) without sacrificing validation accuracy.

This repository contains the codebase for our project on cost-efficient Transformer training via dynamic architecture expansion.

## Team
* **Ifunayal:** Data Pipeline & Baselines
* **John Greig:** Model Growth Architecture
* **Oneal Egwuatu:** Evaluation & Benchmarking 

## Repository Structure
* `model_growth.py`: Core PyTorch engine for progressive dimensionality growth (`expand_width` with `trainable` / `zero` init) and layer deepening (`expand_depth` with `identity`, `duplicate`, and `stack` methods).
* `train.py`: End-to-end training & evaluation pipeline (WikiText-2 / AG News tokenization, small `m_0` & large `M_0` baselines, dynamic growth triggers, TFLOPs tracking, and validation/test evaluation).
* `run_experiments.py`: Automated benchmark runner comparing static baselines (`m_0`, `M_0`), deepening methods (`identity`, `duplicate`, `stack`), and growth timings (`25%`, `50%`, `75%`).
* `data/`: (Ignored) Local directory for cached tokenized English corpora.
* `logs/`: Run logs and JSON evaluation summaries.

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

5. Run training & benchmark experiments:
   ```bash
   # Default dynamic growth run (grows at 50% of steps using identity deepening)
   python train.py

   # Static baselines from scratch (m_0 small and M_0 large)
   python train.py --mode small
   python train.py --mode large

   # Full benchmark suite (baselines + deepening methods + growth timings)
   python run_experiments.py --total-steps 300
   ```
