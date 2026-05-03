# LLM-FE Task 3: Improvisation — README & Difference Log

**Project:** LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers  
**Course:** Data Science  
**Submission Date:** May 3, 2026  
**Task:** Task 3 — Improvisation (builds on Task 2 Replication)

---

## What This Task Is About

This task improves upon the Task 2 replication of the LLM-FE framework. Instead of simply reproducing the paper's results, we made two deliberate, course-grounded improvisations to achieve better generalization and slightly improved accuracy.

---

## Improvisations Made

### 1. Hyperparameter Optimization — More Iterations (20 → 30)

**What changed:** The `global_max_sample_num` parameter in `main.py` was increased from 20 to 30.

**Why:** The LLM-FE framework uses evolutionary search across iterations. Based on the paper's ablation study (Figure 5), accuracy continues to improve as iterations increase. Giving the model a 50% larger search budget allows it to discover stronger feature combinations and avoid local optima.

**Result:** +0.0016 accuracy improvement on balance-scale (0.9548 → 0.9564).

---

### 2. Data Engineering — Multi-Dataset Evaluation

**What changed:** Added a second dataset (`breast-w`) to the evaluation pipeline, alongside the original `balance-scale`.

**Why:** Testing on a single dataset does not confirm generalization. The breast-w dataset (medical classification, 699 samples, 9 features) is a fundamentally different domain from balance-scale (physics-based, 625 samples, 4 features), providing meaningful cross-domain validation.

**Result:** 0.9667 accuracy on breast-w — only 0.003 below the paper's GPT-3.5 result, and significantly above the 0.956 baseline.

---

## Difference Log (Task 3 vs Task 2)

| File | Change Type | Description |
|---|---|---|
| `main.py` | Hyperparameter | Changed `global_max_sample_num` from `20` to `30` (Line 14) |
| `llmfe/sampler.py` | No change | Groq API integration remains identical to Task 2 |
| `comparison_chart.png` | New output | Bar chart comparing all methods across both datasets |
| `improvement_chart.png` | New output | Accuracy gain over baseline — Paper vs Task 3 |
| `iterations_effect.png` | New output | Effect of 20 vs 30 iterations on balance-scale |

**Only one line of code was changed.** All other files are identical to Task 2.

---

## How to Run

### Requirements

```
Python 3.11.9
xgboost==3.2.0
groq (API client)
scikit-learn
pandas
numpy
matplotlib
```

### Setup

```bash
# Clone the repository
git clone https://github.com/nikhilsab/LLMFE
cd LLMFE

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY=your_key_here
```

### Run Task 3 (30 iterations, both datasets)

```bash
# Run on balance-scale
python main.py --dataset balance-scale

# Run on breast-w
python main.py --dataset breast-w
```

> **Note:** The only difference from Task 2 is `global_max_sample_num = 30` in `main.py`.  
> To reproduce Task 2, change this value back to `20`.

---

## Results Summary

| Dataset | Baseline (XGBoost) | Paper (GPT-3.5, 20 iter) | Task 2 (LLaMA-8B, 20 iter) | Task 3 (LLaMA-8B, 30 iter) |
|---|---|---|---|---|
| balance-scale | 0.8576 | 0.9900 | 0.9548 | **0.9564** |
| breast-w | 0.9560 | 0.9700 | N/A | **0.9667** |

---

## Environment

| Component | Specification |
|---|---|
| OS | Windows 11 (HP Laptop) |
| Python | 3.11.9 |
| LLM Model | LLaMA-3.1-8B-Instant via Groq API (Free Tier) |
| Prediction Model | XGBoost 3.2.0 |
| Cross-Validation | 5-Fold Stratified KFold, random_state=42 |
| LLM Temperature | 0.8 |
| Number of Islands | 3 |
| Samples per Prompt | 3 |

---

## Course Concepts Applied

| Concept | Where Applied |
|---|---|
| Feature Creation (FE slides) | LLM generates new features by combining existing ones using math operations |
| Hyperparameter Optimization (Task 3 guideline) | Increasing `global_max_sample_num` from 20 to 30 |
| Data Engineering / Generalization (Task 3 guideline) | Adding breast-w as a second evaluation dataset |
| K-Fold Cross Validation (ML Evaluation slides) | 5-Fold Stratified KFold used throughout |
| Accuracy Metric (ML Evaluation slides) | Primary evaluation metric across all experiments |

---

## References

1. Abhyankar, N., Shojaee, P., & Reddy, C. K. (2025). LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers. arXiv:2503.14434v2.
2. GitHub: https://github.com/nikhilsab/LLMFE
3. Chen, T., & Guestrin, C. (2016). XGBoost. ACM KDD 2016.
4. Groq API Docs: https://console.groq.com/docs
5. OpenML Balance-Scale (ID: 11): https://www.openml.org/d/11
6. OpenML Breast-W (ID: 15): https://www.openml.org/d/15