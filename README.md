
---

## 1. Benchmark Script (Main Paper Results)

**File:** `benchmark_ifoss.py`

This script reproduces the main benchmarking experiments reported in the paper.

### Models Evaluated

Six supervised classifiers are benchmarked:

- Logistic Regression  
- Random Forest  
- SVM (RBF kernel)  
- XGBoost  
- LightGBM  
- CatBoost  

### Evaluation Protocol

The benchmark follows a strict nested procedure:

- **Outer split:**  
  Stratified 80% training / 20% held-out test split  

- **Inner split (Optuna tuning):**  
  80% inner-train / 20% validation split inside the training set  

- **IFOSS tuning objective:**  
  Optuna maximizes **G-Mean at Youden’s J threshold**, as described in the paper.

- **Final evaluation:**  
  All reported metrics are computed only on the held-out test set (`Xl_t`).

### Output

The script prints three tables:

1. Baseline performance (No IFOSS)  
2. Performance with IFOSS  
3. Absolute percentage-point improvement (Δ)

Metrics reported:

- AUC  
- Weighted F1-score  
- Accuracy  
- Balanced Accuracy  
- G-Mean  

---

## 2. UMAP Visualization (Supplementary Figures)

**File:** `umap_visualization.py`

This script generates the qualitative manifold visualizations described in:

**Section: UMAP-Based Visualization of Class Separability**

The figure compares:

1. Original training distribution  
2. Held-out test distribution  
3. Training data after Isolation Forest filtering  
4. Training data after One-Sided Selection undersampling  

These plots correspond to Supplementary Figures **S1–S15**.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
