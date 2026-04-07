import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn import FunctionSampler
from imblearn.under_sampling import OneSidedSelection

from sklearn.metrics import (
    roc_auc_score, classification_report,
    balanced_accuracy_score, roc_curve
)

import lightgbm as lgb

SEED = 42
N_TRIALS = 100
TARGET = "Death"

# =========================
# 1. PREPARE DATA
# =========================
df = df.copy()

# remove Age
if "Age" in df.columns:
    df = df.drop(columns=["Age"])

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# 2. SPLIT (HOLDOUT)
# =========================
Xl, Xl_t, yl, yl_t = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# =========================
# 3. ENCODE (for IFOSS + LGBM)
# =========================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

def encode_all(X):
    X_enc = X.copy()
    for col in cat_cols:
        X_enc[col] = X_enc[col].astype("category").cat.codes
    return X_enc

Xl_enc = encode_all(Xl)
Xl_t_enc = encode_all(Xl_t)

yl_np = yl.values
yl_t_np = yl_t.values

# =========================
# 4. INNER SPLIT (OPTUNA)
# =========================
X_tr, X_val, y_tr, y_val = train_test_split(
    Xl_enc, yl_np,
    test_size=0.2,
    stratify=yl_np,
    random_state=SEED
)

# =========================
# 5. IFOSS
# =========================
def apply_ifoss(X, y, max_samples, contamination, max_features, n_neighbors, n_seeds_S):

    def outlier_rejection(X_, y_):
        iso = IsolationForest(
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=SEED
        )
        iso.fit(X_)
        mask = iso.predict(X_) == 1
        return X_[mask], y_[mask]

    sampler = FunctionSampler(func=outlier_rejection)
    X1, y1 = sampler.fit_resample(X, y)

    oss = OneSidedSelection(
        sampling_strategy="majority",
        n_neighbors=n_neighbors,
        n_seeds_S=n_seeds_S,
        random_state=SEED,
        n_jobs=-1
    )

    X_res, y_res = oss.fit_resample(X1, y1)

    return X_res, y_res

# =========================
# 6. MODEL
# =========================
def build_model():
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1   # 🔥 removes warning spam
    )

# =========================
# 7. METRIC (GMEAN FOR OPTUNA)
# =========================
def youden_gmean(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    gmeans = np.sqrt(tpr * (1 - fpr))
    return np.max(gmeans)

# =========================
# 8. OPTUNA
# =========================
def objective(trial):

    params = {
        "max_samples": trial.suggest_int("max_samples", 500, len(X_tr)),
        "contamination": trial.suggest_float("contamination", 0.01, 0.3),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 10),
        "n_seeds_S": trial.suggest_int("n_seeds_S", 100, 500)
    }

    X_res, y_res = apply_ifoss(X_tr, y_tr, **params)

    model = build_model()
    model.fit(X_res, y_res)

    proba = model.predict_proba(X_val)[:, 1]

    return youden_gmean(y_val, proba)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

best_p = study.best_params
print("Best IFOSS params:", best_p)

# =========================
# 9. FINAL TRAIN
# =========================
X_res, y_res = apply_ifoss(Xl_enc, yl_np, **best_p)

model = build_model()
model.fit(X_res, y_res)

# =========================
# 10. FINAL EVALUATION
# =========================
y_pred = model.predict(Xl_t_enc)
proba = model.predict_proba(Xl_t_enc)[:, 1]

# classification report
report = classification_report(
    yl_t_np, y_pred,
    output_dict=True
)

auc = roc_auc_score(yl_t_np, proba)
bal_acc = balanced_accuracy_score(yl_t_np, y_pred)

# =========================
# 11. FORMAT LIKE PAPER TABLE
# =========================
rows = []

# class 0 and 1
for cls in ["0", "1"]:
    rows.append([
        f"Class {cls}",
        report[cls]["precision"],
        report[cls]["recall"],
        report[cls]["f1-score"],
        report[cls]["support"]
    ])

# macro + weighted
for avg in ["macro avg", "weighted avg"]:
    rows.append([
        avg.title(),
        report[avg]["precision"],
        report[avg]["recall"],
        report[avg]["f1-score"],
        report[avg]["support"]
    ])

table = pd.DataFrame(
    rows,
    columns=["Model", "Precision", "Recall", "F1-Score", "Support"]
)

print("\n===== CLASSIFICATION REPORT =====")
print(table.round(3).to_markdown(index=False))

print("\n===== EXTRA METRICS =====")
print(f"AUC: {auc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
