"""
IFOSS Benchmark Script (Baseline vs Optuna-Tuned IFOSS)

This script benchmarks multiple classifiers with and without IFOSS
(Isolation Forest + One-Sided Selection).

Outputs:
  1) Baseline results (No IFOSS)
  2) Results with Optuna-tuned IFOSS
  3) Absolute percentage-point improvement (With − Without)


Required variables in session:
    Xl, yl      -> training features/labels
    Xl_t, yl_t  -> external test/holdout features/labels
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import optuna
from typing import Tuple

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from imblearn import FunctionSampler
from imblearn.under_sampling import OneSidedSelection

from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    accuracy_score, balanced_accuracy_score,
    roc_curve
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_TRIALS = 200


def _to_numpy(y):
    if hasattr(y, "values"):
        y = y.values
    return np.asarray(y).reshape(-1)


yl_np = _to_numpy(yl)
yl_t_np = _to_numpy(yl_t)


X_inner_train, X_inner_valid, y_inner_train, y_inner_valid = train_test_split(
    Xl, yl_np,
    test_size=0.20,
    stratify=yl_np,
    random_state=SEED
)


cat_idx = [
    Xl.columns.get_loc(c)
    for c in Xl.columns
    if Xl[c].dtype == "object" or Xl[c].dtype.name == "category"
]

cat_cols = [
    c for c in Xl.columns
    if Xl[c].dtype == "object" or Xl[c].dtype.name == "category"
]

num_cols = [c for c in Xl.columns if c not in cat_cols]


svm_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ],
    remainder="drop"
)

TASK_TYPE = "CPU"
THREADS = -1

baseline_models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear", random_state=SEED, max_iter=2000
    ),
    "SVM (RBF)": Pipeline(steps=[
        ("prep", svm_preprocessor),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=SEED
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=1000, random_state=SEED, n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=1000,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        use_label_encoder=False,
        enable_categorical=True
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=1000, random_state=SEED, verbose=-1, n_jobs=-1
    ),
    "CatBoost": cb.CatBoostClassifier(
        iterations=1000,
        loss_function="Logloss",
        random_seed=SEED,
        verbose=0,
        task_type=TASK_TYPE,
        thread_count=THREADS,
        early_stopping_rounds=200
    )
}


def youden_best(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    J = tpr - fpr
    ix = np.argmax(J)

    best_thr = thr[ix]
    sens = tpr[ix]
    spec = 1.0 - fpr[ix]
    gmean = float(np.sqrt(max(sens, 0.0) * max(spec, 0.0)))

    return float(best_thr), float(J[ix]), float(sens), float(spec), gmean


def evaluate_stage(y_true, proba):

    auc = roc_auc_score(y_true, proba)

    best_thr, J, sens, spec, gmean = youden_best(y_true, proba)

    y_pred = (proba >= best_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    _ = fbeta_score(y_true, y_pred, beta=0.5, average="weighted", zero_division=0)

    return dict(
        auc=auc,
        f1w=f1w,
        acc=acc,
        bal_acc=bal_acc,
        gmean=gmean,
        youdenJ=J,
        best_thr=best_thr,
        sens=sens,
        spec=spec
    )


def apply_ifoss(X, y, max_samples, contamination, max_features, n_neighbors, n_seeds_S):

    def outlier_rejection(X_, y_):
        model = IsolationForest(
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=SEED,
            bootstrap=True
        )
        model.fit(X_)
        mask = model.predict(X_) == 1
        return X_[mask], y_[mask]

    sampler = FunctionSampler(func=outlier_rejection)
    X1, y1 = sampler.fit_resample(X, y)

    oss = OneSidedSelection(
        sampling_strategy="majority",
        n_neighbors=n_neighbors,
        n_seeds_S=n_seeds_S,
        n_jobs=-1,
        random_state=SEED
    )

    X_res, y_res = oss.fit_resample(X1, y1)

    if isinstance(X_res, np.ndarray):
        X_res = pd.DataFrame(X_res, columns=X.columns)

    return X_res, y_res


def make_objective(model, name):

    def objective(trial):

        pars = {
            "max_samples": trial.suggest_int("max_samples", 1000, len(X_inner_train)),
            "contamination": trial.suggest_float("contamination", 0.01, 0.5),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "oss_n_neighbors": trial.suggest_int("oss_n_neighbors", 1, 10),
            "oss_n_seeds_S": trial.suggest_int("oss_n_seeds_S", 100, 1000)
        }

        X_res, y_res = apply_ifoss(
            X_inner_train, y_inner_train,
            max_samples=pars["max_samples"],
            contamination=pars["contamination"],
            max_features=pars["max_features"],
            n_neighbors=pars["oss_n_neighbors"],
            n_seeds_S=pars["oss_n_seeds_S"]
        )

        if name == "CatBoost":
            model.fit(
                X_res, y_res,
                eval_set=(X_inner_valid, y_inner_valid),
                cat_features=cat_idx,
                use_best_model=True
            )
        else:
            model.fit(X_res, y_res)

        proba = model.predict_proba(X_inner_valid)[:, 1]
        _, _, _, _, gmean = youden_best(y_inner_valid, proba)

        return float(gmean)

    return objective


results_no_ifoss, results_ifoss = {}, {}

for name, model in baseline_models.items():

    if name == "CatBoost":
        model.fit(Xl, yl_np, eval_set=(Xl_t, yl_t_np),
                  cat_features=cat_idx, use_best_model=True)
    else:
        model.fit(Xl, yl_np)

    proba_base = model.predict_proba(Xl_t)[:, 1]
    res_base = evaluate_stage(yl_t_np, proba_base)

    results_no_ifoss[name] = {
        k: res_base[k] for k in ["auc", "f1w", "acc", "bal_acc", "gmean"]
    }

    print(f"\n>>> Optuna tuning IFOSS for [{name}] (n_trials={N_TRIALS})")

    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(model, name),
                   n_trials=N_TRIALS,
                   show_progress_bar=True)

    best_p = study.best_trial.params

    X_res, y_res = apply_ifoss(Xl, yl_np, **best_p)

    if name == "CatBoost":
        model.fit(X_res, y_res, eval_set=(Xl_t, yl_t_np),
                  cat_features=cat_idx, use_best_model=True)
    else:
        model.fit(X_res, y_res)

    proba_ifoss = model.predict_proba(Xl_t)[:, 1]
    res_ifoss = evaluate_stage(yl_t_np, proba_ifoss)

    results_ifoss[name] = {
        k: res_ifoss[k] for k in ["auc", "f1w", "acc", "bal_acc", "gmean"]
    }


metrics_cols = ["auc", "f1w", "acc", "bal_acc", "gmean"]

df_base = pd.DataFrame(results_no_ifoss).T[metrics_cols]
df_ifoss = pd.DataFrame(results_ifoss).T[metrics_cols]
df_improve = (df_ifoss - df_base) * 100

print("\n===== Table 1: Baseline Results (No IFOSS) =====")
print(df_base.round(6).to_markdown())

print("\n===== Table 2: With IFOSS (Optuna tuned) =====")
print(df_ifoss.round(6).to_markdown())

print("\n===== Table 3: Absolute % Point Change (With IFOSS − No IFOSS) =====")
print(df_improve.round(2).to_markdown())
