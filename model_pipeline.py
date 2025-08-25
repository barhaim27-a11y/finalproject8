from __future__ import annotations
from typing import Tuple, List, Dict, Any
import os, json, shutil
import numpy as np, pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

def _ensure_dirs():
    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path("assets").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def _generate_synthetic(n: int = 195, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    def clip(a,b,c): return np.clip(a,b,c)
    Fo = clip(rng.normal(150,30,n),80,300)
    Fhi = Fo + clip(rng.normal(20,20,n),5,100)
    Flo = Fo - clip(rng.normal(20,20,n),5,100)
    jitterp = clip(rng.normal(0.005,0.003,n),0.0005,0.03)
    jitterabs = clip(rng.normal(0.00005,0.00003,n),0.000005,0.001)
    rap = jitterp * rng.uniform(0.6,1.4,n)
    ppq = jitterp * rng.uniform(0.5,1.5,n)
    ddp = rap*3
    shimmer = clip(rng.normal(0.03,0.015,n),0.005,0.2)
    shimmerdb = clip(rng.normal(0.3,0.2,n),0.02,1.5)
    apq3 = shimmer * rng.uniform(0.6,1.2,n)
    apq5 = shimmer * rng.uniform(0.6,1.2,n)
    apq = shimmer * rng.uniform(0.7,1.3,n)
    dda = apq3*3
    nhr = clip(rng.normal(0.03,0.02,n),0.001,0.3)
    hnr = clip(rng.normal(21,5,n),5,40)
    status = rng.integers(0,2,n)
    rpde = rng.uniform(0.2,0.8,n)
    dfa = rng.uniform(0.5,0.9,n)
    spread1 = rng.normal(-5,2,n)
    spread2 = rng.normal(2.5,1,n)
    d2 = rng.uniform(1.0,3.0,n)
    ppe = rng.uniform(0.1,0.8,n)
    name = [f"synthetic_{i}" for i in range(n)]
    return pd.DataFrame({
        "name": name, "MDVP:Fo(Hz)": Fo, "MDVP:Fhi(Hz)": Fhi, "MDVP:Flo(Hz)": Flo,
        "MDVP:Jitter(%)": jitterp, "MDVP:Jitter(Abs)": jitterabs, "MDVP:RAP": rap, "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp, "MDVP:Shimmer": shimmer, "MDVP:Shimmer(dB)": shimmerdb, "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5, "MDVP:APQ": apq, "Shimmer:DDA": dda, "NHR": nhr, "HNR": hnr, "status": status,
        "RPDE": rpde, "DFA": dfa, "spread1": spread1, "spread2": spread2, "D2": d2, "PPE": ppe
    })

def validate_training_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    required = config.FEATURES + [config.TARGET]
    miss = [c for c in required if c not in df.columns]
    if miss:
        errors.append("Missing columns: " + ", ".join(miss))
        return False, errors
    for col in config.FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric")
    if not pd.api.types.is_numeric_dtype(df[config.TARGET]):
        errors.append(f"Target '{config.TARGET}' must be numeric")
    return len(errors)==0, errors

def load_data(path: str) -> pd.DataFrame:
    _ensure_dirs()
    p = Path(path)
    if not p.exists():
        df = _generate_synthetic()
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return df
    return pd.read_csv(p)

def _preprocessor() -> ColumnTransformer:
    feats = config.FEATURES
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, feats)], remainder="drop")

def _make_classifier(name: str, params: Dict[str, Any]):
    if name == "LogisticRegression":
        return LogisticRegression(random_state=config.RANDOM_STATE, **params)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=config.RANDOM_STATE, **params)
    if name == "SVC":
        return SVC(random_state=config.RANDOM_STATE, **params)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        d = dict(eval_metric="logloss", random_state=config.RANDOM_STATE, tree_method="hist")
        d.update(params or {})
        return XGBClassifier(**d)
    if name == "MLP":
        return MLPClassifier(random_state=config.RANDOM_STATE, **params)
    if name == "KerasNN":
        from scikeras.wrappers import KerasClassifier
        import tensorflow as tf
        def build_keras_model(n_features_in_, hidden1=128, hidden2=64, dropout=0.1, lr=1e-3):
            inputs = tf.keras.Input(shape=(n_features_in_,))
            x = tf.keras.layers.Dense(hidden1, activation="relu")(inputs)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(hidden2, activation="relu")(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            m = tf.keras.Model(inputs, outputs)
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss="binary_crossentropy", metrics=["AUC"])
            return m
        return KerasClassifier(model=build_keras_model, **(params or {}))
    raise ValueError(f"Unknown model: {name}")

def create_pipeline(model_name: str, model_params: dict) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor()), ("classifier", _make_classifier(model_name, model_params))])

def _get_proba(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X); return (d-d.min())/(d.max()-d.min()+1e-8)
    return model.predict(X)

def _save_plots(y_true, y_proba, model_name: str):
    assets = Path("assets"); assets.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {model_name}"); plt.legend(); plt.savefig(assets/"roc.png", dpi=150, bbox_inches="tight"); plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_proba); ap = average_precision_score(y_true, y_proba)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – {model_name}"); plt.legend(); plt.savefig(assets/"pr.png", dpi=150, bbox_inches="tight"); plt.close()
    # CM
    cm = confusion_matrix(y_true, (y_proba>=0.5).astype(int)); disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d"); plt.title("Confusion Matrix"); plt.savefig(assets/"cm.png", dpi=150, bbox_inches="tight"); plt.close()

def _compute_metrics(y_true, y_proba, y_pred, model_name: str) -> Dict[str, Any]:
    return {"model_name": model_name, "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)), "recall": float(recall_score(y_true, y_pred)),
            "n_samples": int(len(y_true))}

def train_model(data_path: str, model_name: str, model_params: dict, test_size: float=0.2, do_cv: bool=True, do_tune: bool=True) -> Dict[str, Any]:
    _ensure_dirs()
    df = load_data(data_path)
    ok, errs = validate_training_data(df)
    if not ok: return {"ok": False, "errors": errs}

    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)

    pipe = create_pipeline(model_name, model_params)

    cv_means = None
    if do_cv:
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        scoring = ["roc_auc","accuracy","f1","precision","recall"]
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        cv_means = {m: float(np.mean(scores[f"test_{m}"])) for m in ["roc_auc","accuracy","f1","precision","recall"]}

    if do_tune:
        grid = config.PARAM_GRIDS.get(model_name, None)
        if grid:
            gs = GridSearchCV(pipe, grid, scoring=config.SCORING, cv=3, n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_

    pipe.fit(X_tr, y_tr)
    y_proba = _get_proba(pipe, X_val); y_pred = (y_proba>=0.5).astype(int)
    metrics = _compute_metrics(y_val, y_proba, y_pred, model_name)
    _save_plots(y_val, y_proba, model_name)

    joblib.dump(pipe, config.TEMP_MODEL_PATH)
    return {"ok": True, "candidate_path": config.TEMP_MODEL_PATH, "val_metrics": metrics, "cv_means": cv_means}

def evaluate_model(model_path: str, data_path: str=None) -> Dict[str, Any]:
    _ensure_dirs()
    if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found: {model_path}")
    pipe = joblib.load(model_path)
    df = load_data(data_path or config.TRAIN_DATA_PATH)
    X = df[config.FEATURES]; y = df[config.TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    y_proba = _get_proba(pipe, X_te); y_pred = (y_proba>=0.5).astype(int)
    metrics = _compute_metrics(y_te, y_proba, y_pred, "loaded_model")
    _save_plots(y_te, y_proba, "loaded_model")
    return metrics

def promote_model_to_production() -> str:
    _ensure_dirs()
    if not os.path.exists(config.TEMP_MODEL_PATH): raise FileNotFoundError("No candidate model to promote. Please train first.")
    shutil.copy(config.TEMP_MODEL_PATH, config.MODEL_PATH)
    os.remove(config.TEMP_MODEL_PATH)
    return f"✅ Promoted to production: {config.MODEL_PATH}"

def run_prediction(input_df: pd.DataFrame) -> tuple[int, float]:
    if not os.path.exists(config.MODEL_PATH): raise FileNotFoundError(f"Production model not found at {config.MODEL_PATH}. Train & promote first.")
    pipe = joblib.load(config.MODEL_PATH)
    pred = int(pipe.predict(input_df)[0])
    if hasattr(pipe, "predict_proba"): proba = float(pipe.predict_proba(input_df)[:,1][0])
    else:
        d = pipe.decision_function(input_df); d = (d-d.min())/(d.max()-d.min()+1e-8); proba = float(d[0])
    return pred, proba

def batch_predict(file_path: str) -> pd.DataFrame:
    df_in = pd.read_csv(file_path)
    missing = [c for c in config.FEATURES if c not in df_in.columns]
    for c in missing: df_in[c] = np.nan
    df_in = df_in[config.FEATURES]
    if not os.path.exists(config.MODEL_PATH): raise FileNotFoundError(f"Production model not found at {config.MODEL_PATH}. Train & promote first.")
    pipe = joblib.load(config.MODEL_PATH)
    def _get_proba_local(model, X):
        if hasattr(model, "predict_proba"): return model.predict_proba(X)[:,1]
        if hasattr(model, "decision_function"):
            d = model.decision_function(X); return (d-d.min())/(d.max()-d.min()+1e-8)
        return model.predict(X)
    proba = _get_proba_local(pipe, df_in)
    pred = (proba>=0.5).astype(int)
    out = df_in.copy(); out["proba_PD"] = proba; out["pred"] = pred
    return out
