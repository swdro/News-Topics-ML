"""
train.py — Train and compare topic classifiers, log everything to MLflow.

BERTopic labels + TF-IDF features → [train.py] → trained classifier

Two models are trained and compared: Logistic Regression (baseline — fast,
linear, strong on sparse text) and XGBoost (challenger — gradient-boosted
trees, non-linear). No single algorithm wins on every dataset, so we train
both, log metrics to MLflow, and pick the best one.

LR is the standard baseline for text classification — handles sparse
high-dimensional features well, trains in seconds, and is interpretable.
If XGBoost can't beat LR, we stick with LR (simpler is better).
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR        = Path("data/processed")
ARTICLES_CLEAN_PATH  = PROCESSED_DIR / "articles_clean.parquet"
ARTICLES_TOPICS_PATH = PROCESSED_DIR / "articles_with_topics.parquet"
TFIDF_MATRIX_PATH    = PROCESSED_DIR / "tfidf_matrix.npz"
MODELS_DIR           = PROCESSED_DIR / "models"

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2    # hold out 20% of data for evaluation
RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "news-topic-classification"

# Logistic Regression hyperparameters
LR_C         = 5.0      # regularization strength (higher = less regularization)
LR_MAX_ITER  = 1000     # max solver iterations
LR_SOLVER    = "lbfgs"  # good default for multiclass

# XGBoost hyperparameters
XGB_N_ESTIMATORS  = 100
XGB_MAX_DEPTH     = 4
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE     = 0.8   # train each tree on 80% of data — reduces overfitting


def load_data():
    """
    Load TF-IDF features and align them with topic labels.

    The tricky part: tfidf_matrix has 141k rows (all articles), but
    articles_with_topics only has 87k (outliers removed). We join on
    article 'id' to select only the rows that have valid topic labels.
    """
    print("  Loading TF-IDF matrix...")
    tfidf_full = sp.load_npz(str(TFIDF_MATRIX_PATH))
    print(f"  Full TF-IDF matrix: {tfidf_full.shape}")

    # articles_clean gives us the row-order that corresponds to tfidf_matrix
    df_clean = pd.read_parquet(ARTICLES_CLEAN_PATH, columns=["id"])
    df_clean["row_idx"] = np.arange(len(df_clean))

    # articles_with_topics has the 87k articles BERTopic labeled
    df_labeled = pd.read_parquet(ARTICLES_TOPICS_PATH, columns=["id", "topic_id"])

    # Join to get the tfidf row index for each labeled article
    df_merged = df_labeled.merge(df_clean, on="id", how="inner")
    print(f"  Labeled articles matched to TF-IDF rows: {len(df_merged):,}")

    # Slice the TF-IDF matrix to only the labeled rows — this is our feature matrix
    X = tfidf_full[df_merged["row_idx"].values, :]
    y_raw = df_merged["topic_id"].values

    # LabelEncoder converts topic IDs (0, 2, 5, 7...) to consecutive integers
    # (0, 1, 2, 3...) — required by XGBoost
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"  Feature matrix X: {X.shape}")
    print(f"  Labels y:         {len(y)} samples, {len(le.classes_)} classes")
    return X, y, le


def split_data(X, y):
    """
    Split into train and test sets (80/20), stratified by class so each
    class's proportion is preserved in both splits. Without stratification
    a small class might end up entirely in train or entirely in test,
    making evaluation unreliable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {X_train.shape[0]:,} articles")
    print(f"  Test:  {X_test.shape[0]:,} articles")
    return X_train, X_test, y_train, y_test


def log_metrics(y_test, y_pred, prefix=""):
    """Compute and return accuracy + macro F1."""
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    mlflow.log_metric(f"{prefix}accuracy", acc)
    mlflow.log_metric(f"{prefix}f1_macro", f1)
    return acc, f1


def train_logistic_regression(X_train, X_test, y_train, y_test, le):
    """
    Train Logistic Regression and log to MLflow.

    lbfgs handles multiclass natively (no one-vs-rest needed), is efficient on
    medium-sized datasets, and converges reliably. C=5 is inverse regularization
    (higher = less penalty on large weights) — tuned by convention for TF-IDF
    text; can be swept with cross-validation if needed.
    """
    print("\n  Training Logistic Regression...")

    with mlflow.start_run(run_name="logistic_regression"):
        # Log hyperparameters
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": LR_C,
            "max_iter": LR_MAX_ITER,
            "solver": LR_SOLVER,
            "n_classes": len(le.classes_),
            "n_features": X_train.shape[1],
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
        })

        model = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc, f1 = log_metrics(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}  |  F1 (macro): {f1:.4f}")

        # Log the full per-class report as a text artifact
        report = classification_report(y_test, y_pred,
                                       target_names=[str(c) for c in le.classes_])
        mlflow.log_text(report, "classification_report.txt")

        # Log the trained model
        mlflow.sklearn.log_model(model, "model")

        # Save locally too
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "logistic_regression.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    print(f"  Saved → {MODELS_DIR / 'logistic_regression.pkl'}")
    return model, acc, f1


def train_xgboost(X_train, X_test, y_train, y_test, le):
    """
    Train XGBoost and log to MLflow.

    XGBoost builds an ensemble of decision trees sequentially — each tree
    corrects the errors of the previous one (gradient boosting).

    Note: XGBoost on 50k TF-IDF features is slower than LR. We use
    tree_method='hist' which bins features into histograms for speed.
    Expect 5-15 minutes depending on hardware.
    """
    print("\n  Training XGBoost (this takes longer than LR — ~5-15 min)...")

    n_classes = len(le.classes_)

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({
            "model": "XGBoost",
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "n_classes": n_classes,
            "n_features": X_train.shape[1],
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
        })

        model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            objective="multi:softmax",
            num_class=n_classes,
            tree_method="hist",   # fast histogram algorithm for sparse data
            device="cpu",         # MPS support for XGBoost is limited; CPU is reliable
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc, f1 = log_metrics(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}  |  F1 (macro): {f1:.4f}")

        report = classification_report(y_test, y_pred,
                                       target_names=[str(c) for c in le.classes_])
        mlflow.log_text(report, "classification_report.txt")
        mlflow.xgboost.log_model(model, "model")

        with open(MODELS_DIR / "xgboost.pkl", "wb") as f:
            pickle.dump(model, f)

    print(f"  Saved → {MODELS_DIR / 'xgboost.pkl'}")
    return model, acc, f1


def print_comparison(lr_acc, lr_f1, xgb_acc, xgb_f1):
    winner = "Logistic Regression" if lr_f1 >= xgb_f1 else "XGBoost"
    print("\n── Model Comparison ──────────────────────────────────────────")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 (macro)':>12}")
    print(f"  {'─'*25} {'─'*10} {'─'*12}")
    print(f"  {'Logistic Regression':<25} {lr_acc:>10.4f} {lr_f1:>12.4f}")
    print(f"  {'XGBoost':<25} {xgb_acc:>10.4f} {xgb_f1:>12.4f}")
    print(f"\n  Winner: {winner}")
    print(f"\n  View all runs:  mlflow ui  (then open http://127.0.0.1:5000)")
    print("──────────────────────────────────────────────────────────────")


def run():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("── Step 1: Load data ─────────────────────────────────────────")
    X, y, le = load_data()

    print("\n── Step 2: Train/test split ──────────────────────────────────")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n── Step 3: Train Logistic Regression ─────────────────────────")
    _, lr_acc, lr_f1 = train_logistic_regression(X_train, X_test, y_train, y_test, le)

    print("\n── Step 4: Train XGBoost ─────────────────────────────────────")
    _, xgb_acc, xgb_f1 = train_xgboost(X_train, X_test, y_train, y_test, le)

    print_comparison(lr_acc, lr_f1, xgb_acc, xgb_f1)
    print("\nTraining complete.")


if __name__ == "__main__":
    run()
