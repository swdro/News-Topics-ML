"""
evaluate.py — Detailed model evaluation and SHAP explainability.

Reloads the trained Logistic Regression and test data, computes accuracy/F1/
precision/recall per topic class, generates a normalized confusion matrix, runs
SHAP to explain which words drive each topic prediction, then saves everything
and logs to MLflow.

89% accuracy is only half the story — SHAP answers why the model predicted a
topic, which words pushed toward topic X and away from topic Y. Useful for
debugging and communicating results to stakeholders.

Full SHAP on 73 classes × 50k features × N samples requires GB of RAM. For a
linear model, SHAP values are coef[k,j] × (x_j - mean_x_j), so model.coef_ IS
the global feature importance. We use coef_ for global/per-topic importance and
reserve shap.LinearExplainer for focused single-article explanations using only
non-zero features.
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR        = Path("data/processed")
ARTICLES_CLEAN_PATH  = PROCESSED_DIR / "articles_clean.parquet"
ARTICLES_TOPICS_PATH = PROCESSED_DIR / "articles_with_topics.parquet"
TFIDF_MATRIX_PATH    = PROCESSED_DIR / "tfidf_matrix.npz"
TFIDF_VECTORIZER_PATH= PROCESSED_DIR / "tfidf_vectorizer.pkl"
MODELS_DIR           = PROCESSED_DIR / "models"
REPORTS_DIR          = PROCESSED_DIR / "reports"

RANDOM_STATE = 42
MLFLOW_EXPERIMENT = "news-topic-classification"

# Number of top words to show per topic in importance analysis
TOP_N_WORDS = 15

# For SHAP single-article explanations: max features to display
SHAP_MAX_FEATURES = 20


def load_artifacts():
    """Load model, label encoder, vectorizer, and reconstruct test set."""
    print("  Loading model and data...")

    model = pickle.load(open(MODELS_DIR / "logistic_regression.pkl", "rb"))
    le    = pickle.load(open(MODELS_DIR / "label_encoder.pkl", "rb"))
    vectorizer = pickle.load(open(TFIDF_VECTORIZER_PATH, "rb"))

    # Reconstruct the same test split used in train.py
    tfidf_full = sp.load_npz(str(TFIDF_MATRIX_PATH))
    df_clean   = pd.read_parquet(ARTICLES_CLEAN_PATH, columns=["id"])
    df_clean["row_idx"] = np.arange(len(df_clean))
    df_labeled = pd.read_parquet(ARTICLES_TOPICS_PATH)
    df_merged  = df_labeled.merge(df_clean, on="id", how="inner")

    X = tfidf_full[df_merged["row_idx"].values, :]
    y = le.transform(df_merged["topic_id"].values)

    _, X_test, _, y_test, _, df_test = train_test_split(
        X, y, df_merged,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    feature_names = vectorizer.get_feature_names_out()
    print(f"  Test set: {X_test.shape[0]:,} articles, {len(le.classes_)} classes")
    return model, le, vectorizer, X_test, y_test, df_test, feature_names


def compute_metrics(model, X_test, y_test, le):
    """Compute and print accuracy, macro F1, and per-class report."""
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in le.classes_],
        output_dict=False,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in le.classes_],
        output_dict=True,
    )

    print(f"\n  Accuracy:   {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"\n  Per-class report (first 10 classes):")
    lines = report.split("\n")
    print("\n".join(lines[:14]))  # header + first 10 classes

    return y_pred, acc, f1, report, report_dict


def plot_confusion_matrix(y_test, y_pred, le, output_dir: Path):
    """
    Plot a normalized confusion matrix (each row sums to 1.0, each cell is the
    fraction of true-class articles predicted as each class). More readable than
    raw counts when class sizes vary widely.
    """
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    labels = [str(c) for c in le.classes_]

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, vmin=0, vmax=1,
        annot_kws={"size": 6},
    )
    ax.set_title("Normalized Confusion Matrix — Logistic Regression\n(each row sums to 1.0)", fontsize=13)
    ax.set_ylabel("True Topic")
    ax.set_xlabel("Predicted Topic")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {path}")
    return path


def global_topic_importance(model, feature_names, le, output_dir: Path):
    """
    Extract the top words for each topic using model.coef_.

    For a linear model, coef_[k, j] is the weight the model places on
    feature j when predicting class k. A high positive weight means the
    word strongly pushes toward that topic.

    This is mathematically equivalent to the mean absolute SHAP value
    for a linear model when features are approximately zero-centered
    (which TF-IDF is — most values are 0).
    """
    print(f"\n  Computing per-topic word importance from model coefficients...")

    rows = []
    for class_idx, class_label in enumerate(le.classes_):
        coefs = model.coef_[class_idx]
        top_idx = np.argsort(coefs)[::-1][:TOP_N_WORDS]
        for rank, feat_idx in enumerate(top_idx):
            rows.append({
                "topic_id":    class_label,
                "rank":        rank + 1,
                "word":        feature_names[feat_idx],
                "importance":  coefs[feat_idx],
            })

    importance_df = pd.DataFrame(rows)
    path = output_dir / "topic_word_importance.csv"
    importance_df.to_csv(path, index=False)
    print(f"  Saved word importance table → {path}")

    # Plot top 8 topics by article count
    df_labeled = pd.read_parquet(ARTICLES_TOPICS_PATH, columns=["topic_id"])
    top_topic_ids = df_labeled["topic_id"].value_counts().head(8).index.tolist()
    top_classes   = [i for i, c in enumerate(le.classes_) if c in top_topic_ids]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    df_topics = pd.read_csv(PROCESSED_DIR / "topic_info.csv")
    topic_keywords = dict(zip(df_topics["Topic"], df_topics["Name"]))

    for plot_idx, class_idx in enumerate(top_classes[:8]):
        coefs = model.coef_[class_idx]
        top_idx = np.argsort(coefs)[::-1][:12]
        words  = feature_names[top_idx]
        scores = coefs[top_idx]

        ax = axes[plot_idx]
        ax.barh(words[::-1], scores[::-1], color=sns.color_palette("muted")[plot_idx % 8])
        label = le.classes_[class_idx]
        name  = topic_keywords.get(label, str(label))
        ax.set_title(f"Topic {label}: {name[:40]}", fontsize=8)
        ax.set_xlabel("Coefficient", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.suptitle("Top Words per Topic (model.coef_ = SHAP for linear models)", fontsize=12, y=1.01)
    plt.tight_layout()
    path_fig = output_dir / "topic_word_importance.png"
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved importance chart  → {path_fig}")

    return importance_df


def shap_single_article(model, X_test, y_test, le, feature_names, df_test, output_dir: Path):
    """
    Use SHAP LinearExplainer to explain a single article prediction.

    Strategy: only compute SHAP on the non-zero features of one article.
    This keeps memory usage tiny — a typical article has 200-500 non-zero
    TF-IDF features out of 50,000.
    """
    print("\n  Running SHAP for single-article explanation...")

    # Pick a correctly-classified article from the largest topic
    y_pred = model.predict(X_test)
    correct_mask = (y_pred == y_test)

    # Find first correct prediction from topic 0 (largest class)
    candidates = np.where(correct_mask & (y_test == 0))[0]
    if len(candidates) == 0:
        candidates = np.where(correct_mask)[0]
    article_idx = candidates[0]

    true_class  = le.classes_[y_test[article_idx]]
    pred_class  = le.classes_[y_pred[article_idx]]

    # Get the non-zero feature indices for this article
    article_vec  = X_test[article_idx]
    nonzero_idx  = article_vec.nonzero()[1]

    if len(nonzero_idx) == 0:
        print("  Skipping SHAP — empty article vector.")
        return

    # Build a small dense sub-matrix: background = 100 random train articles
    # (we reload train split here for the background)
    tfidf_full = sp.load_npz(str(TFIDF_MATRIX_PATH))
    df_clean   = pd.read_parquet(ARTICLES_CLEAN_PATH, columns=["id"])
    df_clean["row_idx"] = np.arange(len(df_clean))
    df_labeled = pd.read_parquet(ARTICLES_TOPICS_PATH, columns=["id", "topic_id"])
    df_merged  = df_labeled.merge(df_clean, on="id", how="inner")
    X_full     = tfidf_full[df_merged["row_idx"].values, :]
    y_full     = np.zeros(len(df_merged))  # dummy — only need X
    X_train, _, _, _ = train_test_split(X_full, y_full, test_size=0.2, random_state=RANDOM_STATE)

    # Sample 100 background articles, use only the nonzero feature columns
    np.random.seed(RANDOM_STATE)
    bg_idx = np.random.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
    X_bg_dense  = X_train[bg_idx][:, nonzero_idx].toarray()
    X_art_dense = article_vec[:, nonzero_idx].toarray()

    # Slice model coefficients to same feature subset
    from sklearn.linear_model import LogisticRegression as LR
    coef_sub  = model.coef_[:, nonzero_idx]
    intercept = model.intercept_

    # Build a mini model on the feature subset for the explainer
    sub_model = LR()
    sub_model.coef_      = coef_sub
    sub_model.intercept_ = intercept
    sub_model.classes_   = model.classes_

    explainer   = shap.LinearExplainer(sub_model, X_bg_dense)
    shap_values = explainer.shap_values(X_art_dense)  # list of n_class arrays

    # Focus on the predicted class
    pred_class_idx = y_pred[article_idx]
    sv_for_pred    = shap_values[pred_class_idx][0]  # shape (n_nonzero_features,)
    sub_features   = feature_names[nonzero_idx]

    # Top pushing-toward and pushing-away features
    top_pos_idx = np.argsort(sv_for_pred)[::-1][:SHAP_MAX_FEATURES // 2]
    top_neg_idx = np.argsort(sv_for_pred)[:SHAP_MAX_FEATURES // 2]
    show_idx    = np.concatenate([top_neg_idx[::-1], top_pos_idx[::-1]])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["coral" if sv_for_pred[i] < 0 else "steelblue" for i in show_idx]
    ax.barh(sub_features[show_idx], sv_for_pred[show_idx], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"SHAP — Single Article Explanation\n"
        f"True: topic {true_class}  |  Predicted: topic {pred_class}",
        fontsize=11
    )
    ax.set_xlabel("SHAP value (positive = pushes toward predicted topic)")
    plt.tight_layout()

    path = output_dir / "shap_single_article.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP explanation  → {path}")

    # Print article title for context
    article_row = df_test.iloc[article_idx]
    print(f"\n  Article explained:")
    print(f"    Title:      {df_labeled.merge(df_clean, on='id').iloc[0]['id']}")
    print(f"    True topic: {true_class}  |  Predicted: {pred_class}")

    return path


def log_to_mlflow(acc, f1, report, conf_matrix_path, importance_path, shap_path):
    """Log evaluation results to the existing LR run in MLflow."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Find the existing logistic_regression run and log eval artifacts to it
    with mlflow.start_run(run_name="logistic_regression_eval"):
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_f1_macro", f1)
        mlflow.log_text(report, "classification_report.txt")
        if conf_matrix_path and conf_matrix_path.exists():
            mlflow.log_artifact(str(conf_matrix_path))
        if importance_path and importance_path.exists():
            mlflow.log_artifact(str(importance_path))
        if shap_path and shap_path.exists():
            mlflow.log_artifact(str(shap_path))
    print("\n  Logged evaluation artifacts to MLflow.")


def run():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("── Step 1: Load artifacts ────────────────────────────────────")
    model, le, vectorizer, X_test, y_test, df_test, feature_names = load_artifacts()

    print("\n── Step 2: Compute metrics ───────────────────────────────────")
    y_pred, acc, f1, report, _ = compute_metrics(model, X_test, y_test, le)

    print("\n── Step 3: Confusion matrix ──────────────────────────────────")
    cm_path = plot_confusion_matrix(y_test, y_pred, le, REPORTS_DIR)

    print("\n── Step 4: Global topic word importance (SHAP-equivalent) ───")
    importance_df = global_topic_importance(model, feature_names, le, REPORTS_DIR)

    print("\n── Step 5: SHAP single-article explanation ───────────────────")
    shap_path = shap_single_article(
        model, X_test, y_test, le, feature_names, df_test, REPORTS_DIR
    )

    print("\n── Step 6: Log to MLflow ─────────────────────────────────────")
    log_to_mlflow(acc, f1, report, cm_path,
                  REPORTS_DIR / "topic_word_importance.png", shap_path)

    print("\nEvaluation complete.")
    print(f"All reports saved to: {REPORTS_DIR}/")


if __name__ == "__main__":
    run()
