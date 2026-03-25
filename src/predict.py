"""
predict.py — Classify new articles using the trained pipeline.

This is the inference module — the end of the pipeline. It takes a raw
article (as text or from the command line) and returns:
  - The predicted topic ID and label
  - The model's confidence (probability)
  - The top words that drove the prediction (from model coefficients)

Usage:
  python -m src.predict --text "The Fed raised interest rates by 25 basis points."
  python -m src.predict --title "Fed hikes rates" --text "Full article text here..."
  python -m src.predict --file path/to/article.txt

Why a separate predict module?
  During training, we fit on a fixed dataset. At inference time, a new article
  arrives that the model has never seen. predict.py loads the *saved* artifacts
  (vectorizer + model) and applies the exact same transformation pipeline that
  was used during training. This guarantees consistency.
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR         = Path("data/processed")
TFIDF_VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
MODELS_DIR            = PROCESSED_DIR / "models"
TOPIC_INFO_PATH       = PROCESSED_DIR / "topic_info.csv"

# Number of top contributing words to display in the explanation
TOP_WORDS = 8


def load_artifacts():
    """Load the saved vectorizer, model, and label encoder."""
    if not (MODELS_DIR / "logistic_regression.pkl").exists():
        print("ERROR: No trained model found. Run train.py first.")
        sys.exit(1)

    vectorizer = pickle.load(open(TFIDF_VECTORIZER_PATH, "rb"))
    model      = pickle.load(open(MODELS_DIR / "logistic_regression.pkl", "rb"))
    le         = pickle.load(open(MODELS_DIR / "label_encoder.pkl", "rb"))

    # Load topic keyword labels for human-readable output
    topic_info = pd.read_csv(TOPIC_INFO_PATH)
    topic_names = dict(zip(topic_info["Topic"], topic_info["Name"]))

    return vectorizer, model, le, topic_names


def preprocess_and_vectorize(title: str, text: str, vectorizer) -> "scipy.sparse matrix":
    """
    Apply the same preprocessing + vectorization as transform.py.

    IMPORTANT: We use the SAVED vectorizer — not a new one. This ensures
    the vocabulary and IDF weights are identical to what the model was
    trained on. Re-fitting on new data would shift all the word scores
    and produce garbage predictions.
    """
    import re
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

    combined = (title + " " + text) if title else text

    # Same preprocessing as transform.preprocess_text
    combined = combined.lower()
    combined = re.sub(r"http\S+|www\S+", " ", combined)
    combined = re.sub(r"[^a-z\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()
    tokens   = [w for w in combined.split() if w not in STOPWORDS and len(w) > 1]
    cleaned  = " ".join(tokens)

    return vectorizer.transform([cleaned])


def predict(title: str, text: str, verbose: bool = True) -> dict:
    """
    Predict the topic of a new article.

    Returns a dict with:
      topic_id    : integer topic ID
      topic_label : human-readable topic name (e.g. "2_russia_russian_comey_trump")
      confidence  : probability of the predicted class (0–1)
      top_words   : list of (word, score) tuples explaining the prediction
    """
    vectorizer, model, le, topic_names = load_artifacts()

    # Vectorize
    vec = preprocess_and_vectorize(title, text, vectorizer)

    # Predict
    pred_encoded   = model.predict(vec)[0]
    proba          = model.predict_proba(vec)[0]
    confidence     = proba.max()
    topic_id       = le.inverse_transform([pred_encoded])[0]
    topic_label    = topic_names.get(topic_id, str(topic_id))

    # Explain: top words driving this prediction
    feature_names = vectorizer.get_feature_names_out()
    coefs         = model.coef_[pred_encoded]
    article_vec   = vec.toarray().flatten()

    # Word contribution = coefficient × TF-IDF score (same as SHAP for linear model)
    contributions = coefs * article_vec
    top_idx       = np.argsort(contributions)[::-1][:TOP_WORDS]
    top_words     = [(feature_names[i], round(contributions[i], 4)) for i in top_idx
                     if contributions[i] > 0]

    result = {
        "topic_id":    topic_id,
        "topic_label": topic_label,
        "confidence":  round(confidence, 4),
        "top_words":   top_words,
    }

    if verbose:
        _print_result(title, text, result, proba, le, topic_names)

    return result


def _print_result(title, text, result, proba, le, topic_names):
    """Pretty-print the prediction result."""
    print()
    print("── Prediction ────────────────────────────────────────────────")
    if title:
        print(f"  Title:        {title[:80]}")
    print(f"  Text:         {text[:80]}{'...' if len(text) > 80 else ''}")
    print()
    print(f"  Topic ID:     {result['topic_id']}")
    print(f"  Topic label:  {result['topic_label']}")
    print(f"  Confidence:   {result['confidence']:.1%}")
    print()
    print(f"  Top words driving this prediction:")
    for word, score in result["top_words"]:
        bar = "█" * int(score * 200)
        print(f"    {word:<30} {score:.4f}  {bar}")

    # Show top 5 runner-up topics
    sorted_idx = np.argsort(proba)[::-1][:5]
    print()
    print("  Top 5 topic probabilities:")
    for idx in sorted_idx:
        tid   = le.inverse_transform([idx])[0]
        name  = topic_names.get(tid, str(tid))[:45]
        bar   = "█" * int(proba[idx] * 50)
        print(f"    [{tid:>3}] {name:<45} {proba[idx]:.2%}  {bar}")
    print("──────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="Predict the topic of a news article."
    )
    parser.add_argument("--text",  type=str, help="Article text", default="")
    parser.add_argument("--title", type=str, help="Article title (optional)", default="")
    parser.add_argument("--file",  type=str, help="Path to a .txt file containing the article")

    args = parser.parse_args()

    if args.file:
        content = Path(args.file).read_text()
        predict(title="", text=content)
    elif args.text:
        predict(title=args.title, text=args.text)
    else:
        # Interactive mode
        print("Enter article text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        predict(title="", text=" ".join(lines))


if __name__ == "__main__":
    main()
