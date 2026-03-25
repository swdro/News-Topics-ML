"""
test_pipeline.py — Unit tests for individual pipeline component functions.

These tests use synthetic data — no real data files needed — so they all
run in CI. They verify that the pipeline functions behave correctly on
controlled inputs where we know the expected output.

Why test individual functions?
  Integration tests (run the whole pipeline and check the output) are
  valuable but slow and fragile. Unit tests give fast, precise feedback:
  if preprocess_text breaks, we know exactly which function failed and why.
"""

import pickle
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Paths (for integration-level tests that need real artifacts) ───────────────
MODELS_DIR           = Path("data/processed/models")
TFIDF_VECTORIZER_PATH= Path("data/processed/tfidf_vectorizer.pkl")

requires_model = pytest.mark.skipif(
    not (MODELS_DIR / "logistic_regression.pkl").exists(),
    reason="Requires trained model. Run train.py first."
)

# ════════════════════════════════════════════════════════════════
# UNIT TESTS — preprocess_text
# ════════════════════════════════════════════════════════════════

# Import the actual function from our module
from src.transform import preprocess_text, STOPWORDS


class TestPreprocessText:
    """Test every preprocessing step in isolation."""

    def test_lowercasing(self):
        """Text should be fully lowercased."""
        result = preprocess_text("The Federal Reserve RAISED Rates")
        assert result == result.lower()

    def test_url_removed(self):
        """URLs should be stripped from the text."""
        result = preprocess_text("Visit https://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_punctuation_removed(self):
        """Punctuation and numbers should be removed."""
        result = preprocess_text("Hello, world! 123 test.")
        assert "," not in result
        assert "!" not in result
        assert "123" not in result
        assert "." not in result

    def test_stopwords_removed(self):
        """Common stopwords should not appear in the output."""
        result = preprocess_text("the cat sat on the mat")
        tokens = result.split()
        for token in tokens:
            assert token not in STOPWORDS, f"Stopword '{token}' survived preprocessing"

    def test_single_char_tokens_removed(self):
        """Single-character tokens should be removed."""
        result = preprocess_text("a b c hello world")
        tokens = result.split()
        for token in tokens:
            assert len(token) > 1, f"Single-char token '{token}' survived"

    def test_meaningful_words_preserved(self):
        """Domain-specific meaningful words should survive preprocessing."""
        result = preprocess_text("The Federal Reserve raised interest rates today")
        assert "federal" in result
        assert "reserve" in result
        assert "interest" in result
        assert "rates" in result

    def test_empty_string(self):
        """Empty string should return empty string without error."""
        result = preprocess_text("")
        assert result == ""

    def test_only_stopwords(self):
        """Text made of only stopwords should return empty string."""
        result = preprocess_text("the and or but is was")
        assert result.strip() == ""

    def test_whitespace_collapsed(self):
        """Multiple spaces should collapse to single spaces."""
        result = preprocess_text("hello    world   test")
        assert "  " not in result

    def test_returns_string(self):
        """Output should always be a string."""
        result = preprocess_text("Some article text here about federal policy")
        assert isinstance(result, str)


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — TF-IDF vectorization
# ════════════════════════════════════════════════════════════════

class TestTfidfVectorization:
    """Test TF-IDF vectorization behavior on synthetic documents."""

    def setup_method(self):
        """Create a small synthetic corpus for testing."""
        self.corpus = [
            "federal reserve interest rates monetary policy",
            "trump senate congress republican democrat vote",
            "nfl football touchdown game quarterback season",
            "film movie actor director oscar award",
            "climate change carbon emissions environment",
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def test_matrix_shape(self):
        """Matrix should have one row per document."""
        assert self.matrix.shape[0] == len(self.corpus)

    def test_matrix_is_sparse(self):
        """Output should be a sparse matrix."""
        assert sp.issparse(self.matrix)

    def test_values_non_negative(self):
        """All TF-IDF values should be >= 0."""
        assert self.matrix.min() >= 0

    def test_distinctive_word_scores_high(self):
        """A word unique to one document should score higher in that document."""
        feature_names = self.vectorizer.get_feature_names_out()
        if "quarterback" in feature_names:
            idx = list(feature_names).index("quarterback")
            # Should score highest in doc 2 (football)
            scores = self.matrix[:, idx].toarray().flatten()
            assert scores[2] == scores.max(), \
                "quarterback should score highest in the football document"

    def test_bigrams_in_vocabulary(self):
        """With ngram_range=(1,2), bigrams should be in the vocabulary."""
        vocab = self.vectorizer.vocabulary_
        bigrams = [k for k in vocab if " " in k]
        assert len(bigrams) > 0, "No bigrams found in vocabulary"

    def test_transform_new_document(self):
        """Transforming a new document should produce a vector of the right size."""
        new_doc = ["stock market crash financial crisis"]
        vec = self.vectorizer.transform(new_doc)
        assert vec.shape == (1, self.matrix.shape[1])

    def test_unseen_word_is_zero(self):
        """A word not in the training vocabulary should have score 0."""
        new_doc = ["zzzzunknownword12345"]
        vec = self.vectorizer.transform(new_doc)
        # All values should be 0 since no words are in the vocabulary
        assert vec.nnz == 0, "Unseen words should produce all-zero vector"


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — classifier behavior
# ════════════════════════════════════════════════════════════════

class TestClassifierBehavior:
    """Test Logistic Regression classifier on synthetic data."""

    def setup_method(self):
        """Train a tiny LR model on a synthetic 3-class problem."""
        np.random.seed(42)
        # 3 classes, 20 features, 60 samples
        X_class0 = np.random.randn(20, 20) + np.array([3] * 20)
        X_class1 = np.random.randn(20, 20) + np.array([-3] * 20)
        X_class2 = np.random.randn(20, 20) + np.array([0] * 20)
        self.X = sp.csr_matrix(np.vstack([X_class0, X_class1, X_class2]))
        self.y = np.array([0]*20 + [1]*20 + [2]*20)

        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.model.fit(self.X, self.y)

    def test_model_has_correct_classes(self):
        """Model should have learned 3 classes."""
        assert len(self.model.classes_) == 3

    def test_predictions_are_valid_classes(self):
        """All predictions should be one of the known class labels."""
        preds = self.model.predict(self.X)
        assert set(preds).issubset(set(self.model.classes_))

    def test_probabilities_sum_to_one(self):
        """predict_proba output should sum to 1.0 per row."""
        probs = self.model.predict_proba(self.X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_high_accuracy_on_separable_data(self):
        """Model should achieve >80% on clearly separable synthetic data."""
        from sklearn.metrics import accuracy_score
        preds = self.model.predict(self.X)
        acc = accuracy_score(self.y, preds)
        assert acc > 0.80, f"Accuracy too low on separable data: {acc:.2f}"

    def test_coef_shape(self):
        """Coefficient matrix should be (n_classes, n_features)."""
        assert self.model.coef_.shape == (3, 20)


# ════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — trained model artifacts
# ════════════════════════════════════════════════════════════════

@requires_model
class TestTrainedModelArtifacts:
    """Tests that the real trained model loads and behaves correctly."""

    def setup_method(self):
        self.model = pickle.load(open(MODELS_DIR / "logistic_regression.pkl", "rb"))
        self.le    = pickle.load(open(MODELS_DIR / "label_encoder.pkl", "rb"))
        self.vec   = pickle.load(open(TFIDF_VECTORIZER_PATH, "rb"))

    def test_model_loads(self):
        """Model, label encoder, and vectorizer should load without error."""
        assert self.model is not None
        assert self.le is not None
        assert self.vec is not None

    def test_model_has_expected_features(self):
        """Model should have been trained on 50,000 TF-IDF features."""
        assert self.model.coef_.shape[1] == 50_000

    def test_inference_on_new_article(self):
        """Model should return a valid prediction for a brand-new article."""
        article = "The Federal Reserve raised interest rates by 25 basis points."
        vec = self.vec.transform([article])
        pred = self.model.predict(vec)
        assert len(pred) == 1
        assert pred[0] in self.model.classes_

    def test_inference_returns_probability(self):
        """predict_proba should return probabilities summing to 1."""
        article = "The Senate voted on the new healthcare bill today."
        vec = self.vec.transform([article])
        probs = self.model.predict_proba(vec)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_label_encoder_roundtrip(self):
        """LabelEncoder should correctly encode and decode class labels."""
        original_labels = self.le.classes_[:5]
        encoded = self.le.transform(original_labels)
        decoded = self.le.inverse_transform(encoded)
        np.testing.assert_array_equal(original_labels, decoded)

    def test_political_article_predicts_political_topic(self):
        """A clearly political article should predict a politics-adjacent topic.

        We look up political topic IDs dynamically from the topic_info CSV
        rather than hardcoding them, since BERTopic topic IDs vary by run.
        """
        topic_info_path = Path("data/processed/topic_info.csv")
        if not topic_info_path.exists():
            pytest.skip("topic_info.csv not found")

        topic_info = pd.read_csv(topic_info_path)
        # Political keywords to look for in topic names
        political_keywords = {"trump", "senate", "congress", "republican",
                              "democrat", "clinton", "obama", "president",
                              "sanders", "cruz", "rubio", "fbi", "gorsuch"}
        political_ids = set()
        for _, row in topic_info.iterrows():
            name_lower = str(row.get("Name", "")).lower()
            if any(kw in name_lower for kw in political_keywords):
                political_ids.add(row["Topic"])

        article = "The president signed the new tax reform bill into law today at the White House."
        vec = self.vec.transform([article])
        pred_class = self.model.predict(vec)[0]
        assert pred_class in political_ids, \
            f"Expected a political topic (one of {political_ids}), got topic {pred_class}"
