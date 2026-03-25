# News Article Topic Modeling & Classification Pipeline

A production-grade, end-to-end ML pipeline that **automatically discovers topics** across 142,000 real news articles and trains a **supervised classifier** to categorize new articles in milliseconds.

Built as a portfolio project demonstrating ML engineering skills: data ingestion, feature engineering, unsupervised topic discovery, supervised classification, experiment tracking, explainability, data quality validation, and CI/CD.

---

## Results

| Model | Accuracy | F1 (Macro) |
|-------|----------|------------|
| Logistic Regression | **89.8%** | **88.3%** |
| XGBoost | 85.9% | 83.9% |

- Topics discovered automatically from 142,000 articles across 15 news publications
- High-confidence labeled articles retained for classifier training
- All experiments tracked and compared in **MLflow**

To see the current topic list (always reflects the latest model run):

```bash
python -m src.describe_topics
```

---

## Architecture

```
Raw Data (638MB CSVs)
        │
        ▼
┌─────────────────┐
│   ingest.py     │  Load 3 CSVs → clean → write to SQLite
│                 │  • Drop null content/title, exact duplicates
└────────┬────────┘  • 142,036 articles → articles.db
         │
         ▼
┌─────────────────┐
│  transform.py   │  Text preprocessing + TF-IDF feature engineering
│                 │  • NLTK stopwords, lowercase, remove punctuation
└────────┬────────┘  • TF-IDF: 50k features, bigrams, sublinear_tf
         │
         ▼
┌─────────────────┐
│ topic_model.py  │  BERTopic unsupervised topic discovery
│                 │  • all-MiniLM-L6-v2 embeddings (MPS-accelerated)
└────────┬────────┘  • 73 topics discovered, 87k articles labeled
         │
         ▼
┌─────────────────┐
│   train.py      │  Supervised classifier training + MLflow logging
│                 │  • Logistic Regression (baseline): F1=0.883
└────────┬────────┘  • XGBoost (challenger):          F1=0.839
         │
         ▼
┌─────────────────┐
│  evaluate.py    │  Metrics, confusion matrix, SHAP explainability
│                 │  • Per-class precision/recall/F1
└────────┬────────┘  • Word importance plots per topic
         │
         ▼
┌─────────────────┐
│   predict.py    │  Inference on new articles
│                 │  • Classify any article in milliseconds
└─────────────────┘
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data storage | SQLite, DuckDB |
| Text features | scikit-learn TF-IDF |
| Topic modeling | BERTopic, sentence-transformers, UMAP, HDBSCAN |
| Classifiers | Logistic Regression, XGBoost |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| Data quality | Pydantic |
| Testing | Pytest |
| CI/CD | GitHub Actions |

---

## Dataset

**All The News** — 200k+ real news articles from 15 US publications (2016–2017).
Source: [Kaggle — snapcrack/all-the-news](https://www.kaggle.com/datasets/snapcrack/all-the-news)

Publications: Breitbart, New York Post, NPR, CNN, Washington Post, Reuters, Guardian, New York Times, The Atlantic, Business Insider, National Review, Talking Points Memo, Vox, BuzzFeed News, Fox News.

---

## Setup

### Prerequisites
- Python 3.10+
- [Homebrew](https://brew.sh) (macOS)
- Kaggle account (for dataset download)

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd news-topic-pipeline
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Apple Silicon: install OpenMP (required by XGBoost)

```bash
brew install libomp
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 5. Download the dataset

1. Go to [Kaggle — All The News](https://www.kaggle.com/datasets/snapcrack/all-the-news)
2. Download and unzip — you'll get `articles1.csv`, `articles2.csv`, `articles3.csv`
3. Move all three files into `data/raw/`

---

## Running the Pipeline

Run each step in order:

```bash
# Step 1: Ingest raw CSVs into SQLite
python -m src.ingest

# Step 2: Text preprocessing + TF-IDF feature engineering
python -m src.transform

# Step 3: BERTopic unsupervised topic discovery (~10-15 min)
python -m src.topic_model

# Step 4: Train classifiers + log to MLflow (~20 min for XGBoost)
python -m src.train

# Step 5: Evaluate model + SHAP explainability
python -m src.evaluate

# Step 6: Predict topic for a new article
python -m src.predict --text "The Federal Reserve raised interest rates today."
```

### View experiment results in MLflow

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

---

## Running Tests

```bash
# Run all tests (unit + integration)
pytest tests/ -v

# Unit tests only (no data files needed — matches CI behavior)
pytest tests/ -v -m "not skipif"

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Project Structure

```
news-topic-pipeline/
├── data/
│   ├── raw/                        # raw Kaggle CSV files (not committed)
│   └── processed/                  # SQLite db, parquet, TF-IDF matrix, models
├── src/
│   ├── ingest.py                   # CSV → SQLite, cleaning
│   ├── transform.py                # text preprocessing, TF-IDF
│   ├── topic_model.py              # BERTopic unsupervised topic discovery
│   ├── train.py                    # train + compare classifiers, MLflow logging
│   ├── evaluate.py                 # metrics, SHAP, confusion matrix
│   └── predict.py                  # inference on new articles
├── tests/
│   ├── test_data_quality.py        # Pydantic schema + data integration tests
│   └── test_pipeline.py            # pipeline unit tests
├── notebooks/
│   ├── eda.ipynb                   # exploratory analysis
│   ├── explain_01_data_pipeline.ipynb
│   ├── explain_02_tfidf.ipynb
│   ├── explain_03_bertopic.ipynb
│   ├── explain_04_training.ipynb
│   └── explain_05_evaluation.ipynb
├── docs/
│   └── architecture_decisions.md   # ADRs for all design decisions
├── .github/workflows/ci.yml        # GitHub Actions: run pytest on push
├── model_card.md                   # model documentation
├── requirements.txt
└── README.md
```

---

## Notebooks

The `explain_*.ipynb` notebooks are designed for learning and interview preparation — they explain every ML concept and design decision from first principles, with toy examples before real data.

| Notebook | Covers |
|----------|--------|
| `explain_01_data_pipeline` | SQLite, cleaning strategy, missing data |
| `explain_02_tfidf` | Preprocessing, TF-IDF math, bigrams |
| `explain_03_bertopic` | Embeddings, UMAP, HDBSCAN, topic discovery |
| `explain_04_training` | LR vs XGBoost, train/test split, MLflow |
| `explain_05_evaluation` | Precision/recall/F1, confusion matrix, SHAP |

To run: `jupyter notebook` then select the **Python (news-pipeline)** kernel.

---

## CI/CD

Every push to `main` triggers GitHub Actions to run the unit test suite. Integration tests (requiring data files) are skipped in CI but run locally.

```
Push to main
    │
    ▼
GitHub Actions
    │
    ├── Install dependencies (cached)
    ├── Download NLTK stopwords
    └── pytest tests/ --cov=src
            │
            ├── 48 tests
            └── Coverage report uploaded as artifact
```

---

## Architecture Decision Records

All design decisions — why SQLite over PostgreSQL, why TF-IDF over raw counts, why LR beats XGBoost here, why SHAP, etc. — are documented in [`docs/architecture_decisions.md`](docs/architecture_decisions.md) using the ADR format.
