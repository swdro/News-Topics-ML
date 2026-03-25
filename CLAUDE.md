# Project Context: News Article Topic Modeling & Classification Pipeline

## Who I Am
- Software Engineer currently working at the LA Department of Water and Power (LADWP)
- Previously worked at the City of Los Angeles Finance Department
- Background: Full-stack development (Angular, Spring Boot, Java, TypeScript, PostgreSQL, Oracle SQL)
- Python experience level: Beginner — I know Python syntax but have not built ML pipelines before
- Machine: Mac with Apple Silicon (M1/M2/M3) — use MPS acceleration where applicable
- Currently pursuing M.S. Computer Science at Georgia Tech

## Why I'm Building This
I am building this project to:
1. Transition into private sector / tech company roles that require ML engineering skills
2. Target roles focused on ML pipelines, data engineering, and AI solutions
3. Have a concrete, impressive project to discuss in interviews
4. Demonstrate hands-on ML experience to supplement my academic coursework

## Project Goal
Build a production-grade, end-to-end News Article Topic Modeling and Classification Pipeline using a hybrid unsupervised + supervised approach.

**Interview narrative:** "I wanted to explore how ML can automatically discover and classify topics across large volumes of unstructured text — a problem directly relevant to enterprise AI and content intelligence systems."

---

## Dataset
**All The News (Kaggle)**
- 200k+ real news articles from multiple outlets
- Download from: https://www.kaggle.com/datasets/snapcrack/all-the-news
- Contains: article title, content, publication, author, date
- Why: Large enough to feel real, varied enough for interesting topic discovery, multiple outlets enable cross-publication analysis

---

## Hybrid Approach: Two Phases

### Phase 1: Simplified Hybrid (Build This First)
**BERTopic (unsupervised topic discovery) → TF-IDF + Logistic Regression (supervised classifier)**

Steps:
1. Ingest and clean raw articles into SQLite
2. Run BERTopic to discover topics automatically
3. Use BERTopic topic assignments as labels
4. Engineer TF-IDF features from article text
5. Train Logistic Regression classifier to predict topics on new articles
6. Track all experiments with MLflow
7. Validate data quality with Great Expectations or Pydantic
8. Write Pytest tests for pipeline components
9. Set up GitHub Actions CI/CD
10. Document everything in README + model card

### Phase 2: Full Hybrid (Upgrade After Phase 1 is Complete)
**BERTopic → DistilBERT fine-tuning (transformer-based classifier)**

Steps (after Phase 1):
1. Replace Logistic Regression classifier with DistilBERT fine-tuning
2. Use Hugging Face Transformers library
3. Leverage MPS acceleration on Apple Silicon
4. Compare DistilBERT vs Logistic Regression metrics in MLflow
5. Update model card with new results

**Do NOT start Phase 2 until Phase 1 is fully working and tested.**

---

## Tech Stack

### Core
- Python 3.10+
- Pandas, NumPy
- SQLite + DuckDB (data storage and SQL feature queries)

### ML Models
- BERTopic (unsupervised topic discovery)
- Scikit-learn: TF-IDF vectorizer, Logistic Regression, Linear Regression (baseline)
- XGBoost (optional challenger model)
- SHAP (model explainability)
- Phase 2: Hugging Face Transformers (DistilBERT)

### Experiment Tracking
- MLflow (log every run — models, metrics, parameters, artifacts)

### Data Quality
- Great Expectations or Pydantic (schema validation, null checks, text length checks)

### Testing
- Pytest (data quality checks, pipeline component tests)

### CI/CD
- GitHub Actions (auto-run pytest on every push to main)

### Documentation
- README.md (project overview, setup instructions, pipeline diagram)
- model_card.md (intended use, training data, evaluation metrics, limitations)
- Jupyter Notebook (EDA and findings summary)

---

## Folder Structure
```
news-topic-pipeline/
├── data/
│   ├── raw/                        # raw Kaggle CSV files
│   └── processed/                  # cleaned, feature-engineered data
├── src/
│   ├── ingest.py                   # load CSV → SQLite, basic cleaning
│   ├── transform.py                # text preprocessing, feature engineering
│   ├── topic_model.py              # BERTopic unsupervised topic discovery
│   ├── train.py                    # TF-IDF + Logistic Regression, MLflow logging
│   ├── evaluate.py                 # metrics, SHAP explainability
│   ├── predict.py                  # inference on new articles
│   └── distilbert_train.py         # Phase 2 only: DistilBERT fine-tuning
├── tests/
│   ├── test_data_quality.py        # schema, nulls, text length checks
│   └── test_pipeline.py            # pipeline component tests
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: run pytest on push
├── notebooks/
│   └── eda.ipynb                   # exploratory analysis + findings summary
├── mlruns/                         # MLflow experiment logs (auto-generated)
├── model_card.md                   # model documentation
├── requirements.txt                # all dependencies
├── CLAUDE.md                       # this file
└── README.md                       # project overview and setup instructions
```

---

## Hour-by-Hour Build Plan (Phase 1)

| Hour | Task |
|------|------|
| 1 | Set up repo, virtual environment, install dependencies, download dataset |
| 2 | `ingest.py` — load CSV into SQLite, basic null/duplicate cleaning |
| 3 | SQL-based EDA in `eda.ipynb` — publication distribution, article length, missing data |
| 4 | `transform.py` — text preprocessing (lowercase, remove punctuation, stopwords) + TF-IDF feature engineering |
| 5 | `topic_model.py` — run BERTopic, inspect discovered topics, assign topic labels |
| 6 | `train.py` — baseline Logistic Regression → XGBoost challenger, log all runs in MLflow |
| 7 | `evaluate.py` — accuracy/F1/precision/recall, SHAP feature importance |
| 8 | Pytest data quality checks + GitHub Actions CI/CD setup |
| 9 | README, model card, findings summary in notebook |
| 10 | Polish, final push to GitHub |

---

## Key Interview Talking Points (Keep These in Mind While Building)

| Requirement | What to Say |
|---|---|
| ML model training + validation | "I compared a Logistic Regression baseline against XGBoost, evaluated on F1 and accuracy, and tracked every experiment run in MLflow" |
| Feature engineering | "I engineered TF-IDF features from raw article text using SQL queries against a SQLite database for feature extraction" |
| Unsupervised + supervised | "I used BERTopic to discover topics without any labels, then used those discovered topics to train a supervised classifier — a hybrid approach" |
| Data quality checks | "I defined schema contracts and null checks using Great Expectations that run automatically in CI before any training job" |
| CI/CD | "Every push to main triggers pytest via GitHub Actions — the pipeline can't proceed without passing data quality gates" |
| Explainability | "I used SHAP to surface which words and features drove classification decisions — important for communicating results to non-technical stakeholders" |
| Documentation | "I wrote a model card documenting intended use, training data, evaluation metrics, and known limitations" |
| Phase 2 upgrade | "After validating the baseline, I upgraded the classifier to a fine-tuned DistilBERT model using Hugging Face Transformers, leveraging MPS acceleration on Apple Silicon" |

---

## Important Guidelines for Claude During This Session

1. **Explain the why** behind every ML concept and code decision — I need to understand it well enough to discuss it in interviews, not just run it
2. **Go step by step** — complete one file fully before moving to the next
3. **Don't skip tests** — Pytest and data quality checks are required, not optional
4. **MLflow logging is mandatory** — every model training run must be logged
5. **Phase 1 must be fully working before touching Phase 2** — do not introduce DistilBERT until I confirm Phase 1 is complete
6. **Remind me to commit to GitHub** after each major milestone
7. **Flag any Apple Silicon specific setup steps** — especially for BERTopic and Transformers
8. **Keep code beginner-friendly** — clear variable names, inline comments explaining what each block does

---

## Resume Bullet This Project Will Earn

> "Built a production-grade news article topic modeling and classification pipeline in Python — encompassing SQL-based feature engineering, BERTopic unsupervised topic discovery, TF-IDF + Logistic Regression and XGBoost model comparison tracked via MLflow, SHAP explainability, automated data-quality validation, and CI/CD via GitHub Actions — later upgraded to a fine-tuned DistilBERT classifier leveraging Apple Silicon MPS acceleration."

---

## How to Start This Session
Say: "I'm ready to start. Let's begin with Hour 1 — setting up the repo, virtual environment, and downloading the dataset."
