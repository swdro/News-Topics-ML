# Architecture Decision Records — News Topic Pipeline

Architecture Decision Records (ADRs) document *why* key decisions were made, not just *what* was built. Each record captures the context, the options considered, the chosen approach, and the tradeoffs accepted. This is a standard practice at software and ML engineering teams.

---

## ADR-001 — Storage: SQLite over raw CSV re-reads

**Phase:** 1 (Ingest)
**Status:** Accepted

### Context

The raw dataset is ~638MB across three CSV files. Every pipeline step (transform, topic model, train, evaluate) needs access to the article data. We needed to decide how downstream steps would access it.

### Options Considered


| Option                    | Pros                                                  | Cons                                                     |
| ------------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| Re-read CSVs in each step | Simple, no extra tooling                              | Slow (638MB per read), no querying, error-prone          |
| SQLite database           | Fast queries, single file, portable, no server needed | Slightly more setup                                      |
| PostgreSQL                | Production-grade, concurrent access                   | Overkill for a local pipeline, requires a running server |


### Decision

Use **SQLite** via Python's built-in `sqlite3` module. The entire dataset is stored in `data/processed/articles.db`.

### Consequences

- All downstream steps query specific columns via SQL rather than loading the full dataset into memory.
- An index on `id` makes article lookups fast.
- DuckDB (used for analytics queries in EDA) can query SQLite files directly.
- The `.db` file is excluded from git via `.gitignore` — it's a derived artifact, not source data.

---

## ADR-002 — Cleaning Strategy: Conservative Drops Only

**Phase:** 1 (Ingest)
**Status:** Accepted

### Context

During ingest, we needed a cleaning policy: how aggressively should we filter articles before storing them?

### Options Considered


| Option                                       | Pros                                 | Cons                                            |
| -------------------------------------------- | ------------------------------------ | ----------------------------------------------- |
| Aggressive filtering (length, date, author)  | Cleaner dataset immediately          | Removes articles that may be fine; hard to undo |
| Conservative (only nulls + exact duplicates) | Preserves data; more filtering later | Some noise remains                              |


### Decision

**Conservative approach at ingest** — only drop rows where `content` or `title` is null, and drop exact content duplicates. All other filtering (e.g. short articles) is deferred to `transform.py`.

### Consequences

- 534 duplicates and 2 null-title rows removed at ingest; 924 stub articles (<200 chars) removed later in transform.
- Separating concerns keeps `ingest.py` focused on loading and `transform.py` focused on quality — easier to debug and test independently.
- If the length threshold changes, only `transform.py` needs to be updated.

---

## ADR-003 — Missing Data: Author and Date Fields

**Phase:** 1 (Ingest)
**Status:** Accepted

### Context

The dataset has ~~15,800 missing `author` values (~~11%) and ~~2,600 missing `date` values (~~2%).

### Options Considered


| Option                             | Pros                  | Cons                                |
| ---------------------------------- | --------------------- | ----------------------------------- |
| Drop rows with missing author/date | Cleaner metadata      | Loses 11% of articles unnecessarily |
| Impute (fill with "Unknown")       | Preserves rows        | Creates fake data                   |
| Keep as-is (null)                  | Honest representation | Downstream steps must handle nulls  |


### Decision

**Keep nulls as-is.** `author` and `date` are metadata fields, not features used in topic modeling or classification. The model only needs `content` and `title`.

### Consequences

- No article data is lost due to missing metadata.
- `evaluate.py` and `eda.ipynb` must handle nulls when grouping by author or date — handled with `dropna()` or `errors='coerce'` where needed.

---

## ADR-004 — Text Preprocessing: NLTK Stopwords over a Hand-Rolled List

**Phase:** 2 (Transform)
**Status:** Accepted

### Context

Stopword removal is a standard NLP preprocessing step. We needed to decide the source of the stopword list.

### Options Considered


| Option                            | Size        | Pros                                         | Cons                                     |
| --------------------------------- | ----------- | -------------------------------------------- | ---------------------------------------- |
| Hand-rolled list                  | ~100 words  | No extra dependency                          | Incomplete; misses inflected forms       |
| NLTK `stopwords.words("english")` | 179 words   | Comprehensive, maintained, industry standard | Adds `nltk` dependency                   |
| spaCy                             | ~300+ words | Very thorough                                | Heavy dependency (~200MB model download) |


### Decision

**NLTK stopwords.** Added `nltk>=3.8.0` to `requirements.txt` and downloaded the `stopwords` and `punkt_tab` corpora at setup time.

### Consequences

- 179-word list covers inflected forms ("ourselves", "yourselves") and contractions ("wouldn't") that a hand-rolled list would miss.
- Adds a small dependency but NLTK is lightweight and widely used in the Python NLP ecosystem.
- Must call `nltk.download('stopwords')` once after installing — documented in README setup steps.

---

## ADR-005 — Feature Engineering: TF-IDF over Bag-of-Words or Raw Counts

**Phase:** 2 (Transform)
**Status:** Accepted

### Context

To train a supervised classifier, article text must be converted to a numeric feature matrix. We needed to choose a vectorization strategy.

### Options Considered


| Option                            | Pros                                              | Cons                                                      |
| --------------------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| Raw word counts (CountVectorizer) | Simple, interpretable                             | Biased toward long articles; common words dominate        |
| TF-IDF                            | Penalizes common words, rewards distinctive words | Slightly more complex                                     |
| Word embeddings (Word2Vec, GloVe) | Captures semantic meaning                         | High dimensionality, slower, harder to explain            |
| Transformer embeddings (BERT)     | Best semantic representation                      | Very slow on 141k articles without GPU; saved for Phase 2 |


### Decision

**TF-IDF with `sublinear_tf=True` and bigrams (`ngram_range=(1,2)`).** Using scikit-learn's `TfidfVectorizer` with `max_features=50,000`.

### Rationale for each parameter:

- `sublinear_tf=True` — applies `log(1 + tf)` instead of raw term frequency, dampening the effect of words that repeat many times in one article.
- `ngram_range=(1, 2)` — captures two-word phrases like "interest rate", "white house", "climate change" that carry more specific topic signal than either word alone.
- `max_features=50,000` — caps vocabulary to the 50k most informative terms; reduces memory and speeds up training.
- `min_df=5` — drops words appearing in fewer than 5 articles (likely typos or proper nouns too rare to generalize).
- `max_df=0.95` — drops words in more than 95% of articles (too common to discriminate between topics).

### Consequences

- Output is a sparse matrix: 141,112 articles × 50,000 features, 99.4% zeros.
- Sparsity is expected and handled efficiently — `scipy.sparse.save_npz` stores only non-zero values.
- The fitted vectorizer is saved to disk (`tfidf_vectorizer.pkl`) so new articles at inference time are transformed identically to training data.
- TF-IDF cannot capture word order or semantic similarity ("good" ≠ "great") — acceptable for Phase 1; Phase 2 upgrades to DistilBERT embeddings.

---

## ADR-006 — Title + Content Concatenation

**Phase:** 2 (Transform)
**Status:** Accepted

### Context

Each article has both a `title` and `content` field. We needed to decide whether to vectorize them separately, use only one, or combine them.

### Options Considered


| Option                                  | Pros                                         | Cons                                      |
| --------------------------------------- | -------------------------------------------- | ----------------------------------------- |
| Content only                            | Simpler                                      | Wastes the dense signal in titles         |
| Title only                              | Very fast                                    | Loses the majority of text                |
| Separate features for title and content | Preserves field structure                    | Doubles feature space; more complex model |
| Concatenate title + content             | Simple; title keywords get a frequency boost | Title words counted twice                 |


### Decision

**Concatenate title + content** before preprocessing (`title + " " + content`).

### Rationale

Titles are manually written to summarize the article's topic — they're dense with signal. Prepending the title gives those keywords a slightly higher term frequency in the TF-IDF vector, which is a desirable bias for topic classification.

### Consequences

- Title words are counted in both the title and (often) the content, giving them a modest TF-IDF boost.
- No increase in feature space — the combined text is treated as one document.

---

## ADR-007 — Artifact Persistence Strategy

**Phase:** 2 (Transform)
**Status:** Accepted

### Context

The TF-IDF transformation of 141k articles takes meaningful compute time. We needed a strategy for passing features between pipeline steps.

### Options Considered


| Option                                       | Pros                                          | Cons                                                 |
| -------------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| Re-run transform in every downstream step    | No files to manage                            | Wasteful; ~30-60 seconds each time                   |
| Save to CSV                                  | Simple                                        | CSVs can't represent sparse matrices; huge file size |
| Save matrix as `.npz` + vectorizer as `.pkl` | Efficient sparse format; exact reconstruction | Two separate files to manage                         |
| MLflow artifact storage                      | Integrated with experiment tracking           | Overkill for intermediate features                   |


### Decision

**Save three artifacts to `data/processed/`:**

1. `articles_clean.parquet` — cleaned DataFrame with `text_clean` column (Parquet is columnar and fast for pandas).
2. `tfidf_matrix.npz` — sparse matrix in scipy's compressed format.
3. `tfidf_vectorizer.pkl` — fitted vectorizer, required at inference time to transform unseen articles identically.

### Consequences

- `train.py` loads in seconds rather than re-running vectorization.
- The fitted vectorizer being saved is critical for production correctness: if a new article comes in at inference time, it must be transformed using the *same* vocabulary and IDF weights as training data.
- All three files are excluded from git (derived artifacts) but should be tracked as MLflow artifacts in a production system.

---

## ADR-008 — Topic Modeling: BERTopic over LDA or NMF

**Phase:** 3 (Topic Modeling)
**Status:** Accepted

### Context
Unsupervised topic discovery requires choosing an algorithm. The three most common approaches are LDA, NMF, and BERTopic.

### Options Considered
| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| LDA (Latent Dirichlet Allocation) | Probabilistic bag-of-words | Fast, well-understood, sklearn built-in | Topics are distributions over words — often hard to interpret; ignores word order |
| NMF (Non-negative Matrix Factorization) | Linear algebra on TF-IDF | Fast, interpretable keyword lists | Still bag-of-words; misses semantic similarity |
| BERTopic | Transformer embeddings + HDBSCAN | Semantically coherent topics; sharp keyword labels; handles noise natively via topic -1 | Slower; requires more memory |

### Decision
**BERTopic** using `sentence-transformers/all-MiniLM-L6-v2` as the embedding model.

### Rationale
BERTopic produces topics that are meaningfully coherent because it clusters articles by *semantic similarity* rather than just word overlap. "car" and "automobile" land in the same cluster. The keyword labels it generates (e.g. `eu_brexit_britain_european`) are sharp enough to use directly as human-readable class names — a key advantage when these labels become supervised classifier targets.

### Consequences
- 19 distinct topics discovered on the 20k sample, covering politics, sports, immigration, tech, health, and international affairs.
- Topic -1 (outliers) captured 53,720 articles (~38%) that didn't fit any cluster — these are dropped before supervised training. This is expected with HDBSCAN.
- 87,392 articles retained with valid topic labels for classifier training.

---

## ADR-009 — BERTopic: Fit on Sample, Transform Full Dataset

**Phase:** 3 (Topic Modeling)
**Status:** Accepted

### Context
Embedding all 141k articles end-to-end before UMAP/HDBSCAN would take 30-60+ minutes and risk memory issues. We needed a scalable strategy.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Fit + transform all 141k | Single pass, no sampling bias | Very slow; memory intensive for UMAP |
| Fit on sample, transform rest | Fast; standard production pattern | Sample must be representative |
| Random sample | Simple | May under-represent minority publications |
| Stratified sample by publication | Representative across all outlets | Slightly more code |

### Decision
**Fit BERTopic on a stratified sample of 20,000 articles, then call `.transform()` on all 141,112.** Stratification is by publication to ensure all 15 outlets are proportionally represented.

### Consequences
- Fit step (~3-8 min on MPS) discovers topic structure from a representative slice.
- `.transform()` skips UMAP/HDBSCAN re-fitting — it just embeds remaining articles and finds their nearest cluster centroid. Much faster.
- `prediction_data=True` must be set on HDBSCAN to enable `.transform()` — included in the model config.
- Topics discovered may not perfectly represent articles in the non-sampled portion, but in practice the topics are stable across samples of this size.

---

## ADR-010 — BERTopic: Embedding Model Choice

**Phase:** 3 (Topic Modeling)
**Status:** Accepted

### Context
BERTopic requires a sentence embedding model. Several options exist with different speed/quality tradeoffs.

### Options Considered
| Model | Embedding Dim | Speed (MPS) | Quality |
|-------|--------------|-------------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast (~7 it/s) | Good — designed for semantic similarity |
| `all-mpnet-base-v2` | 768 | ~2x slower | Better quality, larger model |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Similar | Multilingual — unnecessary here |
| `distilbert-base-nli-mean-tokens` | 768 | Slower | Older model, worse benchmarks |

### Decision
**`all-MiniLM-L6-v2`** — the de facto standard for BERTopic and sentence similarity tasks.

### Rationale
At 384 dimensions it's 2x faster than 768-dim models with only a small quality reduction. For topic discovery (not fine-grained semantic search), the quality difference is negligible. It's also the BERTopic default, meaning extensive community validation.

### Consequences
- Embedding runs at ~7-8 batches/sec on Apple Silicon MPS.
- 384-dim vectors fed into UMAP → reduced to 5 dims → HDBSCAN clustering.
- Model weights cached in `~/.cache/huggingface/` after first download.

---

## ADR-011 — Outlier Handling: Drop Topic -1 Articles

**Phase:** 3 (Topic Modeling)
**Status:** Accepted

### Context
HDBSCAN assigns topic -1 to articles that don't fit any cluster. 53,720 articles (~38%) received this label.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Drop topic -1 articles | Clean labels; no noise in training | Loses 38% of data |
| Assign to nearest topic | Retains all data | Forces articles into potentially wrong topics; noisy labels degrade classifier |
| Keep as a separate "miscellaneous" class | Retains data; honest label | "Misc" is not a useful class to predict |

### Decision
**Drop topic -1 articles** before supervised classifier training.

### Rationale
Noisy labels are more harmful to a classifier than a smaller dataset. A Logistic Regression trained on confidently-labeled examples will generalize better than one trained on 38% mislabeled noise. 87,392 articles with clean labels is still a large, robust training set.

### Consequences
- Training set reduced from 141k to 87k articles.
- Topic distribution across 19 classes is reasonably balanced (155–1,029 per topic).
- In a production system, outlier articles could be periodically reviewed and manually labeled to expand training data.

---

## ADR-012 — Baseline Model: Logistic Regression over Naive Bayes or SVM

**Phase:** 4 (Training)
**Status:** Accepted

### Context
For the supervised classifier baseline, several linear models are standard choices for text classification.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Naive Bayes (MultinomialNB) | Very fast, probabilistic output | Assumes feature independence (unrealistic for text); lower accuracy |
| Linear SVM (LinearSVC) | Strong text classifier, fast | No probability output; harder to interpret |
| Logistic Regression | Fast, probabilistic, interpretable weights, standard baseline | Assumes linear decision boundary |

### Decision
**Logistic Regression** with `C=5`, `solver='lbfgs'`, `multi_class='multinomial'`.

### Rationale
LR is the industry standard baseline for text classification. Its learned coefficients (one weight per word per class) are directly interpretable — you can inspect which words drive each topic prediction. It also outputs calibrated probabilities, which SHAP and evaluate.py depend on.

### Consequences
- Trains in under 60 seconds on 70k × 50k sparse matrix.
- Interpretable: `model.coef_[class_idx]` gives word importance scores per topic.
- `n_jobs=-1` parallelizes across all CPU cores.

---

## ADR-013 — Challenger Model: XGBoost

**Phase:** 4 (Training)
**Status:** Accepted

### Context
After establishing a baseline, we need a challenger to determine if a non-linear model improves topic classification.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Random Forest | Robust, parallelizable | Slow on 50k features; high memory |
| XGBoost | State-of-the-art on tabular data; `tree_method='hist'` handles sparse data | Slower than LR on high-dimensional sparse text; may not outperform LR |
| LightGBM | Faster than XGBoost on large datasets | Less name recognition in interviews; similar behavior |

### Decision
**XGBoost** with `tree_method='hist'`, `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`.

### Rationale
XGBoost is required by the project spec and is a well-known algorithm with strong interview recognition. `tree_method='hist'` bins continuous features into histograms for speed — critical when dealing with 50k features. `device='cpu'` is used because XGBoost's MPS (Apple Silicon GPU) support is immature.

### Consequences
- Expected to train slower than LR (~5-15 min vs <1 min).
- May not outperform LR — TF-IDF text features have linear structure that trees don't exploit well.
- Comparison logged in MLflow provides objective evidence for model selection.

---

## ADR-014 — Evaluation Metric: Macro F1 over Accuracy

**Phase:** 4 (Training)
**Status:** Accepted

### Context
Topic classes are imbalanced — the largest topic (police/crime) has ~6x more articles than the smallest. We needed a primary metric that reflects performance across all classes fairly.

### Options Considered
| Metric | Definition | Problem with imbalanced classes |
|--------|------------|--------------------------------|
| Accuracy | % of correct predictions | A model predicting only the majority class can achieve high accuracy |
| Weighted F1 | F1 averaged by class size | Still dominated by large classes |
| Macro F1 | F1 averaged equally across all classes | Each class counts equally regardless of size |

### Decision
**Macro F1** as the primary metric, with accuracy reported as secondary.

### Consequences
- A model that ignores small topics is penalized even if its overall accuracy is high.
- Both metrics are logged to MLflow so the full picture is visible.

---

## ADR-015 — Stratified Train/Test Split

**Phase:** 4 (Training)
**Status:** Accepted

### Context
Splitting 87k articles into 80% train / 20% test requires deciding whether to stratify by class label.

### Decision
**Stratified split** using `sklearn.model_selection.train_test_split(stratify=y)`.

### Rationale
Without stratification, small classes (155 articles) could end up with too few test examples to evaluate reliably. Stratification guarantees each class has exactly 20% of its articles in the test set.

### Consequences
- Reliable per-class F1 scores in the classification report.
- Reproducible with `random_state=42`.
