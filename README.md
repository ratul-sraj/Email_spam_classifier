# Email Spam Classifier

Classifies messages as **spam** or **ham** using classical NLP — text cleaning,
tokenising, stop-word removal, lemmatisation, TF-IDF — with four classifiers
compared on cross-validated and held-out accuracy.

**Data:** `spam_ham_dataset.csv` — 5,171 messages (3,672 ham / 1,499 spam),
included in this repo.

## Results

10-fold cross-validation on the training split, then evaluated on a held-out
20% test set (stratified), after removing length outliers:

| Model | Precision | Recall | F1 | CV accuracy | Test accuracy |
|---|---|---|---|---|---|
| Naive Bayes | 1.0000 | 0.7787 | 0.8756 | 0.9124 | 0.9405 |
| Random Forest | 0.9407 | 0.9754 | 0.9577 | 0.9799 | 0.9769 |
| K-Neighbours | 0.9737 | 0.9098 | 0.9407 | 0.9587 | 0.9692 |
| **SVC** | 0.9528 | 0.9918 | 0.9719 | **0.9871** | **0.9846** |

SVC wins at 98.5% test accuracy. Naive Bayes has perfect precision on spam but
misses 22% of it (recall 0.78) — it is the conservative model.

Note the train-accuracy column in the notebook: Random Forest and SVC both hit
1.0000 on the training data, which is ordinary overfitting on a sparse
26k-feature matrix; the cross-validated and test figures are what matter.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook EmailSpam.ipynb
```

The notebook loads the CSV by relative path, so it runs from a fresh clone with
no configuration. It downloads the four NLTK corpora it needs
(`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) on first run.

After training it saves `spam_classifier.joblib` (model + fitted vectoriser),
and exposes a `predict_message()` helper:

```python
predict_message("Congratulations! You have won a FREE 1000 pound prize.")
# -> ('spam', 'margin=+1.23')
```

## Layout

```
EmailSpam.ipynb            the analysis, executed with outputs saved
spam_ham_dataset.csv       the dataset
requirements.txt
```

## Method notes

- **Cleaning** — strip non-letters, lowercase, collapse whitespace.
- **Preprocessing** — word tokenise, drop English stop words, lemmatise verbs.
- **Vectorising** — TF-IDF over the joined tokens (26,239 features).
- **Model comparison** — Naive Bayes, Random Forest, K-Neighbours, SVC, each
  cross-validated before the held-out evaluation.
- **Scoring** — precision/recall/F1 are reported for the *spam* class, which is
  the class that matters, rather than averaged over both.

**Limitation:** the dataset is a single corpus of Enron-style mail. A model
trained here will not transfer unchanged to a different mail mix, and there is
no drift monitoring — a real deployment would need re-training on live traffic.
