# Email Spam Classifier

Classifies messages as **spam** or **ham** using classical NLP — text cleaning,
tokenising, stop-word removal, lemmatisation, TF-IDF — with five classifiers
compared on cross-validated F1 and held-out accuracy.

**Data:** `spam_ham_dataset.csv` — 5,171 messages (3,672 ham / 1,499 spam),
included in this repo.

## Results

The primary metric is **F1 on the spam class**. 10-fold cross-validation on the
training split, then a held-out stratified 20% test set, after removing length
outliers:

| Model | Precision | Recall | F1 | CV F1 | Test accuracy |
|---|---|---|---|---|---|
| Naive Bayes | 1.0000 | 0.7787 | 0.8756 | 0.8056 | 0.9405 |
| Logistic Regression | 0.9488 | 0.9877 | 0.9679 | 0.9728 | 0.9824 |
| Random Forest | 0.9407 | 0.9754 | 0.9577 | 0.9631 | 0.9769 |
| K-Neighbours | 0.9737 | 0.9098 | 0.9407 | 0.9181 | 0.9692 |
| **SVC** | 0.9528 | 0.9918 | **0.9719** | **0.9765** | **0.9846** |

SVC wins on F1 at 0.9719. Logistic Regression is a close second — 0.9728 CV F1
and 0.9824 test accuracy — and trains in a fraction of the time, so it is the
better pick if the model has to be refitted regularly.

### Why F1 rather than accuracy

Spam is 27% of the corpus, so predicting "ham" for everything already scores
**0.7313** accuracy while catching no spam at all. Accuracy is inflated by the
majority class and hides the failure that actually matters.

Naive Bayes is the clearest case. Its **0.9405 accuracy** looks respectable, but
its F1 is only **0.8756**, because it misses 22% of spam (recall 0.7787). Ranked
by accuracy it sits near the others; ranked by F1 the gap is plain.

Precision, recall and F1 are all reported for the **spam** class
(`pos_label=1`), not averaged across both.

### The imbalance, and why nothing was resampled

Spam is **26.9%** of the corpus. That is mild — the usual trouble starts below
roughly 10–20% — and neither model is ignoring the minority class: both catch
about 99% of spam, so the classic failure mode is simply absent.

Applying the textbook remedy makes a mail filter worse here, so it is not used:

| Logistic Regression | Precision | Recall | F1 | Real email flagged spam |
|---|---|---|---|---|
| plain | 0.9488 | 0.9877 | 0.9679 | 13 |
| `class_weight="balanced"` | 0.8443 | 1.0000 | 0.9156 | **45** |

Recall is already 0.99, so extra class weight has nothing left to buy and only
spends precision — it **triples** the false positives. For a mail filter that is
the wrong trade: a false positive sends a real message to the spam folder, which
costs more than a missed spam sitting in the inbox.

Resampling is a poor fit for TF-IDF text in any case. SMOTE interpolates between
documents in feature space, producing vectors with non-integer TF-IDF weights
that correspond to no real message, and undersampling would discard real ham
from an already small 4,539-row corpus.

If the operating point ever needs moving, it is a **threshold** decision rather
than a resampling one. On logistic regression, moving the cut from 0.50 to 0.60
halves the false positives (13 → 6) for more missed spam (3 → 15). That choice
belongs to whoever owns the filter's cost model, not to the training script.

Note the train-accuracy column in the notebook: Random Forest and SVC both hit
1.0000 on the training data, which is ordinary overfitting on a sparse
26k-feature matrix; the cross-validated and held-out figures are what matter.

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
- **Model comparison** — Naive Bayes, Logistic Regression, Random Forest,
  K-Neighbours and SVC, each cross-validated by F1 before the held-out
  evaluation.
- **Scoring** — precision/recall/F1 are reported for the *spam* class, which is
  the class that matters, rather than averaged over both.

**Limitation:** the dataset is a single corpus of Enron-style mail. A model
trained here will not transfer unchanged to a different mail mix, and there is
no drift monitoring — a real deployment would need re-training on live traffic.
