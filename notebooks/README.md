# Notebook notes and findings

[`sms_spam_classifier_notebook.ipynb`](sms_spam_classifier_notebook.ipynb) is the
original exploratory and model-training work. It has not been edited as part of
the API setup.

## What the notebook does

1. Loads `data/spam.csv` with Latin-1 encoding.
2. Keeps the label and message columns, renames them to `target` and `text`, and
   encodes ham as `0` and spam as `1`.
3. Removes 403 duplicate rows, leaving 5,169 messages.
4. Explores message character, word, and sentence counts.
5. Lowercases and tokenizes text, removes non-alphanumeric tokens and English
   stop words, then applies Porter stemming.
6. Creates TF-IDF features with a maximum of 3,000 terms.
7. Compares multiple classifiers and exports the fitted TF-IDF vectorizer and
   Multinomial Naive Bayes model as pickle files.

## Main insights

- The cleaned data is imbalanced: 4,516 ham messages and 653 spam messages
  (about 87% ham and 13% spam).
- Spam messages are substantially longer on average: about 138 characters and
  28 words, compared with 70 characters and 17 words for ham.
- Multinomial Naive Bayes achieved approximately **97.10% accuracy** and
  **100% precision** on the notebook's test split, with confusion matrix
  `[[896, 0], [30, 108]]`.
- Bernoulli Naive Bayes had higher accuracy (about 98.36%) but slightly lower
  precision (about 99.19%). The notebook selected Multinomial Naive Bayes
  because avoiding false spam predictions was treated as the priority.
- A soft-voting ensemble reached about 97.97% accuracy and 98.35% precision,
  but the exported single model is smaller and simpler to serve.

## Important limitations

- The metrics come from one train/test split (`random_state=2`), not cross-validation.
- TF-IDF is fitted before the train/test split in the notebook. That leaks test-set
  vocabulary statistics into training, so the reported metrics may be mildly
  optimistic. A future retraining pass should split first and fit TF-IDF only on
  training data.
- The dataset is heavily imbalanced, so accuracy alone is not sufficient. The
  notebook correctly pays special attention to precision, but recall and F1
  should also be tracked for future model versions.
- The pickle artifacts require a compatible scikit-learn version. The API pins
  the artifact-producing version (`1.8.0`) in `pyproject.toml`.

Run the notebook from the project root if you want its relative
`data/spam.csv` path to resolve. Notebook dependencies used for experimentation
(for example pandas, seaborn, wordcloud, and XGBoost) are intentionally not part
of the lean API environment. Its final export cell writes `model.pkl` and
`vectorizer.pkl` to the project root; if you retrain without changing the
notebook, move those two trusted outputs into `artifacts/` before starting the
API.
