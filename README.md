# Automatic Emotion Detection in Text — NLP Pipeline

##  Overview
Multi-class text classification project detecting 28 emotions 
from Reddit comments using classical NLP and ML techniques.

## Dataset
- **Source**: [GoEmotions — Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- **Size**: ~54,000 Reddit comments · 28 emotion classes · 3 splits (Train/Dev/Test)
- **Challenge**: Strong class imbalance (admiration >4,400 vs remorse <600)

##  Pipeline
1. **EDA** — Class distribution, text length analysis, imbalance detection
2. **Preprocessing** — Lowercasing, special character removal, stopword removal, lemmatization
3. **Vectorisation** — TF-IDF (max_features=20,000`, `ngram_range=(1,2))
4. **Modelling** — 5 classifiers with class_weight='balanced'
5. **Evaluation** — F1-macro (preferred metric for imbalanced data), precision, recall

##  Results

 Model  Dev Accuracy  F1-Macro 

**SGD Classifier**   **0.398**  **0.393** 

 Logistic Regression  0.391  0.374 

 SVM (LinearSVC)  0.384  0.329 

 Passive Aggressive  0.349  0.293 

 Naive Bayes  0.336  0.078 

> F1-macro is used as the primary metric due to heavy class imbalance.

## Tech Stack
Python Scikit-learn NLTK Pandas Seaborn Matplotlib TF-IDF

