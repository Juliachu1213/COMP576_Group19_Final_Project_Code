📈 Predicting Stock Market Movements Using Deep Learning-Based Sentiment Analysis of Financial News

Chu-Yun Chu · Kelly Hung · Vivian Liu · Thomas Lin
COMP 576 · Deep Learning · Fall 2025

🔍 Overview

This repository contains the full implementation for our project exploring:

Can deep-learning-based sentiment signals predict next-day DJIA movement?

We build a three-stage pipeline:

Stage 1 — FinBERT Fine-Tuning
Train a financial-domain sentiment classifier using 4,000 labeled headlines.

Stage 2 — Daily Sentiment Aggregation
Apply FinBERT to 8 years of news headlines, compute daily sentiment features, and align them with next-day DJIA labels.

Stage 3 — Market Movement Prediction (MLP/LSTM/GRU)
Train sequence models on daily sentiment signals to predict Up/Down movement.

---

## 🔹 **Project Overview**

Our workflow consists of three stages:

1. **Stage 1 — Fine-tune FinBERT**
   Train a sentiment classifier using ~4,000 labeled financial news headlines.

2. **Stage 2 — Daily Sentiment Aggregation**
   Apply the classifier to daily news headlines (2008–2016) and compute a daily sentiment score aligned with next-day DJIA movement.

3. **Stage 3 — Prediction Models (not included here)**
   Train MLP / LSTM / GRU models using the aggregated sentiment time series to predict next-day DJIA direction.

This repository provides the code for **Stage 1** and **Stage 2**.

---

# ## 📁 **Files Included**

### **`finbert_model.py` — Stage 1: Fine-tuning FinBERT**

This script:

* Downloads the *Sentiment Analysis for Financial News* dataset (Kaggle)
* Preprocesses the headlines and labels
* Fine-tunes FinBERT (`yiyanghkust/finbert-tone`)
* Evaluates the model using accuracy, F1-score, and a confusion matrix
* Saves the fine-tuned model and tokenizer to Google Drive

**Output:**

* `/finbert_model/` — directory containing model weights and tokenizer files
* Confusion matrix printed to console
* Training logs and evaluation metrics

---

### **`daily_sentiment_aggregation.py` — Stage 2: Compute Daily Sentiment Score**

This script:

* Loads the fine-tuned FinBERT model from Stage 1
* Downloads the DJIA news dataset (`Combined_News_DJIA.csv`)
* Applies FinBERT to all 25 headlines per day
* Aggregates predictions into a **daily sentiment score** (mean sentiment)
* Aligns it with the *next-day* DJIA movement (Label column)
* Outputs a CSV with sentiment score + label

**Output:**

* `daily_sentiment_scores.csv`
  Contains: `Date, sentiment_score`
* `news_label_daily_sentiment_scores.csv`
  Contains: `Date, sentiment_score, Label`
  → Used as Stage 3 model input

---

# ## ▶️ **How to Run the Scripts**

### **1. Install Dependencies**

```
pip install transformers datasets accelerate kagglehub
```

---

### **2. Run Stage 1 — Fine-tune FinBERT**

```
python finbert_model.py
```

This will:

* Train FinBERT
* Print accuracy, F1, and confusion matrix
* Save the model in the specified directory (e.g., Google Drive)

---

### **3. Run Stage 2 — Aggregate Daily Sentiment**

After Stage 1 finishes, update the model path inside `daily_sentiment_aggregation.py`, then run:

```
python daily_sentiment_aggregation.py
```

This outputs sentiment scores in CSV format.

# ## 🔧 Dependencies

* Python 3.9+
* PyTorch
* transformers
* datasets
* sklearn
* pandas
* kagglehub

---

# ## 📚 Citation

Datasets used:

1. **Sentiment Analysis for Financial News**
   Kaggle: [https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

2. **Combined News DJIA Dataset**
   Kaggle: [https://www.kaggle.com/datasets/aaron7sun/stocknews](https://www.kaggle.com/datasets/aaron7sun/stocknews)

Base model:

* **FinBERT** (yiyanghkust/finbert-tone)


