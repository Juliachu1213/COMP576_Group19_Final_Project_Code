# 🌟 **Predicting Stock Market Movements Using Deep Learning-Based Sentiment Analysis of Financial News**

**Chu-Yun Chu · Kelly Hung · Vivian Liu · Thomas Lin**
COMP 576 — Deep Learning — Fall 2025

---

# 🔍 **Project Summary**

This repository implements our full pipeline to answer the research question:

## **Can deep-learning-based sentiment signals predict next-day DJIA movement?**

We build a 3-stage system combining NLP (FinBERT), sentiment aggregation, and sequence models (MLP / LSTM / GRU):

```
Financial News → FinBERT Sentiment → Daily Sentiment Time Series → DJIA Prediction
```

---

# 🚀 **Project Pipeline**

## **Stage 1 — FinBERT Fine-Tuning**

Fine-tune a domain-specific financial sentiment model using ~4,000 labeled headlines.

## **Stage 2 — Daily Sentiment Aggregation**

Apply the classifier to 8 years of historical headlines and compute daily sentiment features aligned with next-day DJIA labels.

## **Stage 3 — Market Movement Prediction (MLP / LSTM / GRU)**

Train time-series models using daily sentiment sequences to predict whether the DJIA goes **Up (1)** or **Down (0)** the next day.

---

# 📁 **Repository Structure**

```
├── finbert_model.py                # Stage 1 — Fine-tuning FinBERT
├── daily_sentiment_aggregation.py  # Stage 2 — Daily sentiment computation
├── MLP_training.ipynb              # Stage 3 — MLP model training
├── LSTM_training.ipynb             # Stage 3 — LSTM model training
├── GRU_training.ipynb              # Stage 3 — GRU model training
├── datasets/                       # (optional) processed CSV files
└── README.md
```

---

# 🧠 **Stage 1 — FinBERT Fine-Tuning (`finbert_model.py`)**

This script:

✔ Downloads Kaggle financial sentiment dataset
✔ Preprocesses text + labels
✔ Fine-tunes **FinBERT (`yiyanghkust/finbert-tone`)**
✔ Evaluates with accuracy, F1-score, and confusion matrix
✔ Saves model + tokenizer

---

# 📊 **Stage 2 — Daily Sentiment Aggregation (`daily_sentiment_aggregation.py`)**

This script:

✔ Loads fine-tuned FinBERT
✔ Downloads **Combined_News_DJIA.csv**
✔ Runs sentiment inference on all 25 daily headlines
✔ Aggregates into a **daily sentiment score**
✔ Aligns with next-day DJIA movement
✔ Saves processed datasets

### **Outputs**

```
daily_sentiment_scores.csv
news_label_daily_sentiment_scores.csv   # used for Stage 3
```

---

# 🤖 **Stage 3 — Market Movement Prediction (MLP / LSTM / GRU)**

Training notebooks:

* **`MLP_training.ipynb`** — baseline fully connected network
* **`LSTM_training.ipynb`** — sequence model capturing temporal patterns
* **`GRU_training.ipynb`** — gated recurrent model with efficient memory

Each notebook:

✔ Loads sentiment time-series from Stage 2
✔ Constructs N-day sliding windows
✔ Trains classification model
✔ Plots training/validation curves
✔ Reports accuracy & confusion matrix

---

# ▶️ **How to Run the Pipeline**

## **1️⃣ Install dependencies**

```
pip install transformers datasets accelerate kagglehub pandas scikit-learn torch
```

---

## **2️⃣ Run Stage 1 — Fine-tune FinBERT**

```
python finbert_model.py
```

---

## **3️⃣ Run Stage 2 — Aggregate Sentiment**

Make sure `model_path` points to the fine-tuned FinBERT directory:

```
python daily_sentiment_aggregation.py
```

---

## **4️⃣ Run Stage 3 — Train Prediction Models**

Open any notebook:

```
MLP_training.ipynb
LSTM_training.ipynb
GRU_training.ipynb
```

Run all cells to reproduce model performance.

---

# 📦 **Dependencies**

* Python 3.9+
* PyTorch
* HuggingFace Transformers
* datasets
* accelerate
* scikit-learn
* pandas
* kagglehub

---

# 📚 **Datasets**

### **Financial Sentiment Dataset**

Kaggle — Sentiment Analysis for Financial News
[https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

### **Daily News + DJIA Dataset**

Kaggle — Stock News and DJIA Movement
[https://www.kaggle.com/datasets/aaron7sun/stocknews](https://www.kaggle.com/datasets/aaron7sun/stocknews)

### **Base Pretrained Model**

FinBERT (yiyanghkust/finbert-tone)

---

# 🙌 **Contributors**

Chu-Yun Chu
Kelly Hung
Vivian Liu
Thomas Lin


