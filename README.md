# Fake News Detection using NLP and Machine Learning

This project implements an end-to-end **Fake News Detection** system using Natural Language Processing (NLP) and Machine Learning techniques.  
The model classifies news articles as **Real** or **Fake** based on their textual content.

---

## 🚀 Project Overview

- **Problem Type:** Binary Text Classification
- **Domain:** Natural Language Processing (NLP)
- **Approach:** TF-IDF Vectorization + Linear Support Vector Machine (SVM)
- **Goal:** Detect misinformation in news articles with high accuracy

---

## 📊 Dataset Information

The dataset consists of two separate files:

- `Fake.csv` → Fake news articles  
- `True.csv` → Real news articles  

Each file contains the following columns:
- `title`
- `text`
- `subject`
- `date`

The datasets are merged and labeled as:
- **1 → Fake News**
- **0 → Real News**

📎 Dataset Source:  
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

---

## 🧠 Machine Learning Pipeline

1. Load and merge real and fake news datasets  
2. Text preprocessing (cleaning, normalization)  
3. Feature engineering using **TF-IDF** (unigrams + bigrams)  
4. Train-test split with stratification  
5. Model training using **Linear SVM**  
6. Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix  

---

## 📈 Model Performance

- **Accuracy:** ~99.5%
- **Precision & Recall:** Balanced for both classes
- **Confusion Matrix:** Very low misclassification rate

The model demonstrates strong generalization and reliability on unseen data.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Regular Expressions (re)
- TF-IDF Vectorization
- Linear Support Vector Machine (SVM)

---

> Note: Dataset files are not included in the repository. Please download them from Kaggle using the link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets.

---

## 🎯 Key Learnings

- Importance of text preprocessing in NLP

- Feature extraction using TF-IDF

- Effectiveness of Linear SVM for high-dimensional text data

- Proper evaluation of classification models

- Building clean and reproducible ML pipelines