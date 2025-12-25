import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_and_merge_data():
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)
    return df


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(df):
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    return df


def split_data(df):
    X = df["content"]
    y = df["label"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer


def train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test):
    model = LinearSVC(random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))


def main():
    df = load_and_merge_data()

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    X_train_tfidf, X_test_tfidf, _ = vectorize_text(X_train, X_test)

    print("\nTF-IDF Train Shape:", X_train_tfidf.shape)
    print("TF-IDF Test Shape:", X_test_tfidf.shape)

    train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test)

main()