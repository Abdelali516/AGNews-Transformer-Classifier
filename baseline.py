import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset import load_ag_news

TRAIN_CSV = "/home/abdelali/news-transformer-project/data/train.csv"


def main():
    # 1. Load and split data -- SAME split as train.py (same random_state)
    #    so the comparison between baseline and Transformer is fair
    texts, labels = load_ag_news(TRAIN_CSV)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # 2. TF-IDF: turn each text into a vector of word-importance scores
    #    (fit ONLY on training data, same rule as build_vocab)
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    print(f"TF-IDF feature matrix shape (train): {X_train.shape}")

    # 3. Train a simple Logistic Regression classifier
    start_time = time.time()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    elapsed = time.time() - start_time
    print(f"Baseline trained in {elapsed:.1f}s")

    # 4. Evaluate
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)

    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    print(f"\nBaseline train accuracy: {train_acc:.4f}")
    print(f"Baseline val accuracy:   {val_acc:.4f}")


if __name__ == "__main__":
    main()
