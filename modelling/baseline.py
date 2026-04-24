import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


def train_baseline(X_train, y_train, X_test, y_test):
    """Train and evaluate a majority-class baseline."""

    # DummyClassifier — most frequent class
    model = DummyClassifier(strategy="most_frequent") 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Baseline (Most Frequent) ===")
    print(classification_report(y_test, y_pred))
    return model, y_pred

