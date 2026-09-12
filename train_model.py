"""
train_model.py

Trains a machine-learning classifier on the labeled dataset produced by
phishing_detector.py (labeled_dataset.csv), and compares its performance
against the hand-built rule_based_score() baseline.

Run phishing_detector.py FIRST to generate labeled_dataset.csv, then run
this script.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


FEATURE_COLUMNS = [
    'url_length',
    'num_dots',
    'num_hyphens',
    'has_at_symbol',
    'uses_https',
    'has_ip_as_domain',
    'on_free_hosting',
    'subdomain_entropy',
    'num_forms',
    'has_password_field',
    'form_posts_externally',
]
# Note: domain_age_days is deliberately excluded by default — it's very
# often missing (None) for phishing URLs on free hosting or with WHOIS
# privacy, and scikit-learn's RandomForestClassifier can't handle NaN
# directly. If you want to include it, fill missing values first
# (e.g. df['domain_age_days'].fillna(-1)) before adding it to this list.


def load_dataset(filename='labeled_dataset.csv'):
    """Loads the labeled dataset produced by phishing_detector.py."""
    df = pd.read_csv(filename)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def train_and_evaluate(df, test_size=0.3, random_state=42):
    """
    Splits the data, trains a RandomForestClassifier, and prints an
    evaluation report. Returns the trained model.
    """
    X = df[FEATURE_COLUMNS]
    y = df['label']

    print(f"Dataset size: {len(df)} rows ({y.sum()} phishing, {len(y) - y.sum()} legit)")

    if len(df) < 20:
        print(
            "\nWarning: this dataset is quite small for ML training. "
            "Results below are illustrative, not statistically reliable — "
            "increase the OpenPhish limit in phishing_detector.py and re-run "
            "to collect more data before trusting these numbers."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Classifier evaluation (on held-out test set) ---")
    print(classification_report(y_test, y_pred, target_names=['legit', 'phishing']))

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=['actual: legit', 'actual: phishing'],
        columns=['pred: legit', 'pred: phishing']
    ))

    print("\n--- Feature importance (which signals the model relied on most) ---")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3))

    return model, X_test, y_test


def compare_to_rule_based_baseline(df, threshold=5):
    """
    Evaluates how well the existing rule_based_score() alone would have
    classified the same data, using a simple threshold. This is the
    baseline the ML model should ideally beat.
    """
    predicted = (df['risk_score'] >= threshold).astype(int)
    actual = df['label']

    print(f"\n--- Rule-based baseline (risk_score >= {threshold} => phishing) ---")
    print(classification_report(actual, predicted, target_names=['legit', 'phishing']))


if __name__ == '__main__':
    df = load_dataset('labeled_dataset.csv')

    print("=" * 70)
    print("RULE-BASED BASELINE")
    print("=" * 70)
    compare_to_rule_based_baseline(df)

    print("\n" + "=" * 70)
    print("MACHINE LEARNING MODEL (RandomForestClassifier)")
    print("=" * 70)
    model, X_test, y_test = train_and_evaluate(df)

    joblib.dump(model, 'phishing_model.joblib')
    print("\nSaved trained model to phishing_model.joblib")
    print("Load it later with: model = joblib.load('phishing_model.joblib')")