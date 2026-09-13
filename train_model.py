"""
Trains a classifier on labeled_dataset.csv (produced by phishing_detector.py)
and compares it against the hand-built rule_based_score() baseline.

Run phishing_detector.py first to generate the dataset, then run this.
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
# domain_age_days is left out on purpose — it's often missing for phishing
# URLs (free hosting, WHOIS privacy), and scikit-learn can't handle NaN
# directly. Fill it with a placeholder like -1 first if you want to try it.


def load_dataset(filename='labeled_dataset.csv'):
    df = pd.read_csv(filename)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def train_and_evaluate(df, test_size=0.3, random_state=42):
    X = df[FEATURE_COLUMNS]
    y = df['label']

    print(f"Dataset size: {len(df)} rows ({y.sum()} phishing, {len(y) - y.sum()} legit)")

    if len(df) < 20:
        print(
            "\nHeads up — this dataset is pretty small for ML. Treat these "
            "numbers as a first look, not a reliable result. Bump up the "
            "OpenPhish limit in phishing_detector.py and re-run to get more data."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- How the model did on the test set ---")
    print(classification_report(y_test, y_pred, target_names=['legit', 'phishing']))

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=['actual: legit', 'actual: phishing'],
        columns=['pred: legit', 'pred: phishing']
    ))

    print("\n--- Which features the model leaned on most ---")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3))

    print("\n--- Any URLs the model got wrong ---")
    mismatched = y_test.index[y_test.values != y_pred]
    if len(mismatched) == 0:
        print("None — perfect score on the test set.")
    else:
        for idx in mismatched:
            actual_label = "phishing" if y_test.loc[idx] == 1 else "legit"
            predicted_label = "phishing" if y_pred[list(y_test.index).index(idx)] == 1 else "legit"
            print(f"  {df.loc[idx, 'url']}  (actual: {actual_label}, predicted: {predicted_label})")

    return model, X_test, y_test


def compare_to_rule_based_baseline(df, threshold=5):
    """How well would the original point-based scorer alone have done?"""
    predicted = (df['risk_score'] >= threshold).astype(int)
    actual = df['label']

    print(f"\n--- Rule-based baseline (risk_score >= {threshold} = phishing) ---")
    print(classification_report(actual, predicted, target_names=['legit', 'phishing']))


if __name__ == '__main__':
    df = load_dataset('labeled_dataset.csv')

    print("=" * 70)
    print("RULE-BASED BASELINE")
    print("=" * 70)
    compare_to_rule_based_baseline(df)

    print("\n" + "=" * 70)
    print("MACHINE LEARNING MODEL")
    print("=" * 70)
    model, X_test, y_test = train_and_evaluate(df)

    joblib.dump(model, 'phishing_model.joblib')
    print("\nModel saved to phishing_model.joblib")
    print("Load it later with: model = joblib.load('phishing_model.joblib')")