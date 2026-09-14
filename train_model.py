"""
Trains a classifier on labeled_dataset.csv (produced by phishing_detector.py)
and compares it against the hand-built rule_based_score() baseline.
Run phishing_detector.py first, then this.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


FEATURE_COLUMNS = [
    'fetch_succeeded',
    'url_length',
    'num_dots',
    'num_hyphens',
    'has_at_symbol',
    'uses_https',
    'has_ip_as_domain',
    'on_free_hosting',
    'has_suspicious_keyword',
    'subdomain_length',
    'subdomain_entropy',
    'num_forms',
    'has_password_field',
    'form_posts_externally',
    'domain_age_days',    # NaN-filled with -1 during load
    'domain_age_known',   # 1 if we got a real registration date, 0 if not
]


def load_dataset(filename='labeled_dataset.csv'):
    df = pd.read_csv(filename)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # sklearn can't handle NaN. We fill missing ages with -1 AND keep a
    # separate flag, so the model can tell "brand new domain" apart from
    # "no record found at all" — those are very different things.
    df['domain_age_days'] = df['domain_age_days'].fillna(-1)

    # older CSVs won't have these — backfill sensible defaults
    if 'fetch_succeeded' not in df.columns:
        df['fetch_succeeded'] = 1
    if 'domain_age_known' not in df.columns:
        df['domain_age_known'] = (df['domain_age_days'] != -1).astype(int)
    if 'has_suspicious_keyword' not in df.columns:
        df['has_suspicious_keyword'] = 0
    if 'subdomain_length' not in df.columns:
        df['subdomain_length'] = df['url'].map(
            lambda u: len((u.split('//')[-1].split('/')[0].split('.') or [''])[0])
        )

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

    if y.min() == y.max():
        raise ValueError(
            "Dataset only contains one class — can't train a classifier. "
            "Check that labeled_dataset.csv has both phishing and legit rows."
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
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    mismatched = y_test[y_test != y_pred_series]
    if len(mismatched) == 0:
        print("None — perfect score on the test set.")
    else:
        for idx in mismatched.index:
            actual_label = "phishing" if y_test.loc[idx] == 1 else "legit"
            predicted_label = "phishing" if y_pred_series.loc[idx] == 1 else "legit"
            reasons = df.loc[idx, 'risk_reasons'] if 'risk_reasons' in df.columns else 'n/a'
            print(f"  {df.loc[idx, 'url']}  (actual: {actual_label}, predicted: {predicted_label})")
            print(f"      rule-based reasons: {reasons}")

    return model, X_test, y_test, y_pred


def compare_to_rule_based_baseline(df_test, threshold=4):
    """Scores the old point-based system on the SAME test rows the model saw,
    so the comparison is fair."""
    predicted = (df_test['risk_score'] >= threshold).astype(int)
    actual = df_test['label']

    print(f"\n--- Rule-based baseline (risk_score >= {threshold} = phishing), on the test set ---")
    print(classification_report(actual, predicted, target_names=['legit', 'phishing']))


if __name__ == '__main__':
    df = load_dataset('labeled_dataset.csv')

    model, X_test, y_test, y_pred = train_and_evaluate(df)

    df_test = df.loc[X_test.index]

    print("\n" + "=" * 70)
    print("RULE-BASED BASELINE (same test set)")
    print("=" * 70)
    compare_to_rule_based_baseline(df_test)

    joblib.dump(model, 'phishing_model.joblib')
    print("\nModel saved to phishing_model.joblib")
    print("Load it later with: model = joblib.load('phishing_model.joblib')")