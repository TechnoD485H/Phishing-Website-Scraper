"""
Trains a classifier on labeled_dataset.csv (produced by phishing_scraper.py)
and compares it against the hand-built rule_based_score() baseline.

Run phishing_scraper.py first to generate the dataset, then run this.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
    'has_suspicious_keyword',
    'subdomain_length',
    'subdomain_entropy',
    'num_forms',
    'has_password_field',
    'form_posts_externally',
    'fetch_succeeded',
    'domain_age_known',
]
# domain_age_days itself is left out — it's often missing (None) for real
# phishing URLs, and scikit-learn can't handle NaN directly. domain_age_known
# (whether we got an answer at all) is included instead as a cheap stand-in.


def load_dataset(filename='labeled_dataset.csv'):
    df = pd.read_csv(filename)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            f"Did you run the latest phishing_scraper.py before this?"
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
            "OpenPhish limit in phishing_scraper.py and re-run to get more data."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # A single train/test split can look perfect by luck, especially on a
    # dataset this size. 5-fold cross-validation trains/tests on 5 different
    # splits and reports the spread — a much more trustworthy signal than
    # one number. If this varies a lot, or sits meaningfully below the
    # single-split score above, that's overfitting showing itself.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=random_state),
        X, y, cv=cv, scoring='f1'
    )
    print(f"\n--- 5-fold cross-validation (F1 score per fold) ---")
    print(f"Folds: {[round(s, 3) for s in cv_scores]}")
    print(f"Mean: {cv_scores.mean():.3f}  |  Std dev: {cv_scores.std():.3f}")
    if cv_scores.std() > 0.05:
        print("Noticeable spread across folds — treat the single-split result with caution.")
    elif cv_scores.mean() < 0.95:
        print("Consistently below-perfect across folds — a more honest generalization estimate than one split.")

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
            reasons = df.loc[idx, 'risk_reasons'] if 'risk_reasons' in df.columns else ""
            print(f"  {df.loc[idx, 'url']}  (actual: {actual_label}, predicted: {predicted_label})")
            if reasons:
                print(f"    rule-based reasons: {reasons}")

    return model, X_test, y_test


def compare_to_rule_based_baseline(df, threshold=2):
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