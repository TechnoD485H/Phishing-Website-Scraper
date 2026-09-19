"""
app.py — Flask web server for the PhishScan UI.

Loads the trained phishing_model.joblib and phishing_scraper.py, then
exposes two routes:
  GET  /          — serves the UI (templates/index.html)
  POST /analyze   — accepts a URL, runs full feature extraction +
                    ML prediction, returns JSON

Run:
    python3 app.py

Then open http://localhost:5000 in your browser.

Requirements:
    pip install flask scikit-learn joblib pandas

Make sure phishing_scraper.py and phishing_model.joblib are in the same
directory as this file before starting the server.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Import the feature extraction pipeline from the existing scraper.
# This means the model always sees exactly the same features it was
# trained on — no risk of a subtle mismatch between training and serving.
try:
    from phishing_scraper import analyze_url
except ImportError:
    raise SystemExit(
        "ERROR: phishing_scraper.py not found.\n"
        "Make sure app.py is in the same directory as phishing_scraper.py."
    )

# ── Feature columns — must match train_model.py exactly ──────────────
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
    'brand_edit_distance',
    'impersonates_brand',
]

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'phishing_model.joblib')

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise SystemExit(
        "ERROR: phishing_model.joblib not found.\n"
        "Run train_model.py first to generate it, then start the server."
    )


def features_to_df(features: dict) -> pd.DataFrame:
    """
    Converts the feature dict from analyze_url() into the DataFrame
    shape the model expects — same columns, same order as training.
    """
    row = {}
    for col in FEATURE_COLUMNS:
        val = features.get(col)
        # brand_edit_distance is -1 when there's no close match;
        # boolean columns need to be kept as bool/int for the model
        row[col] = val if val is not None else 0
    return pd.DataFrame([row])


# ── Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # add scheme if missing so analyze_url() and the browser both agree
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        # Step 1: full feature extraction (fetch page, WHOIS/RDAP, etc.)
        features = analyze_url(url)

        # Step 2: ML prediction
        X = features_to_df(features)
        prediction = int(model.predict(X)[0])             # 1 = phishing, 0 = legit
        probability = model.predict_proba(X)[0].tolist()  # [P(legit), P(phishing)]

        phishing_prob = round(probability[1] * 100, 1)
        legit_prob    = round(probability[0] * 100, 1)

        return jsonify({
            'url':            features['url'],
            'prediction':     prediction,           # 1 or 0
            'phishing_prob':  phishing_prob,        # 0-100%
            'legit_prob':     legit_prob,            # 0-100%
            'risk_score':     features['risk_score'],
            'risk_reasons':   features['risk_reasons'],
            'matched_brand':  features.get('matched_brand', ''),
            'on_free_hosting': features.get('on_free_hosting', False),
            'uses_https':     features.get('uses_https', True),
            'domain_age_days': features.get('domain_age_days'),
            'subdomain_entropy': features.get('subdomain_entropy', 0),
            'impersonates_brand': features.get('impersonates_brand', False),
            'fetch_succeeded': features.get('fetch_succeeded', False),
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    # debug=False in production; True only while developing locally
    app.run(host='0.0.0.0', port=5500, debug=True)