import requests
import csv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from datetime import datetime

try:
    import whois  # pip install python-whois
except ImportError:
    whois = None
    print("Note: python-whois not installed. Run 'pip install python-whois' to enable domain-age checks.")

try:
    import pandas as pd  
except ImportError:
    pd = None
    print("Note: pandas not installed. Run 'pip install pandas openpyxl' to enable Excel export.")


# ===========================================================
# PHISHING FEATURE EXTRACTION
# ===========================================================

def fetch_page(session, url, timeout=5):
    """Fetches and parses a single URL. Returns a BeautifulSoup object, or None on failure."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None
    return BeautifulSoup(response.text, 'html.parser')


def extract_features(url, soup):
    """
    Pulls measurable red-flag signals out of a URL + its parsed page.
    Returns a dict of features suitable for scoring or ML training.
    """
    parsed = urlparse(url)
    features = {"url": url}

    # --- URL-based signals ---
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_at_symbol'] = '@' in url
    features['uses_https'] = parsed.scheme == 'https'
    features['has_ip_as_domain'] = bool(
        parsed.hostname and parsed.hostname.replace('.', '').isdigit()
    )

    # --- Page content signals (only meaningful if the page actually loaded) ---
    if soup:
        forms = soup.find_all('form')
        features['num_forms'] = len(forms)
        features['has_password_field'] = bool(soup.find('input', {'type': 'password'}))

        # does any form submit to a DIFFERENT domain than the page itself?
        external_form = False
        for form in forms:
            action = form.get('action', '')
            if action.startswith('http') and parsed.netloc not in action:
                external_form = True
        features['form_posts_externally'] = external_form
    else:
        features['num_forms'] = 0
        features['has_password_field'] = False
        features['form_posts_externally'] = False

    return features


def get_domain_age_days(url):
    """Looks up how old a domain is via WHOIS. Returns None if lookup fails or whois isn't installed."""
    if whois is None:
        return None
    try:
        domain = urlparse(url).netloc
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"WHOIS lookup failed for {url}: {e}")
    return None


def rule_based_score(features, domain_age_days):
    """
    Simple point-based phishing likelihood score. Higher = more suspicious.
    This is a v1 baseline to compare a future ML model against.
    """
    score = 0
    if not features['uses_https']:
        score += 1
    if features['has_ip_as_domain']:
        score += 2
    if features['has_at_symbol']:
        score += 2
    if features['form_posts_externally']:
        score += 3
    if features['has_password_field'] and features['form_posts_externally']:
        score += 2
    if domain_age_days is not None and domain_age_days < 30:
        score += 3
    return score


def analyze_url(session, url):
    """Full pipeline for one URL: fetch -> extract features -> domain age -> score."""
    soup = fetch_page(session, url)
    features = extract_features(url, soup)
    domain_age_days = get_domain_age_days(url)

    features['domain_age_days'] = domain_age_days
    features['risk_score'] = rule_based_score(features, domain_age_days)

    return features


def analyze_urls(urls, max_workers=5):
    """Runs analyze_url() across a list of URLs concurrently, returns a list of feature dicts."""
    results = []
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"
        })
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(analyze_url, session, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Failed to analyze {url}: {e}")
    return results


# ===========================================================
# EXPORT
# ===========================================================

def save_to_csv(rows, filename='phishing_features.csv', fieldnames=None):
    """Writes a list of feature dicts to a CSV file as a proper table."""
    if not rows:
        print("No rows to save.")
        return
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {filename}")


def save_to_excel(rows, filename='phishing_features.xlsx'):
    """Writes a list of feature dicts to an Excel file as a proper table."""
    if not rows:
        print("No rows to save.")
        return
    if pd is None:
        print("pandas not installed — skipping Excel export. Run 'pip install pandas openpyxl'.")
        return
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)  # index=False avoids an extra row-number column
    print(f"Saved {len(rows)} rows to {filename}")


if __name__ == '__main__':
    # Replace these with real examples later (e.g. from PhishTank / UCI dataset)
    test_urls = [
        "https://www.google.com",
        "https://www.python.org",
        "http://192.168.1.1/login"
    ]

    results = analyze_urls(test_urls)
    for r in results:
        print(r)

    save_to_csv(results, "phishing_features.csv")
    save_to_excel(results, "phishing_features.xlsx")