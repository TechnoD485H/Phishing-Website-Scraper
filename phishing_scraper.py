import requests
import csv
import math
import ipaddress
from collections import Counter
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
    import pandas as pd  # pip install pandas openpyxl
except ImportError:
    pd = None
    print("Note: pandas not installed. Run 'pip install pandas openpyxl' to enable Excel export.")


# ===========================================================
# TEST DATA SOURCES
# ===========================================================

def load_openphish_urls(limit=10, timeout=10):
    """Downloads the free OpenPhish community feed (no registration needed)."""
    try:
        response = requests.get("https://openphish.com/feed.txt", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch OpenPhish feed: {e}")
        return []

    urls = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
    return urls[:limit]


LEGIT_TEST_URLS = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.python.org",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.paypal.com",
    "https://www.netflix.com",
    "https://www.linkedin.com",
    "https://www.reddit.com",
    "https://www.spotify.com",
    "https://www.dropbox.com",
    "https://www.adobe.com",
    "https://www.stackoverflow.com",
    "https://www.nytimes.com",
    "https://www.bbc.com",
    "https://www.cloudflare.com",
    "https://www.mozilla.org",
    "https://www.ibm.com",
    "https://www.oracle.com",
    "https://www.salesforce.com",
    "https://www.zoom.us",
    "https://www.slack.com",
    "https://www.shopify.com",
    "https://www.ebay.com",
    "https://www.twitch.tv",
    "https://www.airbnb.com",
    "https://www.uber.com",
    "https://www.chase.com",
]


# ===========================================================
# SAFETY: BLOCK PRIVATE / INTERNAL IP TARGETS
# ===========================================================
# Feeds of "wild" URLs (like OpenPhish) occasionally include entries that
# point at private/internal IP ranges (e.g. 192.168.x.x, 10.x.x.x,
# 127.0.0.1). Fetching those blindly means your scraper could end up
# probing YOUR OWN network (router admin pages, local services, etc.)
# instead of the intended external target. We check and refuse before
# ever making the request.

def is_private_or_local(url):
    """Returns True if the URL's hostname is a private, loopback, or link-local IP address."""
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False  # it's a domain name, not a raw IP — fine to proceed


# ===========================================================
# SAFETY: NEUTRALIZE SPREADSHEET FORMULA INJECTION
# ===========================================================
# If a scraped string starts with =, +, -, or @, Excel/Numbers/Sheets can
# interpret it as a formula when the file is later opened — a known
# technique for smuggling a payload into "data". Since our feature values
# are sourced from untrusted, attacker-controlled pages, every string
# value gets sanitized before it's written to CSV or Excel.

def sanitize_for_spreadsheet(value):
    """Prefixes a leading =, +, -, or @ with a single quote so spreadsheet apps treat it as plain text."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def sanitize_row(row):
    """Applies sanitize_for_spreadsheet() to every value in a feature dict."""
    return {key: sanitize_for_spreadsheet(value) for key, value in row.items()}


# ===========================================================
# FREE-HOSTING / PaaS DETECTION
# ===========================================================
# Phishing kits frequently get hosted on free subdomains of legitimate
# platforms (Weebly, Cloudflare Pages, GitBook, Azure Front Door, etc.).
# Being hosted on one of these — especially with a random-looking
# subdomain — is itself a strong red flag, independent of domain age,
# since the PLATFORM's domain age tells you nothing about the attacker's
# specific page.
FREE_HOSTING_DOMAINS = [
    'weebly.com', 'pages.dev', 'gitbook.io', 'azurefd.net',
    'cloudclusters.net', 'amplifyapp.com', 'godaddysites.com',
    'netlify.app', 'vercel.app', 'herokuapp.com', '000webhostapp.com',
    'github.io', 'firebaseapp.com', 'web.app', 'wixsite.com',
    'blogspot.com', 'sites.google.com', 'repl.co', 'glitch.me',
]


def is_on_free_hosting(url):
    """Returns True if the URL's hostname is a subdomain of a known free-hosting platform."""
    hostname = urlparse(url).hostname or ''
    return any(hostname == d or hostname.endswith('.' + d) for d in FREE_HOSTING_DOMAINS)


# ===========================================================
# SUBDOMAIN ENTROPY
# ===========================================================
# Automated phishing kits often generate random-looking subdomains
# (e.g. "3ib64a1sok-k9t7f7se-evaygbgad7dueuhq") rather than human-chosen
# names. Shannon entropy measures how "random" a string looks — higher
# entropy = less like a real word, more like an autogenerated string.

def subdomain_entropy(url):
    """
    Calculates the Shannon entropy of the attacker-controlled part of a URL's hostname.

    On free-hosting platforms, the interesting part is everything BEFORE
    the platform's own domain — so we strip the known suffix first.
    Without this, a naive "first label" approach would measure "www"
    (near-zero entropy, useless) instead of the actual attacker-chosen
    string.

    On regular domains, we fall back to just the first label (typical
    subdomain position), since we don't have a public-suffix list here
    to reliably find the registrable domain boundary.
    """
    hostname = urlparse(url).hostname or ''

    matched_base = next(
        (base for base in FREE_HOSTING_DOMAINS
         if hostname == base or hostname.endswith('.' + base)),
        None
    )

    if matched_base:
        target = hostname[: -(len(matched_base) + 1)] if hostname != matched_base else ''
        if target.startswith('www.'):
            target = target[4:]
    else:
        target = hostname.split('.')[0]

    if not target:
        return 0.0

    counts = Counter(target)
    length = len(target)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


# ===========================================================
# PHISHING FEATURE EXTRACTION
# ===========================================================

def fetch_page(session, url, timeout=5):
    """Fetches and parses a single URL. Returns a BeautifulSoup object, or None on failure/unsafe target."""
    if is_private_or_local(url):
        print(f"Skipping {url}: points to a private/internal address")
        return None

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
    features['on_free_hosting'] = is_on_free_hosting(url)
    features['subdomain_entropy'] = round(subdomain_entropy(url), 2)

    # --- Page content signals (only meaningful if the page actually loaded) ---
    if soup:
        forms = soup.find_all('form')
        features['num_forms'] = len(forms)
        features['has_password_field'] = bool(soup.find('input', {'type': 'password'}))

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
    if is_private_or_local(url):
        return None
    try:
        domain = urlparse(url).netloc
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            # Some WHOIS servers return a timezone-aware datetime, others return
            # a naive one. datetime.now() is always naive, so mixing the two
            # raises "can't subtract offset-naive and offset-aware datetimes".
            # Stripping tzinfo here normalizes both cases before subtracting.
            if creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=None)
            return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"WHOIS lookup failed for {url}: {e}")
    return None


def rule_based_score(features, domain_age_days):
    """
    Simple point-based phishing likelihood score. Higher = more suspicious.

    Domain age is only trusted as a "this looks safe" signal when the site
    is NOT on a known free-hosting platform. On free hosting, the
    platform's own age is meaningless — what matters is that free hosting
    + a suspicious-looking subdomain is being used to impersonate
    something at all.
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
    if features['on_free_hosting']:
        score += 3
    if features['subdomain_entropy'] >= 3.5:
        score += 3

    if not features['on_free_hosting']:
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
# LABELED DATASET BUILDER (for ML training)
# ===========================================================

def build_labeled_dataset(phishing_urls, legit_urls, max_workers=5):
    """
    Runs the full analysis pipeline across both known-phishing and
    known-legit URLs, and stamps each result with a 'label' column
    (1 = phishing, 0 = legit). This labeled table is what gets used
    to train a classifier later — the ground truth comes from which
    list each URL was sourced from, not from the rule_based_score.
    """
    phishing_set = set(phishing_urls)

    all_urls = list(phishing_urls) + list(legit_urls)
    results = analyze_urls(all_urls, max_workers=max_workers)

    for r in results:
        r['label'] = 1 if r['url'] in phishing_set else 0

    return results


# ===========================================================
# EXPORT
# ===========================================================

def save_to_csv(rows, filename='phishing_features.csv', fieldnames=None):
    """Writes a list of feature dicts to a CSV file as a proper table, sanitized against formula injection."""
    if not rows:
        print("No rows to save.")
        return
    sanitized_rows = [sanitize_row(row) for row in rows]
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(sanitized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sanitized_rows)
    print(f"Saved {len(rows)} rows to {filename}")


def save_to_excel(rows, filename='phishing_features.xlsx'):
    """Writes a list of feature dicts to an Excel file as a proper table, sanitized against formula injection."""
    if not rows:
        print("No rows to save.")
        return
    if pd is None:
        print("pandas not installed — skipping Excel export. Run 'pip install pandas openpyxl'.")
        return
    sanitized_rows = [sanitize_row(row) for row in rows]
    df = pd.DataFrame(sanitized_rows)
    df.to_excel(filename, index=False)
    print(f"Saved {len(rows)} rows to {filename}")


if __name__ == '__main__':
    # Bumped from 10 -> 60: a handful of URLs was enough to sanity-check the
    # pipeline and catch feature bugs (which it did), but is too small a
    # sample to draw real conclusions from or use as ML training data.
    phishing_urls = load_openphish_urls(limit=60)
    print(f"Loaded {len(phishing_urls)} URLs from OpenPhish feed")

    legit_urls = LEGIT_TEST_URLS
    print(f"Using {len(legit_urls)} known-legitimate URLs")

    dataset = build_labeled_dataset(phishing_urls, legit_urls)

    print(f"\n{'LABEL':10} | {'SCORE':5} | URL")
    print("-" * 80)
    for r in sorted(dataset, key=lambda x: x['risk_score'], reverse=True):
        label = "PHISHING" if r['label'] == 1 else "LEGIT"
        print(f"{label:10} | {r['risk_score']:5} | {r['url']}")

    # Quick sanity check: does the rule-based score actually separate
    # the two classes on average? This is the same check we did by eye
    # on the small sample, now automated across the full batch.
    phishing_scores = [r['risk_score'] for r in dataset if r['label'] == 1]
    legit_scores = [r['risk_score'] for r in dataset if r['label'] == 0]
    if phishing_scores and legit_scores:
        avg_phishing = sum(phishing_scores) / len(phishing_scores)
        avg_legit = sum(legit_scores) / len(legit_scores)
        print(f"\nAvg risk_score — phishing: {avg_phishing:.2f} | legit: {avg_legit:.2f}")

    # This labeled file is the one to feed into train_model.py
    save_to_csv(dataset, "labeled_dataset.csv")
    save_to_excel(dataset, "labeled_dataset.xlsx")