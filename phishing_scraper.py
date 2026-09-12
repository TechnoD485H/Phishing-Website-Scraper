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
    print("python-whois not installed — domain-age checks will be skipped. (pip install python-whois)")

try:
    import pandas as pd  # pip install pandas openpyxl
except ImportError:
    pd = None
    print("pandas not installed — Excel export will be skipped. (pip install pandas openpyxl)")


# ---------------------------------------------------------
# Where our test data comes from
# ---------------------------------------------------------

def load_openphish_urls(limit=10, timeout=10):
    """Grabs a batch of currently-active phishing URLs from OpenPhish's free feed."""
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


# ---------------------------------------------------------
# Safety: don't let the scraper touch private/internal IPs
# ---------------------------------------------------------

def is_private_or_local(url):
    """True if a URL points at a private/loopback/internal IP (e.g. your own router)."""
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False  # it's a normal domain name, not a raw IP


# ---------------------------------------------------------
# Safety: stop scraped text from becoming an Excel formula
# ---------------------------------------------------------

def sanitize_for_spreadsheet(value):
    """Neutralizes strings that start with =, +, -, or @ so they can't run as formulas."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def sanitize_row(row):
    return {key: sanitize_for_spreadsheet(value) for key, value in row.items()}


# ---------------------------------------------------------
# Free-hosting detection
# ---------------------------------------------------------
# A lot of phishing pages live on free subdomains of real platforms
# (Weebly, Azure, GitBook, etc.) — being hosted there is a red flag
# on its own, since the platform's age doesn't tell you anything
# about how old the attacker's specific page actually is.

FREE_HOSTING_DOMAINS = [
    'weebly.com', 'pages.dev', 'gitbook.io', 'azurefd.net',
    'cloudclusters.net', 'amplifyapp.com', 'godaddysites.com',
    'netlify.app', 'vercel.app', 'herokuapp.com', '000webhostapp.com',
    'github.io', 'firebaseapp.com', 'web.app', 'wixsite.com',
    'blogspot.com', 'sites.google.com', 'repl.co', 'glitch.me',
]


def is_on_free_hosting(url):
    hostname = urlparse(url).hostname or ''
    return any(hostname == d or hostname.endswith('.' + d) for d in FREE_HOSTING_DOMAINS)


# ---------------------------------------------------------
# Subdomain entropy — how "random" does the subdomain look?
# ---------------------------------------------------------
# Phishing kits often auto-generate subdomains like
# "3ib64a1sok-k9t7f7se-evaygbgad7dueuhq". Shannon entropy gives us
# a number for how random a string looks — real words score low,
# generated gibberish scores high.

def subdomain_entropy(url):
    hostname = urlparse(url).hostname or ''

    # on free hosting, strip the platform's own domain first so we're
    # measuring the attacker's chosen name, not "www"
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


# ---------------------------------------------------------
# Core pipeline: fetch -> extract features -> score
# ---------------------------------------------------------

def fetch_page(session, url, timeout=5):
    """Fetches and parses a page. Returns None if it's unreachable, or points somewhere unsafe."""
    if is_private_or_local(url):
        print(f"Skipping {url} — points to a private/internal address")
        return None

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None
    return BeautifulSoup(response.text, 'html.parser')


def extract_features(url, soup):
    """Pulls out the signals we use to judge whether a page looks like phishing."""
    parsed = urlparse(url)
    features = {"url": url}

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

    if soup:
        forms = soup.find_all('form')
        features['num_forms'] = len(forms)
        features['has_password_field'] = bool(soup.find('input', {'type': 'password'}))

        # does any form send its data to a different domain than the page itself?
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
    """How old is this domain, according to WHOIS? Returns None if we can't tell."""
    if whois is None or is_private_or_local(url):
        return None
    try:
        domain = urlparse(url).netloc
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            # WHOIS servers are inconsistent about timezones — normalize before subtracting
            if creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=None)
            return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"WHOIS lookup failed for {url}: {e}")
    return None


def rule_based_score(features, domain_age_days):
    """Simple point system — higher means more suspicious. Our v1 baseline before ML."""
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

    # only trust "old domain = safe" off free hosting — on free hosting,
    # the platform's age isn't the attacker's age
    if not features['on_free_hosting']:
        if domain_age_days is not None and domain_age_days < 30:
            score += 3

    return score


def analyze_url(session, url):
    soup = fetch_page(session, url)
    features = extract_features(url, soup)
    domain_age_days = get_domain_age_days(url)

    features['domain_age_days'] = domain_age_days
    features['risk_score'] = rule_based_score(features, domain_age_days)

    return features


def analyze_urls(urls, max_workers=5):
    """Runs analyze_url() across a list of URLs at once."""
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


# ---------------------------------------------------------
# Building a labeled dataset (for training the ML model later)
# ---------------------------------------------------------

def build_labeled_dataset(phishing_urls, legit_urls, max_workers=5):
    """Runs everything and tags each row 1 (phishing) or 0 (legit) based on its source list."""
    phishing_set = set(phishing_urls)
    all_urls = list(phishing_urls) + list(legit_urls)
    results = analyze_urls(all_urls, max_workers=max_workers)

    for r in results:
        r['label'] = 1 if r['url'] in phishing_set else 0

    return results


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

def save_to_csv(rows, filename='phishing_features.csv', fieldnames=None):
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
    if not rows:
        print("No rows to save.")
        return
    if pd is None:
        print("pandas isn't installed, so I can't write Excel files. (pip install pandas openpyxl)")
        return
    sanitized_rows = [sanitize_row(row) for row in rows]
    df = pd.DataFrame(sanitized_rows)
    df.to_excel(filename, index=False)
    print(f"Saved {len(rows)} rows to {filename}")


if __name__ == '__main__':
    phishing_urls = load_openphish_urls(limit=60)
    print(f"Loaded {len(phishing_urls)} URLs from OpenPhish")

    legit_urls = LEGIT_TEST_URLS
    print(f"Using {len(legit_urls)} known-legit URLs")

    dataset = build_labeled_dataset(phishing_urls, legit_urls)

    print(f"\n{'LABEL':10} | {'SCORE':5} | URL")
    print("-" * 80)
    for r in sorted(dataset, key=lambda x: x['risk_score'], reverse=True):
        label = "PHISHING" if r['label'] == 1 else "LEGIT"
        print(f"{label:10} | {r['risk_score']:5} | {r['url']}")

    phishing_scores = [r['risk_score'] for r in dataset if r['label'] == 1]
    legit_scores = [r['risk_score'] for r in dataset if r['label'] == 0]
    if phishing_scores and legit_scores:
        avg_phishing = sum(phishing_scores) / len(phishing_scores)
        avg_legit = sum(legit_scores) / len(legit_scores)
        print(f"\nAverage risk score — phishing: {avg_phishing:.2f} | legit: {avg_legit:.2f}")

    save_to_csv(dataset, "labeled_dataset.csv")
    save_to_excel(dataset, "labeled_dataset.xlsx")
    print("test")