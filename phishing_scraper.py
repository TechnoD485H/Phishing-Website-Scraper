"""
Fetches URLs, extracts phishing signals, and builds a labeled dataset.
Run this first, then train_model.py.
"""

import requests
import csv
import math
import ipaddress
import threading
from collections import Counter
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from datetime import datetime

try:
    import whois  # pip install python-whois
except ImportError:
    whois = None
    print("python-whois not installed — skipping domain-age checks. (pip install python-whois)")

try:
    import tldextract  # pip install tldextract
    # offline list only, so it never needs network access
    _tld_extractor = tldextract.TLDExtract(suffix_list_urls=())
except ImportError:
    tldextract = None
    _tld_extractor = None
    print("tldextract not installed — WHOIS lookups will be less reliable on multi-part domains. (pip install tldextract)")

try:
    import pandas as pd  # pip install pandas openpyxl
except ImportError:
    pd = None
    print("pandas not installed — skipping Excel export. (pip install pandas openpyxl)")


# Each thread needs its own session — requests.Session isn't thread-safe,
# and sharing one across workers causes random connection errors.
_thread_local = threading.local()

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")


def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = s
    return _thread_local.session


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
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.bankofamerica.com",
    "https://www.wellsfargo.com",
    "https://www.instagram.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.pinterest.com",
    "https://www.wordpress.com",
    "https://www.yahoo.com",
    "https://www.bing.com",
    "https://www.aws.amazon.com",
    "https://www.digitalocean.com",
    "https://www.atlassian.com",
    "https://www.notion.so",
    "https://www.figma.com",
    "https://www.canva.com",
    "https://www.trello.com",
    "https://www.asana.com",
]


def is_private_or_local(url):
    """Keeps the scraper away from localhost, routers, and internal networks."""
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False  # it's a domain name, not an IP


def hostname_is_ip(hostname):
    """True for raw IPv4 or IPv6 addresses."""
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def sanitize_for_spreadsheet(value):
    """Stops Excel from running a cell as a formula if it starts with =, +, - or @."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def sanitize_row(row):
    return {key: sanitize_for_spreadsheet(value) for key, value in row.items()}


# Phishing pages love free platforms (Netlify, GitHub Pages, etc.). An old
# platform domain says nothing about the attacker's page, so we flag these
# directly instead of trusting domain age.
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


def subdomain_entropy(url):
    """Scores how random the subdomain looks. Phishing kits generate junk like
    '3ib64a1sok-k9t7f7se' — real words score low, gibberish scores high."""
    hostname = urlparse(url).hostname or ''

    # on free hosting, measure the attacker's chosen name, not "www" or the platform
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


def fetch_page(url, timeout=5):
    """Downloads and parses a page. Returns None if it's unreachable or off-limits."""
    if is_private_or_local(url):
        print(f"Skipping {url} — points to a private/internal address")
        return None

    try:
        response = get_session().get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None
    return BeautifulSoup(response.text, 'html.parser')


def extract_features(url, soup):
    """Turns a page into the signals we judge it on."""
    parsed = urlparse(url)
    features = {"url": url}

    features['fetch_succeeded'] = soup is not None
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_at_symbol'] = '@' in url
    features['uses_https'] = parsed.scheme == 'https'
    features['has_ip_as_domain'] = hostname_is_ip(parsed.hostname)
    features['on_free_hosting'] = is_on_free_hosting(url)
    features['subdomain_entropy'] = round(subdomain_entropy(url), 2)

    if soup:
        forms = soup.find_all('form')
        features['num_forms'] = len(forms)
        features['has_password_field'] = bool(soup.find('input', {'type': 'password'}))

        # flag any form that ships its data to a different host — this catches
        # phishing pages that harvest credentials for someone else's server,
        # including protocol-relative actions like "//evil.com/submit"
        page_host = parsed.hostname
        external_form = False
        for form in forms:
            action_host = urlparse(form.get('action', '')).hostname
            if action_host and action_host != page_host:
                external_form = True
        features['form_posts_externally'] = external_form
    else:
        features['num_forms'] = 0
        features['has_password_field'] = False
        features['form_posts_externally'] = False

    return features


def get_registrable_domain(hostname):
    """The domain a WHOIS server would recognize: 'blog.mail.co.uk' -> 'mail.co.uk'."""
    if _tld_extractor is not None:
        ext = _tld_extractor(hostname)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    return hostname[4:] if hostname.startswith('www.') else hostname


def get_domain_age_days(url):
    """How old the domain is, in days. Tries RDAP first, falls back to WHOIS."""
    if is_private_or_local(url):
        return None

    domain = get_registrable_domain(urlparse(url).netloc)

    age = _get_domain_age_rdap(domain)
    if age is not None:
        return age

    return _get_domain_age_whois(domain)


def _get_domain_age_rdap(domain, timeout=6):
    # RDAP is the modern JSON replacement for WHOIS — much easier to parse reliably
    try:
        response = requests.get(f"https://rdap.org/domain/{domain}", timeout=timeout)
        response.raise_for_status()
        data = response.json()

        for event in data.get('events', []):
            if event.get('eventAction') == 'registration':
                date_str = event.get('eventDate')
                if date_str:
                    creation_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if creation_date.tzinfo is not None:
                        creation_date = creation_date.replace(tzinfo=None)
                    return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"RDAP lookup failed for {domain}: {e}")
    return None


def _get_domain_age_whois(domain):
    """Fallback when RDAP has no answer. Less reliable, but better than nothing."""
    if whois is None:
        return None
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = min(d for d in creation_date if d is not None)
        if creation_date:
            if creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=None)
            return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"WHOIS fallback also failed for {domain}: {e}")
    return None


def rule_based_score(features, domain_age_days):
    """The simple point system — higher means more suspicious."""
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

    # only trust "old domain = safe" off free hosting — the platform's
    # age doesn't tell us anything about the attacker's page there
    if not features['on_free_hosting']:
        if domain_age_days is not None and domain_age_days < 30:
            score += 3

    return score


def analyze_url(url):
    soup = fetch_page(url)
    features = extract_features(url, soup)
    domain_age_days = get_domain_age_days(url)

    features['domain_age_days'] = domain_age_days
    features['domain_age_known'] = int(domain_age_days is not None)
    features['risk_score'] = rule_based_score(features, domain_age_days)

    return features


def analyze_urls(urls, max_workers=5):
    """Analyzes a list of URLs in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(analyze_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Failed to analyze {url}: {e}")
    return results


def build_labeled_dataset(phishing_urls, legit_urls, max_workers=5):
    """Tags every row: 1 = phishing, 0 = legit, depending on which list it came from."""
    phishing_set = set(phishing_urls)
    all_urls = list(phishing_urls) + list(legit_urls)
    results = analyze_urls(all_urls, max_workers=max_workers)

    for r in results:
        r['label'] = 1 if r['url'] in phishing_set else 0

    return results


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
    phishing_urls = load_openphish_urls(limit=150)
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