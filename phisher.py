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


# ---------------------------------------------------------
# SITE CONFIGS — describe each site's structure here.
# Adding a new site = adding an entry, no new code needed.
# (Used for the generic pagination scraper below, e.g. the
# car-titles demo. NOT used for phishing feature extraction.)
# ---------------------------------------------------------
SITE_CONFIGS = {
    "webscraper_test": {
        "base_url": "https://webscraper.io/test-sites/pagination",
        "pagination_type": "query_param",   # e.g. ?page=2
        "page_param": "page",
        "title_tag": "h3",
        "title_class": None,
        "num_pages": 17,
    },
    # "books_toscrape": {
    #     "base_url": "https://books.toscrape.com/catalogue/page-{page}.html",
    #     "pagination_type": "path",          # e.g. /page-2.html
    #     "title_tag": "h3",
    #     "title_class": None,
    #     "num_pages": 50,
    # },
}


# ===========================================================
# PART 1: GENERIC PAGINATION SCRAPER (your existing project)
# ===========================================================

def build_page_url(config, page):
    """Builds the correct URL for a given page, based on the site's pagination style."""
    if config["pagination_type"] == "query_param":
        return f"{config['base_url']}?{config['page_param']}={page}"
    elif config["pagination_type"] == "path":
        return config["base_url"].format(page=page)
    else:
        raise ValueError(f"Unknown pagination_type: {config['pagination_type']}")


def get_titles(session, url, tag, css_class=None, timeout=5):
    """Fetches one page and extracts text from all matching elements."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    elements = soup.find_all(tag, class_=css_class) if css_class else soup.find_all(tag)
    return [el.get_text(strip=True) for el in elements]


def scrape_site(config, max_workers=5):
    """Generic engine: reads a config, scrapes every page, returns all titles."""
    urls = [build_page_url(config, page) for page in range(1, config["num_pages"] + 1)]
    all_titles = []

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"
        })
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(
                    get_titles, session, url, config["title_tag"], config["title_class"]
                ): url
                for url in urls
            }
            for future in as_completed(future_to_url):
                all_titles.extend(future.result())

    print(f"[{config['base_url']}] Total items collected: {len(all_titles)}")
    return all_titles


def save_to_csv(rows, filename='titles.csv', fieldnames=None):
    """
    Writes rows to CSV.
    - If rows are plain strings (e.g. titles), writes them under a single 'title' column.
    - If rows are dicts (e.g. phishing features), writes them as a proper table using fieldnames.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        if rows and isinstance(rows[0], dict):
            writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(file)
            writer.writerow(["title"])
            for row in rows:
                writer.writerow([row])
    print(f"Saved {len(rows)} rows to {filename}")


# ===========================================================
# PART 2: PHISHING FEATURE EXTRACTION (new)
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


if __name__ == '__main__':
    # --- Part 1 demo: generic pagination scraper (unchanged from before) ---
    car_titles = scrape_site(SITE_CONFIGS["webscraper_test"])
    car_titles.sort()
    save_to_csv(car_titles, "car_titles.csv")
    print(car_titles)

    # --- Part 2 demo: phishing feature extraction on a handful of test URLs ---
    # Replace these with real examples later (e.g. from PhishTank / UCI dataset)
    test_urls = [
        "https://www.google.com",
        "https://www.python.org",
    ]

    results = analyze_urls(test_urls)
    for r in results:
        print(r)

    save_to_csv(results, "phishing_features.csv")