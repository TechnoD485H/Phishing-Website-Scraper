import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# SITE CONFIGS — describe each site's structure here.
# Adding a new site = adding an entry, no new code needed.
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
    "books_toscrape": {
        "base_url": "https://books.toscrape.com/catalogue/page-{page}.html",
        "pagination_type": "path",          # e.g. /page-2.html
        "title_tag": "h3",
        "title_class": None,
        "num_pages": 50,
    },
}


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


if __name__ == '__main__':
    car_titles = scrape_site(SITE_CONFIGS["webscraper_test"])
    print(car_titles[:5])

    book_titles = scrape_site(SITE_CONFIGS["books_toscrape"])
    print(book_titles[:5])