import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_titles(session, url, tag='h3'):
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    return [el.get_text(strip=True) for el in soup.find_all(tag)]

def scrape_all_pages(base_url, tag='h3', page_param='page', num_pages=17, max_workers=5):
    urls = [f"{base_url}?{page_param}={page}" for page in range(1, num_pages + 1)]
    all_titles = []
    
    with requests.Session() as session:  # reuses one connection across all requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(get_titles, session, url, tag): i + 1
                for i, url in enumerate(urls)
            }
            
            for future in as_completed(future_to_page):
                titles = future.result()
                all_titles.extend(titles)
    
    print(f"Total items collected: {len(all_titles)}")
    return all_titles

if __name__ == '__main__':
    all_titles = scrape_all_pages(
        base_url="https://webscraper.io/test-sites/pagination",
        tag='h3',
        num_pages=17,
        max_workers=8
    )
    print(all_titles)

    #test