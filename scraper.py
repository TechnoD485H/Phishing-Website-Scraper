import requests
from bs4 import BeautifulSoup

def scrape(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None
    return BeautifulSoup(response.text, 'html.parser')

def get_car_titles(url):
    soup = scrape(url)
    if not soup:
        return []
    
    titles = []
    for tag in soup.find_all('h3'): #inital test with 'a' tags giving negative result - 'None'. Most titles in h3.
        titles.append(tag.get_text(strip=True))
    return titles

if __name__ == '__main__':
    all_titles = []

    for page in range(1, 18):  # pages 1 to 17
        page_url = f"https://webscraper.io/test-sites/pagination?page={page}"
        titles = get_car_titles(page_url)
        all_titles.extend(titles)
        print(f"Page {page}: found {len(titles)} cars")

    print(f"\nTotal cars collected: {len(all_titles)}")
    print(all_titles)
        




#     url = "https://webscraper.io/test-sites/pagination"
#     car_titles = get_car_titles(url)
#     print(car_titles)

    # if soup:
    #     print(soup.a.title)  # just print the page title, not the whole HTML dump

#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     print(soup)

