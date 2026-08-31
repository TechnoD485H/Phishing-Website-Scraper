import requests
from bs4 import BeautifulSoup

def scrape(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # errors out if the site returns a bad status (404, 500, etc.)
    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

if __name__ == '__main__':
    url = "https://www.youtube.com/"
    soup = scrape(url)
    if soup:
        print(soup.title)  # just print the page title, not the whole HTML dump

#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     print(soup)

