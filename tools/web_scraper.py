"""Simple web scraper placeholder using requests + bs4."""
import requests
from bs4 import BeautifulSoup

def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)
