import requests
from bs4 import BeautifulSoup


def scrape_url(url):
    """
    Fetches a webpage and extracts clean readable text.
    Strips out nav, footer, scripts, ads — just the main content.
    Returns the text as a string.
    """
    print(f"Scraping: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The site may be slow or blocking scrapers.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP error: {e}")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to the URL. Check the link and try again.")

    soup = BeautifulSoup(response.text, "html.parser")

    # remove junk tags that don't have readable content
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # try to find the main content area first
    main = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id="content") or
        soup.find(class_="content") or
        soup.find(class_="post-body") or
        soup.body
    )

    if not main:
        raise Exception("Could not extract content from this page.")

    # get text, clean up excessive whitespace
    lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
    clean = "\n".join(line for line in lines if len(line) > 30)

    if len(clean.split()) < 50:
        raise Exception("Not enough content found on this page.")

    print(f"Scraped {len(clean.split())} words from {url}")
    return clean