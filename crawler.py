import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from indexer import index_document


def get_clean_text(soup):
    """
    Strips junk tags and returns clean readable text from a page.
    """
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    main = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id="content") or
        soup.find(class_="content") or
        soup.body
    )

    if not main:
        return ""

    lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
    return "\n".join(line for line in lines if len(line) > 30)


def make_doc_id(url):
    """
    Converts a URL into a clean doc_id string.
    e.g. https://docs.python.org/3/library/functions.html
         → docs.python.org_3_library_functions
    """
    parsed = urlparse(url)
    path   = parsed.path.strip("/").replace("/", "_").replace(".html", "").replace(".htm", "")
    domain = parsed.netloc.replace("www.", "")
    return f"{domain}_{path}" if path else domain


def crawl_and_index(root_url, max_pages=10, on_progress=None):
    """
    Crawls a website starting from root_url.
    Indexes each page as a separate document in ChromaDB.

    root_url    : starting URL
    max_pages   : hard cap on how many pages to crawl
    on_progress : optional callback(current, total, url) for live updates

    Returns a summary dict.
    """
    parsed_root = urlparse(root_url)
    base_domain = parsed_root.netloc  # only follow links on this domain

    visited   = set()   # URLs already crawled
    queue     = [root_url]
    indexed   = []      # successfully indexed pages
    failed    = []      # pages that errored
    skipped   = []      # pages with too little content

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"\nStarting crawl: {root_url}")
    print(f"Domain filter: {base_domain} | Max pages: {max_pages}\n")

    while queue and len(indexed) + len(failed) < max_pages:
        url = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        current_count = len(indexed) + len(failed) + 1
        print(f"[{current_count}/{max_pages}] Crawling: {url}")

        if on_progress:
            on_progress(current_count, max_pages, url)

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # skip non-HTML pages (PDFs, images etc)
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                skipped.append({"url": url, "reason": "not HTML"})
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            text = get_clean_text(soup)

            if len(text.split()) < 50:
                skipped.append({"url": url, "reason": "too little content"})
                print(f"  ↳ Skipped — not enough content")
                continue

            # index this page
            doc_id = make_doc_id(url)
            index_document(doc_id, text)
            indexed.append({"url": url, "doc_id": doc_id, "words": len(text.split())})
            print(f"  ↳ Indexed as '{doc_id}' ({len(text.split())} words)")

            # find new links on this page to add to queue
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()

                # skip anchors, mailto, javascript links
                if href.startswith(("#", "mailto:", "javascript:")):
                    continue

                # resolve relative URLs
                full_url = urljoin(url, href)
                parsed   = urlparse(full_url)

                # only follow links on the same domain
                if parsed.netloc != base_domain:
                    continue

                # strip fragments and query strings for dedup
                clean_url = parsed._replace(fragment="", query="").geturl()

                if clean_url not in visited:
                    queue.append(clean_url)

        except requests.exceptions.Timeout:
            failed.append({"url": url, "reason": "timeout"})
            print(f"  ↳ Failed — timeout")
        except requests.exceptions.HTTPError as e:
            failed.append({"url": url, "reason": str(e)})
            print(f"  ↳ Failed — {e}")
        except Exception as e:
            failed.append({"url": url, "reason": str(e)})
            print(f"  ↳ Failed — {e}")

    summary = {
        "root_url"     : root_url,
        "indexed_count": len(indexed),
        "failed_count" : len(failed),
        "skipped_count": len(skipped),
        "pages"        : indexed,
        "failed"       : failed,
    }

    print(f"\nCrawl complete.")
    print(f"  Indexed : {len(indexed)} pages")
    print(f"  Failed  : {len(failed)} pages")
    print(f"  Skipped : {len(skipped)} pages")

    return summary