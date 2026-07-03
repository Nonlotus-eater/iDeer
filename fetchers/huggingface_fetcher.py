"""
Fetch HuggingFace Daily Papers and Trending Models
"""

import requests
from bs4 import BeautifulSoup
import re


def _year_from_arxiv_id(arxiv_id: str) -> str:
    match = re.match(r"^(\d{2})(\d{2})\.", str(arxiv_id or ""))
    if not match:
        return ""
    yy = int(match.group(1))
    return str(1900 + yy if yy >= 91 else 2000 + yy)


def get_daily_papers(max_results: int = 50) -> list:
    """
    Fetch HuggingFace Daily Papers from API.

    Args:
        max_results: Maximum number of papers to return

    Returns:
        List[Dict]: Papers with id, title, abstract, authors, upvotes, paper_url
    """
    url = "https://huggingface.co/api/daily_papers"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to fetch daily papers: {e}")
        return []

    papers = []
    for item in data[:max_results]:
        paper = item.get("paper", {})
        paper_id = paper.get("id", "")
        title = paper.get("title", "No title available")
        abstract = paper.get("summary", "No abstract available")

        # Extract authors
        authors = paper.get("authors", [])
        author_names = [a.get("name", "") for a in authors if a.get("name")]

        upvotes = item.get("paper", {}).get("upvotes", 0)
        if upvotes == 0:
            upvotes = item.get("numComments", 0)

        paper_url = f"https://huggingface.co/papers/{paper_id}" if paper_id else ""
        arxiv_id = paper_id  # HuggingFace uses arXiv IDs

        paper_info = {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": author_names,
            "upvotes": upvotes,
            "year": _year_from_arxiv_id(arxiv_id),
            "venue_or_status": "HuggingFace Daily / arXiv preprint",
            "paper_url": paper_url,
            "arxiv_id": arxiv_id,
        }
        papers.append(paper_info)

    return papers


def get_trending_models(max_results: int = 30) -> list:
    """
    Fetch HuggingFace Trending Models by scraping the page.

    Args:
        max_results: Maximum number of models to return

    Returns:
        List[Dict]: Models with model_id, author, downloads, likes, tags, model_url
    """
    url = "https://huggingface.co/models?sort=trending"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch trending models page: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    models = []

    # Find model cards - they are usually in article tags or divs with specific classes
    model_cards = soup.find_all("article", class_=re.compile(r".*"))

    for card in model_cards[:max_results]:
        try:
            # Extract model link and ID
            link = card.find("a", href=re.compile(r"^/[^/]+/[^/]+$"))
            if not link:
                continue

            model_path = link.get("href", "").strip("/")
            if "/" not in model_path:
                continue

            parts = model_path.split("/")
            if len(parts) >= 2:
                author = parts[0]
                model_name = parts[1]
                model_id = model_path
            else:
                continue

            model_url = f"https://huggingface.co/{model_id}"

            # Extract description if available
            desc_elem = card.find("p")
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # Extract likes count
            likes = 0
            likes_elem = card.find(string=re.compile(r"^\d+$"))
            if likes_elem:
                try:
                    likes = int(likes_elem.strip())
                except:
                    pass

            # Look for likes in SVG heart icon context
            heart_spans = card.find_all("span")
            for span in heart_spans:
                text = span.get_text(strip=True)
                if text.isdigit():
                    likes = max(likes, int(text))

            # Extract downloads (if shown)
            downloads = 0
            download_text = card.find(string=re.compile(r"\d+[KMB]?"))
            if download_text:
                text = download_text.strip()
                if "K" in text:
                    try:
                        downloads = int(float(text.replace("K", "")) * 1000)
                    except:
                        pass
                elif "M" in text:
                    try:
                        downloads = int(float(text.replace("M", "")) * 1000000)
                    except:
                        pass

            # Extract tags
            tags = []
            tag_elements = card.find_all("span", class_=re.compile(r"tag|label|badge"))
            for tag_elem in tag_elements:
                tag_text = tag_elem.get_text(strip=True)
                if tag_text and len(tag_text) < 50:
                    tags.append(tag_text)

            model_info = {
                "model_id": model_id,
                "model_name": model_name,
                "author": author,
                "description": description,
                "downloads": downloads,
                "likes": likes,
                "tags": tags,
                "model_url": model_url,
            }
            models.append(model_info)

        except Exception as e:
            continue

    return models


def get_trending_models_api(max_results: int = 30) -> list:
    """
    Fetch HuggingFace Trending Models using the API (more reliable).

    Args:
        max_results: Maximum number of models to return

    Returns:
        List[Dict]: Models with model_id, author, downloads, likes, tags, model_url
    """
    # Use sort=likes with descending order to get popular models
    url = f"https://huggingface.co/api/models?sort=likes&direction=-1&limit={max_results}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Failed to fetch trending models from API: {e}")
        return []

    models = []
    for item in data[:max_results]:
        model_id = item.get("id", "")
        if "/" in model_id:
            author, model_name = model_id.split("/", 1)
        else:
            author = ""
            model_name = model_id

        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "author": author,
            "description": item.get("description", "") or item.get("cardData", {}).get("description", ""),
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "tags": item.get("tags", []) or item.get("pipeline_tag", []),
            "model_url": f"https://huggingface.co/{model_id}",
        }
        models.append(model_info)

    return models


if __name__ == "__main__":
    # Test daily papers
    print("=== Testing Daily Papers ===")
    papers = get_daily_papers(5)
    for p in papers:
        print(f"Title: {p['title'][:50]}...")
        print(f"Upvotes: {p['upvotes']}, URL: {p['paper_url']}")
        print()

    # Test trending models
    print("=== Testing Trending Models ===")
    models = get_trending_models_api(5)
    for m in models:
        print(f"Model: {m['model_id']}")
        print(f"Likes: {m['likes']}, Downloads: {m['downloads']}")
        print()
