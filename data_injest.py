import feedparser
import requests
import time
import json
from helper_function import *

def fetch_arxiv_papers(query, max_results=50, start=0):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results
    }

    response = requests.get(url, params=params)
    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        papers.append({
            "source": "arxiv",
            "search_query": query,
            "paper_id": entry.id,
            "title": entry.title.strip(),
            "authors": ", ".join([a.name for a in entry.authors]),
            "abstract": entry.summary.strip(),
            "published": entry.published,
            "categories": ", ".join([tag.term for tag in entry.tags]),
            "pdf_url": next(
                (link.href for link in entry.links if link.type == "application/pdf"),
                None
            ),
            "insight": insigth_extraction(entry.summary.strip())
        })

    time.sleep(3)
    return papers


queries = [
    'all:"machine learning"',
    'all:"large language model"',
    'all:"generative AI"',
    'all:"retrieval augmented generation" OR all:RAG',
    'all:"semantic search"',
    'all:"vector database"',
    'all:"sentence transformer"'
]

all_papers = []

for query in queries:
    papers = fetch_arxiv_papers(query, max_results=20)
    all_papers.extend(papers)


unique_papers = {paper["paper_id"]: paper for paper in all_papers}
final_papers = list(unique_papers.values())


with open("arxiv_papers.json", "w", encoding="utf-8") as f:
    json.dump(final_papers, f, indent=4, ensure_ascii=False)