import os, requests
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("S2_API_KEY")
headers = {"x-api-key": key}
params = {
    "query": "speculative decoding LLM",
    "limit": 5,
    "fields": "title,abstract,year,citationCount,paperId",
}
r = requests.get(
    "https://api.semanticscholar.org/graph/v1/paper/search",
    headers=headers,
    params=params,
    timeout=15,
)
data = r.json()
for p in data["data"]:
    has_abstract = "YES" if p.get("abstract") else "NONE"
    print(f"abstract: {has_abstract} | {p['title'][:60]}")