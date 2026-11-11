"""Fetch a fixed, stratified sample of arXiv metadata (title, abstract, dates, categories) as Atom XML.

Usage: python3 -I scripts/fetch_arxiv.py <raw_dir>

arXiv has no single download file for a small sample, so this script issues one arXiv API query per
(category, year): the first PER_QUERY papers listed in the category whose first version was
submitted in March of that year, oldest first. The queries are fixed below, so the sample is
reproducible up to later metadata edits (a new version can change a title or abstract); the
SHA-256 of every response is written to <raw_dir>/queries.json for the converter's provenance.

arXiv states that descriptive metadata (title, abstract, authors, identifiers, classification) is
available under CC0 1.0 (https://info.arxiv.org/help/api/tou.html). The terms ask for at most one
request every three seconds on a single connection; this script waits WAIT_S between requests.
Only the standard library is used; responses are saved as data and nothing from them is executed.
"""

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://export.arxiv.org/api/query"
CATEGORIES = [
    "cs.LG", "cs.CL", "cs.CV", "cs.CR",                     # computer science
    "math.PR", "math.NT", "math.AP",                        # mathematics
    "astro-ph.GA", "astro-ph.EP", "cond-mat.mtrl-sci",      # physics and astronomy
    "quant-ph", "physics.ao-ph",
    "q-bio.PE", "q-bio.NC",                                 # quantitative biology
    "econ.EM", "q-fin.ST",                                  # economics and finance
    "stat.ME",                                              # statistics
]
YEARS = [2018, 2020, 2022, 2024]
PER_QUERY = 25
WAIT_S = 3.5
TRIES = 3
AGENT = "semaudit-dataset-fetch/1.0 (research sample; standard-library urllib)"


def query_url(category: str, year: int) -> str:
    search = f"cat:{category} AND submittedDate:[{year}03010000 TO {year}03312359]"
    params = {"search_query": search, "start": 0, "max_results": PER_QUERY,
              "sortBy": "submittedDate", "sortOrder": "ascending"}
    return f"{API}?{urllib.parse.urlencode(params)}"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main(raw_dir: str) -> None:
    out = Path(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for category in CATEGORIES:
        for year in YEARS:
            url = query_url(category, year)
            name = f"{category}_{year}.xml"
            body = b""
            for attempt in range(TRIES):
                time.sleep(WAIT_S)
                body = fetch(url)
                # The API occasionally answers with an empty feed; retry before accepting a short page.
                if body.count(b"<entry>") >= PER_QUERY or attempt == TRIES - 1:
                    break
            (out / name).write_bytes(body)
            entries = body.count(b"<entry>")
            manifest.append({"file": name, "category": category, "year": year, "url": url,
                             "entries": entries, "sha256": hashlib.sha256(body).hexdigest(),
                             "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"{name}: {entries} entries", flush=True)
    (out / "queries.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
