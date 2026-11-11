"""Convert the arXiv metadata sample written by fetch_arxiv.py into the harness's records.csv.

Usage: python3 -I scripts/convert_arxiv.py <raw_dir> <out_dir>

Reads <raw_dir>/queries.json and the Atom XML responses it lists. A paper returned by several
queries is kept once, at its first appearance. Authors, comments and journal references are not
read, so the records hold no personal names beyond any that appear in a title or abstract.
Abstracts are whitespace-normalized; arXiv caps them at 1,920 characters, so MAX_ABSTRACT is only a
guard. Papers whose primary category is an alias across arXiv's groups (math.ST = stat.TH,
math-ph = math.MP, math.IT = cs.IT, math.NA = cs.NA, eess.SY = cs.SY, econ.GN = q-fin.EC), or lies in
eess or stat.ML, are dropped: their "main field" is ambiguous between two groups (for example
statistics or mathematics, computer science or engineering), and the main-field predicates in
data/arxiv-*/dataset.toml are read from the primary category.

Columns: id (x + arXiv identifier), title, abstract, submitted (date of the first version), and
hidden columns that are never shown to models: arxiv_id, year, primary_category, primary_group
(arXiv's top-level group of the primary category), categories (";"-delimited, primary included),
query (the category and year of the query that returned the paper).
Only the standard library is used; the input is treated as data and nothing from it is executed.
Files that declare a DOCTYPE or entities are refused.
"""

import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_ABSTRACT = 2000
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PHYSICS = {"astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin",
           "nucl-ex", "nucl-th", "physics", "quant-ph"}
GROUPS = {"cs": "computer science", "math": "mathematics", "q-bio": "quantitative biology",
          "q-fin": "quantitative finance", "stat": "statistics", "econ": "economics",
          "eess": "electrical engineering and systems science"}
AMBIGUOUS_PRIMARY = {"math.ST", "stat.TH", "math-ph", "math.MP", "math.IT", "cs.IT", "math.NA", "cs.NA",
                     "eess.SY", "cs.SY", "econ.GN", "q-fin.EC", "stat.ML"}


def clean(text: str | None, limit: int | None = None) -> str:
    text = " ".join((text or "").split())
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return text


def group(category: str) -> str:
    archive = category.split(".")[0]
    return "physics" if archive in PHYSICS else GROUPS.get(archive, archive)


def entries(body: bytes):
    head = body[:2048].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in body.upper():
        raise SystemExit("refusing XML with a DOCTYPE or entity declarations")
    for e in ET.fromstring(body).findall("a:entry", NS):
        yield {
            "arxiv_id": re.sub(r"v\d+$", "", e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]),
            "title": clean(e.findtext("a:title", "", NS)),
            "abstract": clean(e.findtext("a:summary", "", NS), MAX_ABSTRACT),
            "published": e.findtext("a:published", "", NS),
            "primary": e.find("arxiv:primary_category", NS).get("term"),
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
        }


def main(raw_dir: str, out_dir: str) -> None:
    raw = Path(raw_dir)
    queries = json.loads((raw / "queries.json").read_text(encoding="utf-8"))
    papers: dict[str, dict] = {}
    sha = {}
    returned: set[str] = set()
    for q in queries:
        body = (raw / q["file"]).read_bytes()
        sha[q["file"]] = hashlib.sha256(body).hexdigest()
        if sha[q["file"]] != q["sha256"]:
            raise SystemExit(f"{q['file']} does not match the hash recorded when it was fetched")
        for p in entries(body):
            returned.add(p["arxiv_id"])
            if p["primary"] in AMBIGUOUS_PRIMARY or p["primary"].startswith("eess."):
                continue
            papers.setdefault(p["arxiv_id"], {**p, "query": f"{q['category']} {q['year']}"})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "title", "abstract", "submitted", "arxiv_id", "year", "primary_category",
                         "primary_group", "categories", "query"])
        for aid, p in papers.items():
            cats = [p["primary"]] + [c for c in p["categories"] if c != p["primary"]]
            writer.writerow([f"x{aid}", p["title"], p["abstract"], p["published"][:10], aid,
                             p["published"][:4], p["primary"], group(p["primary"]),
                             ";" + ";".join(cats) + ";", p["query"]])
    provenance = {
        "source": "arXiv API (https://export.arxiv.org/api/query), one query per category and year, "
                  "fetched by harness/scripts/fetch_arxiv.py",
        "queries": len(queries),
        "fetched_utc": [min(q["fetched_utc"] for q in queries), max(q["fetched_utc"] for q in queries)],
        "downloads": [{"file": "queries.json", "url": "written by fetch_arxiv.py (query manifest)",
                       "sha256": hashlib.sha256((raw / "queries.json").read_bytes()).hexdigest()}]
                     + [{"file": q["file"], "url": q["url"], "sha256": sha[q["file"]], "entries": q["entries"]}
                        for q in queries],
        "entries_returned": sum(q["entries"] for q in queries),
        "distinct_papers": len(returned),
        "dropped_ambiguous_primary": len(returned) - len(papers),
        "records": len(papers),
        "max_abstract_chars": MAX_ABSTRACT,
        "fields_not_read": ["authors", "comments", "journal references", "DOIs", "links"],
        "licence": "CC0 1.0 (descriptive metadata)",
        "licence_text": "\"You are free to use descriptive metadata about arXiv e-prints\" under the CC0 1.0 "
                        "Public Domain Declaration; the terms define descriptive metadata to include title, "
                        "abstract, authors, identifiers and classification terms (arXiv API Terms of Use). "
                        "The e-prints themselves (PDFs, sources) are not covered and are not used.",
        "licence_url": "https://info.arxiv.org/help/api/tou.html",
        "attribution": "Metadata courtesy of arXiv.org (not required under CC0).",
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items() if k != "downloads"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
