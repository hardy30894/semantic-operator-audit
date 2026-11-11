"""Convert the Abt-Buy benchmark (DeepMatcher textual version) into the harness's records.csv.

Usage: python3 -I scripts/convert_abt_buy.py <exp_data_dir> <out_dir>

Records from Abt get ids a<id> and source "abt"; records from Buy get b<id> and source "buy". Gold
entities are the connected components of the positive pairs in train, valid and test; a record with
no positive pair is its own entity. Only the standard library is used; the input is treated as
data and nothing from it is executed.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

MAX_DESCRIPTION = 300


def clean(text: str | None, limit: int | None = None) -> str:
    text = " ".join((text or "").split())
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return text


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def main(exp_dir: str, out_dir: str) -> None:
    exp = Path(exp_dir)
    tables = {"a": ("abt", read(exp / "tableA.csv")), "b": ("buy", read(exp / "tableB.csv"))}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    positives = 0
    for split in ("train.csv", "valid.csv", "test.csv"):
        for row in read(exp / split):
            if row["label"] == "1":
                positives += 1
                ra, rb = find(f"a{row['ltable_id']}"), find(f"b{row['rtable_id']}")
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cluster", "source", "title", "description", "price"])
        for prefix, (source, rows) in tables.items():
            for r in rows:
                rid = f"{prefix}{r['id']}"
                cluster = find(rid)
                sizes[cluster] = sizes.get(cluster, 0) + 1
                writer.writerow([rid, cluster, source, clean(r["name"]),
                                 clean(r["description"], MAX_DESCRIPTION), clean(r["price"])])
    provenance = {
        "source": "Abt-Buy (DeepMatcher textual), http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/",
        "archive": "abt_buy_exp_data.zip",
        "sha256": {name: hashlib.sha256((exp / name).read_bytes()).hexdigest()
                   for name in ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv")},
        "records": sum(len(rows) for _, rows in tables.values()),
        "positive_pairs": positives,
        "entities_with_2_or_more_records": sum(1 for n in sizes.values() if n >= 2),
        "max_description_chars": MAX_DESCRIPTION,
        "licence": "not stated on the download page; cite Koepcke et al. 2010 and DeepMatcher, do not redistribute",
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
