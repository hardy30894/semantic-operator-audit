"""Convert WDC Products offers (multi-class split, 80% corner cases) into the harness's records.csv.

Usage: python3 -I scripts/convert_wdc_products.py <raw_dir> <out_dir>

Uses the large training split: 2,841 offers of 500 products, 3-10 offers per product, so every
product can seed a matching block with three of its offers. Descriptions are whitespace-normalized
and cut at a word boundary near 300 characters to bound prompt length. Only the standard library is
used; the input is treated as data and nothing from it is executed.
"""

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

SOURCE = "wdcproductsmulti80cc20rnd000un_train_large.json.gz"
MAX_DESCRIPTION = 300


def clean(text: str | None, limit: int | None = None) -> str:
    text = " ".join((text or "").split())
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return text


def main(raw_dir: str, out_dir: str) -> None:
    src = Path(raw_dir) / SOURCE
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        offers = [json.loads(line) for line in fh if line.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cluster", "source", "title", "description", "brand", "price"])
        for o in offers:
            price = f"{o['price']} {o.get('priceCurrency') or ''}".strip() if o.get("price") else ""
            writer.writerow([f"w{o['id']}", str(o["cluster_id"]), "", clean(o.get("title")),
                             clean(o.get("description"), MAX_DESCRIPTION), clean(o.get("brand")), price])
    clusters = {o["cluster_id"] for o in offers}
    provenance = {
        "source": "WDC Products, https://webdatacommons.org/largescaleproductcorpus/wdc-products/",
        "archive": "80multi.zip",
        "file": SOURCE,
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "records": len(offers),
        "clusters": len(clusters),
        "max_description_chars": MAX_DESCRIPTION,
        "licence": "not stated on the download page; cite the WDC Products paper and do not redistribute",
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
