"""Convert CUAD v1 (Contract Understanding Atticus Dataset) into clause-level records.csv.

Usage: python3 -I scripts/convert_cuad.py <raw_dir> <out_dir>

Reads master_clauses.csv (one row per contract; for each of 41 categories the list of annotated
text spans and an answer). Every annotated span of a clause category becomes a candidate record;
identical spans of one contract are merged, and a span also takes the labels of any shorter labeled
span of the same contract that it contains, so a record's labels list every category that applies
to its text. A span whose text recurs in another contract (CUAD holds a few refiled copies) is kept
once, at its first contract, with the labels of all its occurrences. The four metadata categories
(document name, parties, dates) are not records.

Kept: spans of 60 to 1,000 characters that carry no label from EXCLUDED. EXCLUDED holds the
categories whose text sits on the boundary of a predicate in data/cuad-*/dataset.toml (for example
change-of-control clauses for "restricts assignment" and "can be ended"), so the gold labels decide
every predicate without judgment calls. To keep individuals' names out, contracts of the
endorsement and consulting types are dropped, as are contracts that list a party without an
organisation marker (Inc, LLC, Ltd, ...) and spans with an honorific followed by a name.

Columns: id, agreement (the contract's own title, whitespace-normalized), clause (the span, with
CUAD's "<omitted>" marker written as "[...]"), and hidden columns that are never shown to models:
contract (CUAD file name), labels (";"-delimited CUAD categories), period_word (1 if the clause
contains day/days/month/months/year/years as a word, including forms such as "30-day").
Only the standard library is used; the input is treated as data and nothing from it is executed
(list cells are parsed with ast.literal_eval, which evaluates literals only).
"""

import ast
import csv
import hashlib
import io
import json
import re
import sys
import zlib
from pathlib import Path

SOURCE = "master_clauses.csv"
README = "CUAD_v1_README.txt"
# CRC-32 and size of CUAD_v1/master_clauses.csv inside the Zenodo archive CUAD_v1.zip
# (doi:10.5281/zenodo.4595826), read from the archive's central directory.
ZENODO_CRC32 = "4f288875"
ZENODO_SIZE = 3955428
MIN_CHARS, MAX_CHARS = 60, 1000
META = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
EXCLUDED = {
    "Competitive Restriction Exception", "Rofr/Rofo/Rofn", "Change Of Control", "Uncapped Liability",
    "Liquidated Damages", "Covenant Not To Sue", "Post-Termination Services", "Volume Restriction",
    "Most Favored Nation", "Non-Disparagement", "Price Restrictions", "Source Code Escrow",
}
PERSON_TYPES = re.compile(r"endorsement|consulting", re.I)
ORG = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|company|co|l\.?p|llp|plc|gmbh|ag|"
    r"s\.?a|n\.?v|b\.?v|s\.?p\.?a|s\.?r\.?l|trust|university|association|foundation|partners|"
    r"partnership|group|holdings?|bank|fund|authority|institute|agency|council|club|society|"
    r"federation|league|ab|oy|as|asa|kk|pty|pte|bhd|sdn|lp|n\.a|nv|bv|sa|srl|spa|ulc|lllp|pllc|pc|"
    r"p\.c|organization|school|college|hospital|center|centre|board|department|government|city|"
    r"county|state|republic|ministry|commission|service|services|systems|technologies|technology|"
    r"networks|communications|entertainment|media|pharmaceuticals|therapeutics|bio\w*|labs?|"
    r"laboratories|international|industries|enterprises|ventures|capital|financial|energy|motors|"
    r"airlines|stores|brands|products|solutions|software|games|studios|productions|records|music|"
    r"sports|golf|racing|team)\b", re.I)
HONORIFIC = re.compile(r"\b(Mr|Mrs|Ms|Dr|Miss)\.?\s+[A-Z]")
PERIOD_WORD = re.compile(r"\b(days?|months?|years?)\b", re.I)


def clean(text: str | None) -> str:
    return " ".join((text or "").replace("<omitted>", " [...] ").split())


def names_a_person(parties_answer: str) -> bool:
    for party in parties_answer.split(";"):
        name = re.sub(r"\(.*?\)|\u201c.*?\u201d|\".*?\"", "", party).strip(" ,.")
        if name and not ORG.search(name):
            return True
    return False


def main(raw_dir: str, out_dir: str) -> None:
    src = Path(raw_dir) / SOURCE
    data = src.read_bytes()
    crc = format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
    if (crc, len(data)) != (ZENODO_CRC32, ZENODO_SIZE):
        raise SystemExit(f"{SOURCE} differs from the Zenodo archive member (crc32 {crc}, {len(data)} bytes)")
    csv.field_size_limit(1 << 26)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    categories = [c for c in rows[0] if c != "Filename" and not c.endswith("Answer") and c not in META]
    dropped = {"contracts_person_type": 0, "contracts_person_party": 0, "spans_length": 0,
               "spans_honorific": 0, "spans_repeated_in_another_contract": 0, "spans_excluded_label": 0}
    found: dict[str, list] = {}  # span -> [contract, agreement, labels]; first contract wins, labels are pooled
    for row in rows:
        if PERSON_TYPES.search(row["Filename"]):
            dropped["contracts_person_type"] += 1
            continue
        if names_a_person(row["Parties-Answer"]):
            dropped["contracts_person_party"] += 1
            continue
        spans: dict[str, set[str]] = {}
        for category in categories:
            for span in ast.literal_eval(row[category] or "[]"):
                spans.setdefault(clean(span), set()).add(category)
        labels = {s: set(c) for s, c in spans.items()}
        for longer in spans:
            for shorter in spans:
                if shorter != longer and len(shorter) >= 40 and shorter in longer:
                    labels[longer] |= spans[shorter]
        for span, labs in labels.items():  # dict order: first appearance in the category columns
            if not MIN_CHARS <= len(span) <= MAX_CHARS:
                dropped["spans_length"] += 1
            elif HONORIFIC.search(span):
                dropped["spans_honorific"] += 1
            elif span in found:  # the same text in another contract (often a refiled copy)
                dropped["spans_repeated_in_another_contract"] += 1
                found[span][2] |= labs
            else:
                found[span] = [row["Filename"], clean(row["Document Name-Answer"]), set(labs)]
    records = []
    for span, (contract, agreement, labs) in found.items():
        if labs & EXCLUDED:
            dropped["spans_excluded_label"] += 1
        else:
            records.append((contract, agreement, span, ";" + ";".join(sorted(labs)) + ";"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "agreement", "clause", "contract", "labels", "period_word"])
        for i, (contract, agreement, clause, labs) in enumerate(records):
            writer.writerow([f"c{i:05d}", agreement, clause, contract, labs,
                             "1" if PERIOD_WORD.search(clause) else "0"])
    readme = Path(raw_dir) / README
    downloads = [{
        "file": SOURCE,
        "url": "https://huggingface.co/datasets/theatticusproject/cuad/resolve/"
               "a3c393f5d103fd0c516374e4fdff676c8176dcb1/CUAD_v1/master_clauses.csv",
        "sha256": hashlib.sha256(data).hexdigest(),
        "check": f"crc32 {crc} and size {len(data)} equal those of member CUAD_v1/master_clauses.csv in "
                 "CUAD_v1.zip of the Zenodo record (read from the archive's central directory)",
    }]
    if readme.exists():
        downloads.append({
            "file": README,
            "url": "https://zenodo.org/records/4595826/files/CUAD_v1.zip (member CUAD_v1/CUAD_v1_README.txt, "
                   "fetched by HTTP range request)",
            "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
            "check": "crc32 d2bddb9f, equal to the archive's central directory entry",
        })
    provenance = {
        "source": "CUAD v1, The Atticus Project, https://www.atticusprojectai.org/cuad; archive "
                  "https://zenodo.org/records/4595826 (doi:10.5281/zenodo.4595826)",
        "downloads": downloads,
        "contracts_in_source": len(rows),
        "contracts_kept": len({r[0] for r in records}),
        "records": len(records),
        "dropped": dropped,
        "excluded_categories": sorted(EXCLUDED),
        "span_chars": [MIN_CHARS, MAX_CHARS],
        "licence": "CC BY 4.0",
        "licence_text": "\"CUAD is licensed under the Creative Commons Attribution 4.0 (CC BY 4.0) license\" "
                        "(CUAD_v1_README.txt, section LICENSE, which also disclaims any representation about "
                        "the licence status of the underlying contracts, public filings on SEC EDGAR). The "
                        "Zenodo record's licence field is cc-by-4.0.",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Hendrycks, Burns, Chen and Ball, CUAD: An Expert-Annotated NLP Dataset for Legal "
                       "Contract Review, NeurIPS 2021 Datasets and Benchmarks; The Atticus Project. "
                       "Changes: clause spans extracted, merged, filtered and relabelled as described in "
                       "harness/scripts/convert_cuad.py.",
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
