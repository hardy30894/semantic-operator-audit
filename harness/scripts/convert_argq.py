"""Convert IBM-ArgQ-Rank-30kArgs (crowd arguments with stance and quality labels) into records.csv.

Usage: python3 -I scripts/convert_argq.py <raw_dir> <out_dir>

Reads train.csv, dev.csv and test.csv (30,497 arguments on 71 debate topics; each labeled by 10
annotators for stance and for quality). Kept: the 39 topics in TOPICS, and arguments whose stance
label is unanimous after IBM's reliability weighting (stance_WA_conf = 1), so the stance gold
needs no judgment call. Of the 32 topics left out, 26 touch on religion, sexuality and gender,
ethnicity, reproduction, death, weapons and violence, child welfare, organ sale or named conflicts;
the other 6 propose to "end", "abandon", "stop" or "close" something, which sits on the boundary of
the predicate "proposes to ban, prohibit or abolish".

Columns: id, topic, argument (whitespace-normalized, otherwise verbatim), and hidden columns that
are never shown to models: split, stance (pro or con), wa (weighted-average quality score in
[0, 1]), mace_p (MACE-P quality score), n_words (whitespace-separated tokens in the argument that
contain a letter or digit, so a lone dash is not a word).
Only the standard library is used; the input is treated as data and nothing from it is executed.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

SPLITS = ("train", "dev", "test")
# topic -> True if the topic proposes to ban, prohibit or abolish something
TOPICS = {
    "Homeschooling should be banned": True,
    "Payday loans should be banned": True,
    "We should abolish intellectual property rights": True,
    "We should abolish the Olympic Games": True,
    "We should abolish the three-strikes laws": True,
    "We should abolish zoos": True,
    "We should ban algorithmic trading": True,
    "We should ban cosmetic surgery": True,
    "We should ban cosmetic surgery for minors": True,
    "We should ban factory farming": True,
    "We should ban fast food": True,
    "We should ban human cloning": True,
    "We should ban naturopathy": True,
    "We should ban private military companies": True,
    "We should ban telemarketing": True,
    "We should ban the use of child actors": True,
    "We should ban whaling": True,
    "We should fight for the abolition of nuclear weapons": True,
    "We should prohibit flag burning": True,
    "Entrapment should be legalized": False,
    "Homeopathy brings more harm than good": False,
    "Intelligence tests bring more harm than good": False,
    "Social media brings more harm than good": False,
    "The use of public defenders should be mandatory": False,
    "We should adopt a multi-party system": False,
    "We should adopt a zero-tolerance policy in schools": False,
    "We should adopt an austerity regime": False,
    "We should adopt libertarianism": False,
    "We should fight urbanization": False,
    "We should introduce compulsory voting": False,
    "We should legalize cannabis": False,
    "We should limit executive compensation": False,
    "We should limit judicial activism": False,
    "We should oppose collectivism": False,
    "We should subsidize Wikipedia": False,
    "We should subsidize journalism": False,
    "We should subsidize space exploration": False,
    "We should subsidize student loans": False,
    "We should subsidize vocational education": False,
}
MIN_STANCE_CONF = 0.999


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def main(raw_dir: str, out_dir: str) -> None:
    raw = Path(raw_dir)
    rows = []
    for split in SPLITS:
        with open(raw / f"{split}.csv", newline="", encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    kept = [r for r in rows if r["topic"] in TOPICS and float(r["stance_WA_conf"]) >= MIN_STANCE_CONF
            and r["stance_WA"] in ("1", "-1")]
    missing = set(TOPICS) - {r["topic"] for r in rows}
    if missing:
        raise SystemExit(f"topics not found in the input: {sorted(missing)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "topic", "argument", "split", "stance", "wa", "mace_p", "n_words"])
        for i, r in enumerate(kept):
            argument = clean(r["argument"])
            words = [w for w in argument.split() if any(ch.isalnum() for ch in w)]
            writer.writerow([f"g{i:05d}", r["topic"], argument, r["set"],
                             "pro" if r["stance_WA"] == "1" else "con", r["WA"], r["MACE-P"], len(words)])
    base = ("https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k/resolve/"
            "590726b3765b1b90c5e53a17e3b1f77d92d3aa8a/")
    files = [f"{s}.csv" for s in SPLITS] + [f for f in ("README.md",) if (raw / f).exists()]
    provenance = {
        "source": "IBM Debater IBM-ArgQ-Rank-30kArgs (Gretz et al., AAAI 2020), "
                  "https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k",
        "downloads": [{"file": f, "url": base + f, "sha256": hashlib.sha256((raw / f).read_bytes()).hexdigest()}
                      for f in files],
        "arguments_in_source": len(rows),
        "topics_kept": len(TOPICS),
        "ban_topics": sum(TOPICS.values()),
        "min_stance_conf": MIN_STANCE_CONF,
        "records": len(kept),
        "licence": "CC BY-SA 3.0",
        "licence_text": "\"Released under CC-BY-SA 3.0\" (dataset card README.md, section Licensing "
                        "Information, after a copyright line for IBM 2014). The IBM Debater datasets page states "
                        "the datasets are \"Released under CC-BY-SA\" and asks that every copy or modified "
                        "version carry a notice saying so, with a link to "
                        "http://creativecommons.org/licenses/by-sa/3.0/.",
        "licence_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "licence_sources": [
            "https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k (README.md, "
            "revision 590726b3765b1b90c5e53a17e3b1f77d92d3aa8a; its metadata header says cc-by-3.0, its "
            "licensing section CC-BY-SA 3.0; the stricter CC BY-SA 3.0 is applied)",
            "https://web.archive.org/web/20221230115544/https://research.ibm.com/haifa/dept/vst/debating_data.shtml",
        ],
        "attribution": "Gretz, Friedman, Cohen-Karlik, Toledo, Lahav, Aharonov and Slonim, A Large-scale "
                       "Dataset for Argument Quality Ranking: Construction and Analysis, AAAI 2020; IBM "
                       "Research. This records.csv is a modified version (topic and stance-agreement filter, "
                       "added word counts) and is released under CC BY-SA 3.0.",
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
