"""Convert IBM Debater ArgKP (arguments matched to key points) into records.csv with gold clusters.

Usage: python3 -I scripts/convert_argkp.py <raw_dir> <out_dir>

Input: ArgKP v1 (Bar-Haim et al., ACL 2020) as IBM released it in the Key Point Analysis 2021 shared
task repository, folder kpm_data/ at commit 2f5e2d5a: arguments_{train,dev}.csv (6,515 crowd
arguments on 28 debate topics, a subset of IBM-ArgQ-Rank-30kArgs), key_points_{train,dev}.csv (243
key points written by an expert debater, at most 7 per topic and stance), labels_{train,dev}.csv
(24,093 labeled (argument, key point) pairs) and readme.txt, the release notes that state the
licence. Each argument was shown with every key point of its topic and stance to 8 crowd annotators.
A pair is labeled 1 if at least 60% of them matched it and 0 if at most 15% did; pairs in between
(undecided) are not in the release. The converter refuses an input file whose SHA-256 differs from
the release's.

Kept: arguments on the 17 topics in TOPICS that match exactly one key point (label 1) and carry
label 0 for every other key point of their topic and stance, so none of their pairs is undecided.
That key point is the argument's gold cluster: two records are the same entity when they make the
same key point.

Topics: the sensitive-topic exclusions of convert_argq.py. Its docstring gives counts but no list:
26 topics left out as sensitive, and 6 left out only because their verb (end, abandon, stop, close)
sits on the boundary of its ban-topic predicate. EXCLUDED reconstructs the list for ArgKP's topics.
Nine are clearly sensitive. The counts make one more topic with such a verb sensitive without naming
it; the two plausible ones, "We should close Guantanamo Bay detention camp" and "We should abandon
marriage", are both left out. "We should abandon the use of school uniform" and "We should end
mandatory retirement" touch none of its sensitive categories, and the ban-topic predicate is not
used here, so they are kept. The other 15 kept topics are among the 39 that convert_argq.py keeps.

Columns:
- shown to models: topic, argument (whitespace-normalized, spelling left as written);
- hidden, never shown: cluster (ArgKP id of the argument's key point: the gold entity), source (ArgKP
  id of the argument; every record is its own source, so any two arguments of one key point form a
  placement tuple, a self-join), stance (pro or con), n_words (whitespace-separated tokens that
  contain a letter or digit), split (train or dev in ArgKP), key_point (text of the key point),
  title (the topic and the argument joined by one space: semaudit.data.sample_blocks reads this
  field, and only it, to rank a block's negatives by token Jaccard similarity).
Only the standard library is used; the input is treated as data and nothing from it is executed.
"""

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = "https://github.com/IBM/KPA_2021_shared_task"
REVISION = "2f5e2d5ad5a387fe731da2f00b423ce1d845c5d2"
RAW_BASE = f"https://raw.githubusercontent.com/IBM/KPA_2021_shared_task/{REVISION}/kpm_data/"
SPLITS = ("train", "dev")
# SHA-256 of the release files at REVISION (each also matches its git blob id on GitHub).
EXPECTED_SHA256 = {
    "arguments_train.csv": "aa7eaa6b399bdc1b2b679e23185591c5bd59264234b1cba87a3c1df534db6235",
    "arguments_dev.csv": "e06b763c41606c3290ddfa414109b3d3d47331ddd322ff5e4dfa0079ec9afb1c",
    "key_points_train.csv": "dc9625378749b8e0f85ba9d5babec7823fa4d8c6a9da5906b42f3401c7feaf60",
    "key_points_dev.csv": "328beb1551afa445518ff97c6e6706249e470571dcdc81b72713e5c05a6ff73c",
    "labels_train.csv": "e6d866cdd426f63e143e4e9aea1d33d5c3eb9f06777fc29508cf54e1810319ab",
    "labels_dev.csv": "d321711a07ca449bd1fe438fcfc55bfa478d2fdf7e8b74bd7a7394f17aff7fc4",
    "readme.txt": "fe22917afd4031591aed53c3a40d35a8f3ec24756759bcc88efde0e5f9dbd1c5",
}
TOPICS = (
    # among the 39 topics convert_argq.py keeps
    "Homeschooling should be banned",
    "We should abolish intellectual property rights",
    "We should adopt an austerity regime",
    "We should adopt libertarianism",
    "We should ban human cloning",
    "We should ban private military companies",
    "We should ban the use of child actors",
    "We should fight for the abolition of nuclear weapons",
    "We should fight urbanization",
    "We should introduce compulsory voting",
    "We should legalize cannabis",
    "We should prohibit flag burning",
    "We should subsidize journalism",
    "We should subsidize space exploration",
    "We should subsidize vocational education",
    # left out by convert_argq.py only for its ban-topic predicate
    "We should abandon the use of school uniform",
    "We should end mandatory retirement",
)
# topic -> why it is left out (convert_argq.py's sensitive categories)
EXCLUDED = {
    "Assisted suicide should be a criminal offence": "sensitive: death",
    "The vow of celibacy should be abandoned": "sensitive: religion",
    "We should abolish capital punishment": "sensitive: death",
    "We should abolish the right to keep and bear arms": "sensitive: weapons and violence",
    "We should adopt atheism": "sensitive: religion",
    "We should end affirmative action": "sensitive: ethnicity",
    "We should legalize prostitution": "sensitive: sexuality",
    "We should legalize sex selection": "sensitive: reproduction",
    "We should prohibit women in combat": "sensitive: gender, weapons and violence",
    "We should close Guantanamo Bay detention camp": "possibly sensitive (a named conflict, detention): left out to be safe",
    "We should abandon marriage": "possibly sensitive (sexuality and gender): left out to be safe",
}
STANCES = {"1": "pro", "-1": "con"}


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def ordinal(item_id: str) -> tuple[int, int]:
    """arg_3_17 or kp_3_2 -> (3, 17): ArgKP ids are <kind>_<topic>_<index>."""
    _, topic, index = item_id.split("_")
    return int(topic), int(index)


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(raw_dir: str, out_dir: str) -> None:
    raw = Path(raw_dir)
    sha256 = {}
    for name, expected in EXPECTED_SHA256.items():
        sha256[name] = hashlib.sha256((raw / name).read_bytes()).hexdigest()
        if sha256[name] != expected:
            raise SystemExit(f"{raw / name}: SHA-256 {sha256[name]} differs from the release's {expected}")
    arguments = {r["arg_id"]: dict(r, split=s) for s in SPLITS for r in read(raw / f"arguments_{s}.csv")}
    key_points = {r["key_point_id"]: r for s in SPLITS for r in read(raw / f"key_points_{s}.csv")}
    labels = [r for s in SPLITS for r in read(raw / f"labels_{s}.csv")]

    topics = {r["topic"] for r in arguments.values()}
    if set(TOPICS) & set(EXCLUDED) or topics != set(TOPICS) | set(EXCLUDED):
        raise SystemExit(f"topics not classified: {sorted(topics ^ (set(TOPICS) | set(EXCLUDED)))}")
    side_points: dict[tuple[str, str], set[str]] = {}  # (topic, stance) -> its key points
    for kp, r in key_points.items():
        side_points.setdefault((r["topic"], r["stance"]), set()).add(kp)
    pair_labels: dict[str, dict[str, str]] = {}
    for r in labels:
        arg, kp = arguments[r["arg_id"]], key_points[r["key_point_id"]]
        if (arg["topic"], arg["stance"]) != (kp["topic"], kp["stance"]) or r["label"] not in ("0", "1"):
            raise SystemExit(f"unexpected label row: {r}")
        if r["key_point_id"] in pair_labels.setdefault(r["arg_id"], {}):
            raise SystemExit(f"duplicate label row: {r}")
        pair_labels[r["arg_id"]][r["key_point_id"]] = r["label"]

    kept: list[tuple[str, str]] = []  # (argument id, its key point id)
    dropped: Counter[str] = Counter()
    for arg_id in sorted(arguments, key=ordinal):
        arg = arguments[arg_id]
        if arg["topic"] in EXCLUDED:
            dropped["topic left out"] += 1
            continue
        got = pair_labels.get(arg_id, {})
        matches = [kp for kp, label in got.items() if label == "1"]
        if not matches:
            dropped["matches no key point"] += 1
        elif len(matches) > 1:
            dropped["matches two or more key points"] += 1
        elif set(got) != side_points[(arg["topic"], arg["stance"])]:
            dropped["matches one key point, undecided for another"] += 1
        else:
            kept.append((arg_id, matches[0]))
    texts = Counter(clean(arguments[a]["argument"]).lower() for a, _ in kept)
    if max(texts.values()) > 1:
        raise SystemExit("two kept arguments have the same text")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = ["id", "cluster", "source", "topic", "argument", "stance", "n_words", "split", "key_point", "title"]
    with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for i, (arg_id, kp) in enumerate(kept):
            arg = arguments[arg_id]
            argument = clean(arg["argument"])
            words = [w for w in argument.split() if any(ch.isalnum() for ch in w)]
            writer.writerow([f"k{i:05d}", kp, arg_id, arg["topic"], argument, STANCES[arg["stance"]], len(words),
                             arg["split"], clean(key_points[kp]["key_point"]), f"{arg['topic']} {argument}"])

    sizes = Counter(kp for _, kp in kept)
    kept_topic_points = [kp for kp, r in key_points.items() if r["topic"] in TOPICS]
    provenance = {
        "source": f"IBM Debater ArgKP v1 (Bar-Haim et al., ACL 2020), folder kpm_data/ of {REPO}",
        "revision": REVISION,
        "downloads": [{"file": name, "url": RAW_BASE + name, "sha256": sha256[name]} for name in EXPECTED_SHA256],
        "arguments_in_source": len(arguments),
        "key_points_in_source": len(key_points),
        "labeled_pairs_in_source": len(labels),
        "matching_pairs_in_source": sum(r["label"] == "1" for r in labels),
        "topics_in_source": len(topics),
        "topics_kept": len(TOPICS),
        "topics_left_out": EXCLUDED,
        "rule": "an argument is kept if it matches exactly one key point (label 1: at least 60% of 8 annotators) and "
                "has label 0 (at most 15%) for every other key point of its topic and stance; that key point is its "
                "cluster",
        "arguments_dropped": dict(dropped),
        "records": len(kept),
        "stance": dict(Counter(STANCES[arguments[a]["stance"]] for a, _ in kept)),
        "split": dict(Counter(arguments[a]["split"] for a, _ in kept)),
        "key_points_in_kept_topics": len(kept_topic_points),
        "clusters": len(sizes),
        "clusters_with_2_or_more_records": sum(n >= 2 for n in sizes.values()),
        "clusters_with_3_or_more_records": sum(n >= 3 for n in sizes.values()),
        "cluster_size": {"min": min(sizes.values()), "median": statistics.median(sizes.values()),
                         "max": max(sizes.values())},
        "licence": "CDLA-Sharing-1.0",
        "licence_text": "\"(c) Copyright IBM 2020. Released under Community Data License Agreement \u2013 Sharing, "
                        "Version 1.0 (https://cdla.io/sharing-1-0/).\" (kpm_data/readme.txt, the ArgKP v1 release "
                        "notes, under \"The dataset is released under the following licensing and copyright terms:\")",
        "licence_url": "https://cdla.dev/sharing-1-0/",
        "licence_sources": [
            f"{REPO}/blob/{REVISION}/kpm_data/readme.txt",
            "https://web.archive.org/web/20221230115544/https://research.ibm.com/haifa/dept/vst/debating_data.shtml "
            "(IBM's Debater datasets page: its datasets are released under CC-BY-SA \"unless specified otherwise in "
            "their release notes\"; ArgKP's release notes specify CDLA-Sharing-1.0)",
            f"{REPO}/blob/{REVISION}/LICENSE (Apache License 2.0, the repository's code licence; kpm_data/ carries "
            "its own)",
        ],
        "attribution": "Bar-Haim, Eden, Friedman, Kantor, Lahav and Slonim, From Arguments to Key Points: Towards "
                       "Automatic Argument Summarization, ACL 2020; (c) Copyright IBM 2020. The arguments are a "
                       "subset of IBM-ArgQ-Rank-30kArgs (Gretz et al., AAAI 2020).",
        "changes": "records.csv is a modified version of the ArgKP data (Enhanced Data under CDLA-Sharing-1.0), made by "
                   "harness/scripts/convert_argkp.py: 17 of the 28 topics kept; only arguments that match exactly one "
                   "key point with no undecided pair kept; each argument joined with its key point; whitespace "
                   "normalized; new ids; word counts and a similarity key added. A published copy must be under "
                   "CDLA-Sharing-1.0 with this notice.",
        "not_used": "test_data/ of the same repository (3 topics added in 2021, ArgKP-2021): its files carry no "
                    "licence notice of their own",
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
