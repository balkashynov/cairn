#!/usr/bin/env python3
"""Publish numeric evidence and a small attributed demo from full local audits."""

import argparse
import copy
import gzip
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from cairn_search.report import write_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
args = parser.parse_args()
destination = Path("benchmarks")
(destination / "provenance").mkdir(parents=True, exist_ok=True)
(destination / "numeric").mkdir(exist_ok=True)
Path("demo").mkdir(exist_ok=True)
summary = {}
checksums = {}
for path in sorted(args.results.glob("*.json")):
    report = json.loads(path.read_text())
    if "summary" not in report or "protocol" not in report:
        continue
    assert len(report["queries"]) == report["protocol"]["queries"]
    assert all(q["methods"]["certified"]["ordered_exact"] for q in report["queries"])
    numeric = copy.deepcopy(report)
    numeric.pop("documents", None)
    for query in numeric["queries"]:
        query.pop("query", None)
    payload = json.dumps(numeric, separators=(",", ":"), allow_nan=False).encode()
    output = destination / "numeric" / (path.stem + ".json.gz")
    output.write_bytes(gzip.compress(payload, mtime=0))
    checksums[str(output)] = hashlib.sha256(output.read_bytes()).hexdigest()
    summary[path.stem] = {k: v for k, v in numeric.items() if k != "queries"}
    summary[path.stem]["ordered_exact_queries"] = len(numeric["queries"])
    provenance = Path("data") / path.stem / "provenance.json"
    shutil.copyfile(provenance, destination / "provenance" / (path.stem + ".json"))

    if path.stem == "scifact":
        demo = copy.deepcopy(report)
        # Deliberately include failures, successes and int8 near misses. The
        # aggregate table remains computed over all 300 queries, not this subset.
        ordered = sorted(
            report["queries"],
            key=lambda q: (
                -len(q["methods"]["binary_4x"]["missed_ids"]),
                q["query_id"],
            ),
        )
        chosen = ordered[:6] + ordered[-3:]
        for query in report["queries"]:
            if query["methods"]["int8_1x"]["missed_ids"] and query not in chosen:
                chosen.append(query)
                if len(chosen) == 12:
                    break
        demo["queries"] = sorted(chosen, key=lambda q: q["query_id"])
        demo["verified_queries"] = len(report["queries"])
        demo["mean_eligible"] = statistics.mean(
            q["methods"]["certified"]["stats"]["eligible"] for q in report["queries"]
        )
        ids = {
            str(hit["id"])
            for q in demo["queries"]
            for method in q["methods"].values()
            for hit in method["hits"]
        }
        demo["documents"] = {k: v for k, v in demo["documents"].items() if k in ids}
        demo["display_note"] = (
            "The table and headline metrics cover all 300 SciFact test queries. "
            "The explorer shows 12 deliberately selected examples of binary misses, "
            "int8 near misses and successful retrieval. It is a replay of measured results."
        )
        demo["source_note"] += (
            " SciFact: Wadden et al., Allen Institute for AI, EMNLP 2020. "
            "Claims: CC BY 4.0. Abstract excerpts: S2ORC, ODC-By 1.0. "
            "Excerpts are shortened; original source IDs are shown."
        )
        write_report(demo, "demo/index.html")
        Path("demo/.nojekyll").touch()

for path in args.results.glob("memory-*.txt"):
    shutil.copyfile(path, destination / path.name)
(destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(destination / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
print(f"Packaged {len(summary)} datasets, numeric results and demo/index.html")
