"""Check a saved ANN run against an exact oracle, without replacing its search engine."""

import json
import statistics
from pathlib import Path

from .audit import read_fbin
from .client import Cairn


def compare_run(index, queries, candidates, *, k=10, tag=None, binary=None, limit=None):
    if k < 1:
        raise ValueError("k must be positive")
    query_vectors = read_fbin(queries, limit)
    if not len(query_vectors):
        raise ValueError("comparison requires queries")
    supplied = {}
    with Path(candidates).open() as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            qi = row.get("query_id")
            ids = row.get("ids")
            if type(qi) is not int or qi < 0 or qi in supplied:
                raise ValueError(f"invalid or duplicate query_id on line {line_number}")
            if (
                not isinstance(ids, list)
                or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(
                    f"ids must be unique nonnegative integer row IDs on line {line_number}"
                )
            supplied[qi] = ids
    if limit is None and set(supplied) != set(range(len(query_vectors))):
        raise ValueError("candidate query IDs must exactly match all query rows")
    output = []
    with Cairn(index, binary=binary) as client:
        n = client.info["layout"]["rows"]
        for qi, q in enumerate(query_vectors):
            if qi not in supplied:
                raise ValueError(f"missing candidate query {qi}")
            ids = supplied[qi]
            if any(i >= n for i in ids):
                raise ValueError(f"out-of-range document ID in query {qi}")
            exact = client.search(q, k=k, tag=tag)
            gold = [h["id"] for h in exact["hits"]]
            expected = set(gold)
            top = set(ids[:k])
            pool = set(ids)
            output.append(
                {
                    "query_id": qi,
                    "neighbor_recall_at_k": len(top & expected) / len(expected)
                    if expected
                    else 1.0,
                    "candidate_recall_at_k": len(pool & expected) / len(expected)
                    if expected
                    else 1.0,
                    "missed_top_k": sorted(expected - top),
                    "missing_from_candidate_pool": sorted(expected - pool),
                    "ordered_exact": ids[: len(gold)] == gold
                    and len(ids[:k]) == len(gold),
                    "reference": exact,
                }
            )
    return {
        "schema_version": 1,
        "k": k,
        "queries": len(output),
        "tag": tag,
        "summary": {
            "neighbor_recall_at_k": statistics.mean(
                r["neighbor_recall_at_k"] for r in output
            ),
            "candidate_recall_at_k": statistics.mean(
                r["candidate_recall_at_k"] for r in output
            ),
            "queries_with_missing_candidates": sum(
                bool(r["missing_from_candidate_pool"]) for r in output
            ),
            "ordered_exact_queries": sum(r["ordered_exact"] for r in output),
        },
        "results": output,
        "note": "IDs are index row numbers, in the retrieval engine returned order. Exactness is vector ranking, not relevance. Scores and source-engine latency are not inferred.",
    }
