"""Query-level recall audit against Cairn's exhaustive reference."""

from __future__ import annotations

import json
import math
import platform
import statistics
import struct
import time
from pathlib import Path

from .client import Cairn


def read_fbin(path, limit=None):
    import numpy as np

    path = Path(path)
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError("truncated fbin header")
    n, d = struct.unpack("<II", header)
    if not 1 <= d <= 65536 or path.stat().st_size != 8 + n * d * 4:
        raise ValueError("invalid fbin shape or size")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    return np.memmap(path, dtype="<f4", offset=8, shape=(n, d), mode="r")[:limit]


def read_jsonl(path):
    if path is None:
        return []
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * p
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def ndcg(hits, relevant, k):
    gains = [float(relevant.get(str(h["id"]), 0)) for h in hits[:k]]
    ideal = sorted((float(v) for v in relevant.values()), reverse=True)[:k]

    def dcg(scores):
        return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(scores))

    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator else None


def audit(
    index,
    queries,
    *,
    binary=None,
    k=10,
    oversamples=(1, 4, 16),
    tag=None,
    limit=None,
    query_metadata=None,
    corpus_metadata=None,
    qrels=None,
    repeats=3,
    include_faiss=False,
):
    if repeats < 1 or k < 1 or any(n < 1 for n in oversamples):
        raise ValueError("k, repeats and oversamples must be positive")
    queries = read_fbin(queries, limit)
    if not len(queries):
        raise ValueError("audit requires queries")
    if len(set(oversamples)) != len(oversamples):
        raise ValueError("oversamples must be unique")
    modes = [("exhaustive", "exhaustive", 1), ("certified", "certified", 1)]
    modes += [(f"binary_{n}x", "binary", n) for n in oversamples]
    modes += [(f"int8_{n}x", "int8", n) for n in oversamples]
    records, timing, roundtrips = (
        [],
        {m[0]: [] for m in modes},
        {m[0]: [] for m in modes},
    )
    qm = read_jsonl(query_metadata)
    cm = read_jsonl(corpus_metadata)
    rel = {}
    if qrels:
        if not qm or not cm:
            raise ValueError("qrels require query and corpus metadata")
        ids = {str(row.get("_id", i)): i for i, row in enumerate(cm)}
        for row in json.loads(Path(qrels).read_text()):
            if str(row["corpus_id"]) in ids and float(row["score"]) > 0:
                rel.setdefault(str(row["query_id"]), {})[
                    str(ids[str(row["corpus_id"])])
                ] = float(row["score"])
    # A single process maps the index throughout. Warm all methods, then rotate
    # method order per query and repetition so no method always runs first.
    with Cairn(index, binary=binary) as client:
        info = client.info
        if cm and len(cm) != info["layout"]["rows"]:
            raise ValueError("corpus metadata count does not match index")
        if qm and len(qm) < len(queries):
            raise ValueError("query metadata count does not match queries")
        for _, mode, over in modes:
            for q in queries[: min(3, len(queries))]:
                client.search(q, k=k, mode=mode, oversample=over, tag=tag)
        for qi, q in enumerate(queries):
            results = {}
            for repetition in range(repeats):
                shift = (qi + repetition) % len(modes)
                for label, mode, over in modes[shift:] + modes[:shift]:
                    start = time.perf_counter()
                    result = client.search(q, k=k, mode=mode, oversample=over, tag=tag)
                    roundtrips[label].append((time.perf_counter() - start) * 1000)
                    timing[label].append(result["stats"]["elapsed_ms"])
                    if repetition == 0:
                        results[label] = result
            ref = results["exhaustive"]
            gold = {h["id"] for h in ref["hits"]}
            row = {
                "query_id": qi,
                "query": qm[qi] if qi < len(qm) else None,
                "methods": {},
            }
            relevant = rel.get(str(qm[qi].get("_id", qi)), {}) if qi < len(qm) else {}
            for label, _, _ in modes:
                result = results[label]
                got = {h["id"] for h in result["hits"]}
                method = {
                    "recall_at_k": len(gold & got) / len(gold) if gold else 1.0,
                    "ordered_exact": result["hits"] == ref["hits"],
                    "missed_ids": sorted(gold - got),
                    **result,
                }
                if relevant:
                    method["ndcg_at_k"] = ndcg(result["hits"], relevant, k)
                row["methods"][label] = method
            if not row["methods"]["certified"]["ordered_exact"]:
                raise AssertionError(f"certification mismatch on query {qi}")
            records.append(row)
    summary = {}
    for label, _, _ in modes:
        r = [q["methods"][label] for q in records]
        summary[label] = {
            "recall_at_k": statistics.mean(x["recall_at_k"] for x in r),
            "queries_with_misses": sum(x["recall_at_k"] < 1 for x in r),
            "p50_ms": percentile(timing[label], 0.5),
            "p95_ms": percentile(timing[label], 0.95),
            "p99_ms": percentile(timing[label], 0.99),
            "roundtrip_p50_ms": percentile(roundtrips[label], 0.5),
            "mean_exact_vectors": statistics.mean(
                x["stats"]["exact_scored"] for x in r
            ),
            "mean_logical_float_bytes": statistics.mean(
                x["stats"]["logical_float_bytes"] for x in r
            ),
            "timed_samples": len(timing[label]),
        }
        ndcgs = [x["ndcg_at_k"] for x in r if "ndcg_at_k" in x]
        if ndcgs:
            summary[label]["ndcg_at_k"] = statistics.mean(ndcgs)
    document_ids = {
        h["id"] for row in records for m in row["methods"].values() for h in m["hits"]
    }
    # Metadata stays in the local report only. Never copy an entire corpus into it.
    documents = {str(i): cm[i] for i in document_ids if i < len(cm)}
    report = {
        "schema_version": 1,
        "environment": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "index": info,
        "protocol": {
            "k": k,
            "queries": len(queries),
            "repeats": repeats,
            "tag": tag,
            "cache": "warm; methods interleaved after warm-up",
            "latency": "native search excludes JSON IPC; roundtrip includes Python and IPC",
            "quality": "neighbor recall against stored normalized f32 exhaustive search; relevance nDCG separate",
            "io": "logical vector bytes are not measured physical disk reads or resident memory",
        },
        "summary": summary,
        "queries": records,
        "documents": documents,
    }
    if include_faiss:
        report["faiss"] = faiss_baseline(index, queries, records, k, tag, repeats)
    return report


def faiss_baseline(index, queries, records, k, tag, repeats):
    import faiss
    import numpy as np

    if tag is not None:
        raise ValueError("FAISS baseline currently supports unfiltered audits only")
    with Path(index).open("rb") as f:
        header = f.read(256)
    n, d, _, _, _, offset, _ = struct.unpack_from("<7Q", header, 8)
    x = np.memmap(index, dtype="<f4", mode="r", offset=offset, shape=(n, d))
    faiss.omp_set_num_threads(1)
    engine = faiss.IndexFlatIP(d)
    engine.add(x)
    q = queries.astype("float64")
    q = (q / np.linalg.norm(q, axis=1)[:, None]).astype("float32")
    for v in q[:3]:
        engine.search(v[None], min(k, n))
    timings, recalls = [], []
    for qi, v in enumerate(q):
        for rep in range(repeats):
            start = time.perf_counter()
            _, ids = engine.search(v[None], min(k, n))
            timings.append((time.perf_counter() - start) * 1000)
            if rep == 0:
                gold = {h["id"] for h in records[qi]["methods"]["exhaustive"]["hits"]}
                recalls.append(len(gold & set(ids[0].tolist())) / len(gold))
    # Also report the batch workload, where BLAS can win. It is not comparable
    # to one-at-a-time query latency, so never mix these numbers in a speedup.
    start = time.perf_counter()
    engine.search(q, min(k, n))
    batch_ms = (time.perf_counter() - start) * 1000
    return {
        "version": faiss.__version__,
        "index": "IndexFlatIP",
        "threads": 1,
        "query_batch_size": 1,
        "p50_ms": percentile(timings, 0.5),
        "p95_ms": percentile(timings, 0.95),
        "recall_at_k": statistics.mean(recalls),
        "timed_samples": len(timings),
        "all_queries_batch_total_ms": batch_ms,
        "all_queries_batch_per_query_ms": batch_ms / len(q),
        "note": "FAISS accumulates f32; near ties can differ from Cairn f64. All original vectors resident. Separate timing pass.",
    }
