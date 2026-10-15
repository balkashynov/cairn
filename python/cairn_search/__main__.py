from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import write_fbin


def main():
    p = argparse.ArgumentParser(
        description="Cairn import, embedding and recall-audit workbench"
    )
    sub = p.add_subparsers(dest="command", required=True)
    imp = sub.add_parser(
        "import-npy", help="Convert a float32 NumPy matrix into a new fbin"
    )
    imp.add_argument("input", type=Path)
    imp.add_argument("output", type=Path)
    emb = sub.add_parser(
        "embed",
        help="Explicitly send JSONL text records to Cohere and export float embeddings",
    )
    emb.add_argument("input", type=Path)
    emb.add_argument("output", type=Path)
    emb.add_argument("--model", default="embed-v5.0-pro")
    emb.add_argument("--dimension", type=int, default=1024)
    emb.add_argument(
        "--input-type", choices=["search_query", "search_document"], required=True
    )
    aud = sub.add_parser(
        "audit", help="Measure recall, losses and latency against exact search"
    )
    aud.add_argument("index", type=Path)
    aud.add_argument("queries", type=Path)
    aud.add_argument("--output", type=Path, required=True)
    aud.add_argument("--k", type=int, default=10)
    aud.add_argument("--limit", type=int)
    aud.add_argument("--repeats", type=int, default=3)
    aud.add_argument("--tag", type=int)
    aud.add_argument("--oversamples", default="1,4,16")
    aud.add_argument("--query-metadata", type=Path)
    aud.add_argument("--corpus-metadata", type=Path)
    aud.add_argument("--qrels", type=Path)
    aud.add_argument("--faiss", action="store_true")
    aud.add_argument("--dataset", default="Local embedding audit")
    aud.add_argument(
        "--source-note",
        default="User-supplied embeddings; model provenance not specified.",
    )
    cmp = sub.add_parser(
        "compare-run",
        help="Check an existing ANN result file against Cairn exact ground truth",
    )
    cmp.add_argument("index", type=Path)
    cmp.add_argument("queries", type=Path)
    cmp.add_argument("candidates", type=Path)
    cmp.add_argument("--output", type=Path, required=True)
    cmp.add_argument("--k", type=int, default=10)
    cmp.add_argument("--tag", type=int)
    cmp.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.command == "import-npy":
        import numpy as np

        x = np.load(a.input, mmap_mode="r", allow_pickle=False)
        if x.ndim != 2 or x.dtype.kind != "f":
            p.error("input must be a two-dimensional floating-point matrix")
        print(write_fbin(a.output, x))
    elif a.command == "embed":
        if a.output.exists() or a.output.with_suffix(".provenance.json").exists():
            p.error("output exists; choose a new export path")
        from .cohere import embed

        with a.input.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        vectors = embed(
            [r["text"] for r in rows],
            model=a.model,
            dimension=a.dimension,
            input_type=a.input_type,
        )
        shape = write_fbin(a.output, vectors)
        a.output.with_suffix(".provenance.json").write_text(
            json.dumps(
                {
                    "provider": "Cohere",
                    "model": a.model,
                    "dimension": a.dimension,
                    "input_type": a.input_type,
                    "source": str(a.input),
                    "rows": shape[0],
                    "embedding_type": "float",
                    "quantizer": "Cairn computes its own int8 at build time",
                },
                indent=2,
            )
        )
        print(a.output)
    elif a.command == "compare-run":
        from .compare import compare_run

        if a.output.exists():
            p.error("output exists; choose a new path")
        result = compare_run(
            a.index, a.queries, a.candidates, k=a.k, tag=a.tag, limit=a.limit
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result["summary"], indent=2))
    else:
        from .audit import audit
        from .report import write_report

        json_path = a.output.with_suffix(".json")
        html_path = a.output.with_suffix(".html")
        if json_path.exists() or html_path.exists():
            p.error("report output exists; choose a new output prefix")
        report = audit(
            a.index,
            a.queries,
            k=a.k,
            limit=a.limit,
            repeats=a.repeats,
            tag=a.tag,
            oversamples=tuple(int(x) for x in a.oversamples.split(",")),
            query_metadata=a.query_metadata,
            corpus_metadata=a.corpus_metadata,
            qrels=a.qrels,
            include_faiss=a.faiss,
        )
        report.update(dataset=a.dataset, source_note=a.source_note)
        # Keep short excerpts in the shareable report; full texts stay with the user's corpus.
        for doc in report["documents"].values():
            if "text" in doc:
                doc["text"] = doc["text"][:240]
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        write_report(report, html_path)
        print(json.dumps(report["summary"], indent=2))
        print(html_path)


if __name__ == "__main__":
    main()
