"""Export a real FAISS HNSW run and measure candidate loss with Cairn.

Run after preparing SciFact and building its Cairn index. This deliberately uses a
small efSearch to make the audit interesting; it is not a tuned HNSW benchmark.
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from cairn_search import Cairn
from cairn_search.audit import read_fbin
from cairn_search.compare import compare_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("index", type=Path)
parser.add_argument("queries", type=Path)
parser.add_argument("--output", type=Path, default=Path("results/hnsw"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with Cairn(args.index) as client:
    layout = client.info["layout"]
vectors = np.memmap(
    args.index,
    mode="r",
    dtype="<f4",
    offset=layout["vectors_offset"],
    shape=(layout["rows"], layout["dimensions"]),
)
queries = np.asarray(read_fbin(args.queries), dtype=np.float64)
queries = np.asarray(
    queries / np.linalg.norm(queries, axis=1)[:, None], dtype=np.float32
)
faiss.omp_set_num_threads(1)
engine = faiss.IndexHNSWFlat(layout["dimensions"], 16, faiss.METRIC_INNER_PRODUCT)
engine.hnsw.efConstruction = 80
engine.hnsw.efSearch = 16
engine.add(vectors)
_, ids = engine.search(queries, min(40, layout["rows"]))
candidate_file = args.output / "candidates.jsonl"
with candidate_file.open("x") as f:
    for query_id, row in enumerate(ids):
        f.write(json.dumps({"query_id": query_id, "ids": row.tolist()}) + "\n")
report = compare_run(args.index, args.queries, candidate_file, k=10)
report["source_engine"] = {
    "name": "FAISS IndexHNSWFlat",
    "version": faiss.__version__,
    "M": 16,
    "efConstruction": 80,
    "efSearch": 16,
    "threads": 1,
    "candidate_count": 40,
    "note": "Deliberately small search budget; integration demonstration, not a tuned benchmark.",
}
(args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["summary"], indent=2))
