# Cairn

**Quantized search with exact top-k recovery. Find what your ANN pipeline missed.**

Cairn is a Rust library and local CLI for retrieval engineers who need exact nearest-neighbor ground truth or an auditable search path over a static corpus. It scans compact int8 codes, bounds the error of every score, and reads original vectors only when they could still enter the answer. No candidate-count tuning is needed in certified mode.

The accompanying Python workbench imports existing embeddings, audits fixed candidate budgets, compares saved ANN runs, and generates an interactive report. Everything runs locally; the optional Cohere adapter is the only component that calls a model API.

**Status:** working engineering prototype. Exactness is relative to stored normalized float32 vectors and the documented float64 reference score. It is not a guarantee of semantic relevance. See [the algorithm and numerical contract](docs/DESIGN.md).

[**Explore the measured demo**](https://balkashynov.github.io/cairn/) · [Algorithm](docs/DESIGN.md) · [Reproducible benchmarks](docs/BENCHMARKS.md)

On an Apple M3 Pro, Cairn matched all **2,124 / 2,124** benchmark top-10 rankings. At **999,999 vectors x 768 dimensions**, native median search was **23.3 ms** versus **51.2 ms** for single-query, single-thread FAISS, scoring **0.15%** of original vectors on average. Large-run tail latency was variable, and FAISS batching wins on the smaller datasets. [Full measurements and limits](docs/BENCHMARKS.md).

## Try it in one minute

Requires Rust 1.93+ and Python 3.10+.

```bash
cargo build --release --locked
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python examples/quickstart.py
```

The example constructs 100 vectors with identical binary signs. Fixed binary candidates miss row 99; certified search returns it. It uses no network, model weights, or API key.

```text
binary:    nearest row 0,  approximate
certified: nearest row 99, exact
```

A second regression covers int8: 100 almost identical vectors quantize to the same codes, and a 4x candidate budget drops the closest one. Certified search detects the ambiguity and scores all 100. Worst-case fallback is intentional, not hidden.

## Search your embeddings

```bash
python -m pip install -e '.[workbench]'
cairn-workbench import-npy documents.npy documents.fbin
./target/release/cairn build documents.fbin documents.cairn
./target/release/cairn search documents.cairn queries.fbin --k 10
```

```python
from cairn_search import Cairn

with Cairn("documents.cairn") as index:
    result = index.search(query_embedding, k=10)
    print(result["hits"])
    print(result["stats"]["exact_scored"], "full vectors scored")
```

The binary also works without Python. `serve` accepts JSON lines over stdin and writes one response per line; it opens no network listener. Set `CAIRN_BIN` if the binary is outside the checkout or install it with `cargo install --path . --locked`.

## Audit the tradeoff

```bash
cairn-workbench audit documents.cairn queries.fbin \
  --oversamples 1,4,16 --repeats 3 --output audit
```

Open `audit.html`. It shows neighbor recall, missed IDs, per-query candidates, full-vector work, latency and optional relevance nDCG. Add `--query-metadata queries.jsonl --corpus-metadata documents.jsonl --qrels qrels.json` to inspect actual text and relevance judgments. Metadata records use `_id`, `title` and `text`; qrels use `query_id`, `corpus_id`, `score`. Add `--faiss` after installing `.[benchmark]` for an optimized baseline.

Already using FAISS, Qdrant, USearch, or another retriever? Export the returned row IDs, in rank order:

```json
{"query_id":0,"ids":[81,7,15,99,4]}
{"query_id":1,"ids":[32,10,17,22,8]}
```

```bash
cairn-workbench compare-run documents.cairn queries.fbin ann-run.jsonl \
  --k 5 --output comparison.json
```

A runnable [FAISS HNSW example](examples/audit_faiss.py) exports a real candidate pool and audits it:

```bash
python examples/audit_faiss.py data/scifact/index.cairn data/scifact/queries.fbin
```

With its deliberately small search budget, 105 of 300 SciFact queries missed at least one exact neighbor even in the 40-result candidate pool. The comparison identifies those losses without changing the source engine.

The comparison separates top-k ordering from candidate-pool coverage. A result missing from the candidate pool cannot be recovered by a later reranker. IDs are zero-based index row numbers; translate application IDs before exporting.

## Cohere

Cairn accepts float embeddings from any provider. [Cohere Embed 5](https://cohere.com/blog/embed-5) makes compression and first-stage retrieval especially relevant; its Pro/Fast models share an embedding space. Cairn's quantizer is **local symmetric int8**, not Cohere's API int8 encoding. The included binary baseline uses sign bits of floats, not a claim to reproduce every provider's binary output.

```bash
export COHERE_API_KEY=...  # only needed for explicit API calls
python examples/cohere_search.py
```

Or embed JSONL records with a `text` field:

```bash
cairn-workbench embed documents.jsonl documents.fbin \
  --input-type search_document --model embed-v5.0-pro --dimension 1024
cairn-workbench embed queries.jsonl queries.fbin \
  --input-type search_query --model embed-v5.0-pro --dimension 1024
```

The adapter preserves document/query input types, refuses silent truncation, validates dimensions and writes a provenance sidecar. Use compatible embedding spaces and dimensions; numeric vectors alone cannot establish model identity. Public benchmark results use precomputed **Embed English v3** and a historical Wikipedia corpus. Live Embed 5 behavior is not yet validated.

## Implementation

- Runtime-dispatched ARM NEON/SDOT and x86 AVX2 integer dot products, with a scalar fallback. No GPU required.
- Conservative residual-norm bounds, deterministic ties and exact fallback when candidates are ambiguous.
- One immutable memory-mapped file, checked offsets, SHA-256 integrity, deep invariant validation and atomic no-clobber publication.
- Equality filters applied before selection. Pass one little-endian `u64` tag per document to `build --tags tags.u64`; query with `--tag 7`. These tags are not an authorization system.
- Bounded top-k heaps and one `f64` bound per corpus row. Query files stream; original vectors are never copied wholesale into the Rust heap.

```bash
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
python -m pytest -q
./target/release/cairn verify documents.cairn --deep
```

## Scope and tradeoffs

Cairn exhaustively scans compressed codes. It is aimed at static local corpora and reference-quality audits, not distributed billion-scale retrieval or mutable database workloads. Near ties can force scoring every original vector. Total disk size is larger than float32 alone because originals are retained. Memory mapping is not a hard resident-memory limit; the OS page cache decides residency. Use `verify --deep` for indexes from an untrusted producer, and never modify or truncate a file while it is mapped.

FAISS's batched BLAS path can substantially outperform this single-query engine for offline throughput. An ANN index can be faster when approximate results are acceptable. The [benchmark protocol](docs/BENCHMARKS.md) reports both advantages and limits, including Python roundtrip overhead.

Cohere's [BinaryVectorDB](https://github.com/cohere-ai/BinaryVectorDB), [SQLite-vec rescoring](https://github.com/asg017/sqlite-vec/pull/276), FAISS, USearch, and [RaBitQ](https://arxiv.org/abs/2405.12497) are important prior art. Cairn contributes a small auditable implementation and a practical loss-diagnosis workflow; it does not claim a new quantization algorithm. This is an independent project, not a Cohere product or endorsement.

MIT for code. Dataset-derived demonstration material retains its original attribution and terms; see [data provenance](docs/DATA.md).
