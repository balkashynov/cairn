# Cairn

**See which close matches your vector search left out.**

Cairn is a local tool for engineers checking changes to document search. Give it your embeddings (the number lists used to compare documents) and the results your search returned. It computes an exact reference ranking and reports which close matches were missing. You can keep your existing search engine.

[**Walk through one real search in your browser**](https://balkashynov.github.io/cairn/#example) · [How the engine works](https://balkashynov.github.io/cairn/#how-it-works) · [Benchmark evidence](docs/BENCHMARKS.md)

## A concrete example

An engineer is testing compressed search over 5,183 scientific papers. One query asks about classroom and web-based collaborative learning:

1. **Run the faster search.** It uses binary compressed vectors to select 40 candidates, scores their original vectors, and returns ten papers. The list looks complete.
2. **Check against the exact ranking.** Cairn finds that only six of the closest ten survived. Four close matches never appear in the returned list.
3. **Inspect a miss.** A paper ranked fifth by the original-vector scores is absent. A later reranker cannot recover a document that was left out of its candidate pool.
4. **Check the impact.** Both methods fully score 40 originals, and both have a relevance score of 1.000 for this query. The audit finds a numerical ranking difference; it does not establish a better answer or less full-vector work.
5. **Try another setting.** A fixed int8 shortlist of 40 also recovers all ten. In fact, it recovers every exact neighbor across all 300 queries in this dataset. The useful output is evidence for choosing a setting, not a reason to always replace approximate search.

These are saved SciFact measurements, replayed in the [guided demo](https://balkashynov.github.io/cairn/#example). The page includes other queries and full result lists after the walkthrough. It does not execute Rust or call a model in your browser.

## What you run locally

- **A Rust library and command-line engine** computes the exact reference over your stored vectors. It scans compressed int8 embeddings and calculates how wrong each estimate could be, then checks original vectors where needed.
- **A Python workbench** imports embeddings, compares saved search results, and reports missing IDs, ranking agreement, work and timing. Optional document text and relevance judgments make the audit easier to inspect.

Use it when changing compression, adjusting a search budget, or investigating a retrieval regression. FAISS already offers exact reference search; Cairn packages an inspectable search engine and loss-audit workflow. Its performance advantage depends on the workload.

**Status:** working engineering prototype. “Exact” refers to normalized stored float32 vectors and the [documented scoring contract](docs/DESIGN.md). Closest by vector score does not necessarily mean most useful to a person. [Linux/macOS verification](docs/VERIFICATION.md).

## How the engine avoids unnecessary full scores

Suppose we want the best three documents. Cheap estimates put A, B, C first. After checking their originals, C scores only **0.801**. D's estimate is **0.800**, but its error allowance means the full score could reach **0.830**. Cairn checks D: it scores **0.827** and replaces C. Documents whose maximum possible score is below the current cutoff can be skipped.

That is the synthetic, step-by-step [algorithm lesson](https://balkashynov.github.io/cairn/#how-it-works). It explains a different question from the recorded audit above: when must we check beyond the initial shortlist? If every score is ambiguous, Cairn checks every original. Fixed int8 search succeeds on our public benchmark datasets; the committed near-tie regression demonstrates where a fixed shortlist can fail.

## Run a small failure case

Requires Rust 1.93+ and Python 3.10+.

```bash
git clone https://github.com/balkashynov/cairn.git
cd cairn
cargo build --release --locked
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python examples/quickstart.py
```

This deliberately constructed example gives 100 vectors the same binary signs, so compressed search cannot distinguish them. The closest original is row 99 (a document ID), but the fixed shortlist leaves it out. Cairn’s `certified` mode checks the ambiguous candidates and finds it. No network, model weights, or API key is needed.

```text
binary:    nearest row 0,  approximate
certified: nearest row 99, exact
```

A second regression covers int8: 100 almost identical vectors quantize to the same codes, and a 4x candidate budget drops the closest one. Certified search detects the ambiguity and scores all 100. Worst-case fallback is intentional, not hidden.

## Search your embeddings

```bash
python -m pip install -e '.[workbench]'
cairn-workbench import-npy documents.npy documents.fbin
cairn-workbench import-npy queries.npy queries.fbin
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

## Recorded performance

On an Apple M3 Pro, Cairn matched all **2,124 / 2,124** benchmark top-10 rankings. At **999,999 vectors x 768 dimensions**, native median search was **23.3 ms** versus **51.2 ms** for single-query, single-thread FAISS, scoring **0.15%** of original vectors on average after scanning every compressed vector. Large-run tail latency was variable, and FAISS batching wins on the smaller datasets. Fixed int8 search with a 4x candidate budget also recovered all benchmark neighbors and was faster than certified search. [Full measurements and limits](docs/BENCHMARKS.md).

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
