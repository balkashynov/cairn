# Measured results and reproduction

Measured on 2026-10-15 UTC, Apple M3 Pro, 18 GiB RAM, macOS 27.0.1, Rust 1.93.0 release build, NEON SDOT kernel, Python 3.12.11 and FAISS 1.15.1. All five datasets ran sequentially with no concurrent benchmark. These are measurements from one laptop, not a hardware-independent promise.

Certified search matched the **ordered** exhaustive top-10 on **2,124 / 2,124 queries**. The full numeric results and source provenance are committed; benchmark inputs and full text reports are regenerated locally.

## Single-query latency

All times are milliseconds. Native search excludes IPC. The Python roundtrip includes serialization, IPC and client overhead. FAISS is called in-process from Python, one thread, one query per call. A fair application-level comparison uses the roundtrip column; an embedded Rust application uses the native column.

| Dataset | Vectors x dims | Queries | Cairn native p50 / p95 | Cairn Python p50 | FAISS p50 / p95 | Mean full vectors scored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 5,183 x 1024 | 300 | 0.174 / 0.196 | 0.464 | 0.316 / 0.341 | 40.00 (0.772%) |
| Programmers | 32,176 x 1024 | 876 | 0.886 / 0.978 | 1.198 | 2.032 / 2.485 | 40.00 (0.124%) |
| FiQA | 57,638 x 1024 | 648 | 1.558 / 1.681 | 1.875 | 3.729 / 4.475 | 40.03 (0.069%) |
| Wikipedia 100k | 99,999 x 768 | 200 | 2.659 / 3.391 | 2.938 | 4.969 / 6.163 | 870.67 (0.871%) |
| Wikipedia 1m | 999,999 x 768 | 100 | 23.275 / 67.327 | 23.634 | 51.209 / 67.157 | 1,452.09 (0.145%) |

At one million vectors, Cairn's median was 2.20x faster than this single-query FAISS baseline, but p95 was essentially tied and Cairn p99 was 198.65 ms. The run had substantial tail variability. No cold-cache, memory-pressure or tail-latency robustness claim is supported. At SciFact size, Python overhead erases the native speed advantage. These limits should accompany any headline result.

## Throughput and approximate alternatives

| Dataset | FAISS all-query batch, ms/query | Binary 4x Recall@10 | Binary 4x queries with misses | Int8 4x Recall@10 | Exact relevance nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.023 | 0.9173 | 145 / 300 | 1.0000 | 0.7181 |
| Programmers | 0.060 | 0.9088 | 449 / 876 | 1.0000 | 0.4368 |
| FiQA | 0.110 | 0.9023 | 354 / 648 | 1.0000 | 0.4214 |
| Wikipedia 100k | 0.181 | 0.7820 | 157 / 200 | 1.0000 | not available |
| Wikipedia 1m | 52.065 | 0.8190 | 73 / 100 | 1.0000 | not available |

FAISS batching is dramatically faster on the four smaller datasets. The one-million all-query batch was unexpectedly slower per query than smaller batches; this is one observed run, not evidence that Cairn beats a tuned BLAS implementation for bulk ground-truth generation. Use a tuned batched exact baseline before choosing an offline evaluation engine.

Int8 4x happened to recover every neighbor in these datasets and was faster than certified search. That is a valid alternative when measured recall is sufficient. It has no general exactness guarantee: the committed near-tie regression makes its recall fail, and Cairn deliberately falls back to exhaustive scoring. Binary 4x is still much faster when its recall tradeoff is acceptable. Local sign-bit quantization here is not Cohere's native binary encoding.

Neighbor recall and relevance nDCG answer different questions. Certified and exhaustive nDCG agree because their rankings agree. An approximate neighbor miss can leave nDCG unchanged. SciFact's exact nDCG of 0.7181 also matches the embedding publisher's rounded result, a useful data-pipeline sanity check.

## Memory and disk

Each row below is a separate native CLI process measured with macOS `/usr/bin/time -l`; maximum resident set size is reported in bytes and converted to MiB below. No Python, FAISS, or builder process is included. Query files stream. There was no forced page-cache eviction; startup and mapping are included in the process lifetime. These are residency observations, not a memory cap or physical-I/O measurement.

| Dataset | Queries in process | Peak RSS, MiB | Index on disk, MiB |
| --- | ---: | ---: | ---: |
| Wikipedia 100k | 1 | 102.1 | 378.4 |
| Wikipedia 100k | 200 | 243.6 | 378.4 |
| Wikipedia 1m | 1 | 864.0 | 3784.2 |
| Wikipedia 1m | 100 | 1181.9 | 3784.2 |

Repeated queries increase residency as additional original-vector pages enter the process working set. The million-vector file is 3.70 GiB, larger than the original 2.86 GiB float matrix. Its quantized codes and metadata occupy about 763 MiB; certified scratch adds about 7.6 MiB. The low fraction of original vectors scored does not imply a proportional RSS reduction.

## Inspect the evidence

- [`summary.json`](../benchmarks/summary.json): complete aggregate timing, quality and index metadata.
- [`numeric/`](../benchmarks/numeric): gzip JSON for every query, with returned IDs/scores, misses, work counters and a recorded native timing. Aggregate quantiles use all timed repetitions; per-query objects retain one representative repetition, not every raw sample.
- [`checksums.json`](../benchmarks/checksums.json): SHA-256 for those numeric archives.
- [`provenance/`](../benchmarks/provenance): pinned sources and downloaded or extracted hashes.
- [`source-manifest.json`](../benchmarks/source-manifest.json): hashes of the implementation at measurement time. After measurement, the report presentation was refined and Rust index offsets were made private with an immutable accessor. The search algorithm and Python measurement code are unchanged. The final binary was checked against all 2,124 recorded exhaustive rankings again; see `final-verification.json`.
- [`requirements.txt`](../benchmarks/requirements.txt): installed benchmark dependency versions.
- [`demo/`](../demo): 12 selected SciFact examples; the displayed aggregates still cover all 300 queries.

## Setup

```bash
cargo build --release --locked
python -m pip install -e '.[benchmark,test]'
python scripts/prepare_beir.py scifact
python scripts/prepare_beir.py cqadupstack-programmers
python scripts/prepare_beir.py fiqa
python scripts/prepare_wikipedia.py
for dataset in scifact cqadupstack-programmers fiqa cohere-wiki-100k; do
  ./target/release/cairn build data/$dataset/corpus.fbin data/$dataset/index.cairn
done
scripts/benchmark_all.sh results/final
```

Use Python 3.12+ for the preparation scripts (`hashlib.file_digest`, safe tar extraction). The client itself supports Python 3.10+. Native builds refuse existing destinations; reuse existing immutable indexes or choose new paths.

The optional larger run downloads a roughly 3.1 GB compressed archive. It retains the portable corpus instead of an extra NumPy copy:

```bash
python scripts/prepare_wikipedia_large.py
./target/release/cairn build data/cohere-wiki-1m/corpus.fbin data/cohere-wiki-1m/index.cairn
cairn-workbench audit data/cohere-wiki-1m/index.cairn data/cohere-wiki-1m/queries.fbin \
  --repeats 3 --faiss --output results/final/cohere-wiki-1m
```

## Protocol

- Complete test queries for SciFact, programmers and FiQA. Historical Wikipedia tests use the first 200 queries for 100k and first 100 for 1m, selected before measurement.
- Top-10, three timed repetitions. A persistent native worker maps one index. Every method is warmed on three queries. Method order rotates across query and repetition.
- Native timings include query normalization, scratch allocation, code scan, heaps and exact scoring. Python roundtrip medians also include serialization, IPC and client overhead. Startup/index validation is not included in per-query native latency.
- Exhaustive Cairn is the correctness reference, not the only performance baseline. FAISS `IndexFlatIP` uses one CPU thread, the same stored normalized vectors and normalized queries, and one query per call. It runs in a separate timing pass. Its f32 reduction may differ on near ties; disagreement is measured.
- FAISS's all-query batch throughput is reported separately. It is typically much better than this single-query engine. Do not compare batch throughput to per-request latency as if they were the same workload.
- No cache eviction is forced. These are warm-cache laptop measurements, not cold storage, managed-service, cloud VM or concurrency benchmarks. Repetitions from one machine are not independent hardware samples.
- Neighbor Recall@10 measures the intersection with exact stored-vector neighbors. Relevance nDCG@10 uses the original qrels and is a separate quantity. A neighbor loss need not reduce relevance nDCG, and a change in nDCG is not a bound on ANN recall.
- Memory reports label native-process peak RSS and query count. Logical full-vector bytes are not physical storage reads or an RSS estimate. Total index disk size includes the original vectors.
- Synthetic equality tags test filter behavior only; they do not establish real tenant distributions or security controls.

[Source provenance and data licenses](DATA.md). The default importer and benchmarks use locally derived scalar int8 and float sign bits. They do not evaluate Cohere's API-native quantized output, and no live Embed 5 call is implied.

## Existing retriever integration

`examples/audit_faiss.py` builds a real FAISS `IndexHNSWFlat` over the identical stored vectors, with `M=16`, `efConstruction=80`, `efSearch=16`, one thread, and 40 returned candidates. On SciFact, the saved run had neighbor/candidate Recall@10 0.9430; 105 of 300 queries missed at least one exact neighbor from the candidate pool. This intentionally small search budget illustrates the workflow, not a tuned HNSW comparison. No source-engine latency claim is made. [Numeric integration evidence](../benchmarks/numeric/scifact-hnsw-integration.json.gz).
