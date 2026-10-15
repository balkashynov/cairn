# Verification record

## Native engine and Python client

The initial public revision `71d879eb9e8d54ce77d148944c9911abb3a210dc` passed [CI on Ubuntu and macOS](https://github.com/balkashynov/cairn/actions/runs/38690668358): Rust 1.93.0 formatting, clippy with warnings denied, 10 unit/integration tests, one compile-fail API safety check, release build, Python lint/format checks, nine Python tests and the runnable no-API example. Later documentation and evidence-packaging changes retain the same search code. Current commit checks are visible in the repository Actions tab.

Local validation also ran release-mode Rust tests and the real-data audit on an Apple M3 Pro. The final binary was rechecked after index-layout API hardening: ordered IDs and exact scores match all 2,124 saved exhaustive results. [Final binary/source hashes and counts](../benchmarks/final-verification.json).

The tests cover integer SIMD/scalar agreement including extreme signed lanes and tails; randomized exact rankings; residual bounds on anisotropic and nearly tied inputs; deterministic ties; filtered and empty eligible sets; k larger than the corpus; zero/NaN/infinite rejection; extreme finite magnitudes; malformed headers and checksum/bound corruption; no-clobber publication; concurrent immutable readers; independent NumPy scoring; JSON worker errors and closure; Cohere request batching and response validation through a mock; report injection escaping; Unicode JSONL separators; and external-candidate loss versus ranking loss.

The exact-recovery guarantee is conditional on valid, immutable indexes and the documented arithmetic contract. Tests are evidence, not a formal proof. Live Cohere API requests, Windows, big-endian hosts, cold-cache latency, concurrency throughput, forced memory pressure and production service integration were not validated.

## Report interaction

The generated demo was opened in the Codex in-app browser. The headline correctly shows all 300 SciFact queries while the explorer labels its 12 selected examples. Switching Binary 4x to Certified changes a recorded miss into complete top-k recovery. The next-miss button advances among failures; selecting Certified reports no missed neighbors. Console warnings/errors were empty. The narrow viewport keeps the table horizontally scrollable and stacks result columns. This is functional browser inspection, not a full accessibility audit.

## External-engine integration

The real FAISS HNSW example ran over SciFact with the parameters in its script. The audit reports 105/300 queries with an exact neighbor absent from the 40-candidate pool. This demonstrates using Cairn alongside an existing engine. The small search budget is deliberate and does not represent tuned HNSW performance.

## Evidence boundaries

All benchmark text/vectors come from attributed public datasets. No private corporate workload, live Embed 5 result, external user endorsement or Cohere acceptance is claimed. Public numeric results exclude corpus and query text; the compact SciFact replay retains source IDs and attribution. No outreach is part of this verification record.
