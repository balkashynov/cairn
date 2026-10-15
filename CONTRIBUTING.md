# Contributing

Useful contributions include independently reproducing the bound checks, finding an exactness counterexample, measuring a new CPU architecture, or improving the saved-run audit workflow. Include input shapes, source/model provenance, compiler, hardware and a runnable reproducer. Synthetic results should be labeled as such.

Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo build --release`, and `python -m pytest -q`. Performance changes should compare the same queries and index against the previous revision. Keep ordinary single-query timing separate from batch throughput, startup, physical I/O and memory residency.

Do not add an approximate early exit to certified mode. Do not weaken strict tie handling or the immutable-file contract. An optimized kernel must match scalar integer arithmetic on extreme values and tails before it is used.
