# Exact recovery from quantized scans

## Scoring contract

The builder normalizes each finite, nonzero input vector using a float64 norm, then rounds each coordinate to float32. Query normalization follows the same rule. The reference score is a left-to-right float64 sum of products of these stored float32 coordinates. Larger scores rank first; lower row ID breaks ties. This is close to cosine on the input vectors, but the precise contract is the stored representation. Different float32 BLAS reductions may disagree on near ties.

Only dimensions 1 through 65,536 and row counts up to `u32::MAX` are supported. Big-endian hosts are rejected. Zero, NaN and infinite inputs fail explicitly.

## Quantization and bound

For a stored vector `x`, choose `s_x = max(abs(x)) / 127` and integer codes `c_x = round(x / s_x)`, clamped to `[-127,127]`. The reconstructed vector is `x_hat = s_x c_x`. Store its scale, an upper estimate of the residual norm `e_x = ||x - x_hat||`, and an upper estimate of `||x||`. Quantize a query the same way.

The approximate score is an exact integer dot product followed by two float64 scale multiplications:

```text
a = dot(c_q, c_x) * s_q * s_x
```

The difference from the full score decomposes as:

```text
q.x - q_hat.x_hat = (q - q_hat).x + q_hat.(x - x_hat)
```

Triangle and Cauchy-Schwarz give the deterministic real-arithmetic bound:

```text
|q.x - a| <= e_q * ||x|| + ||q_hat|| * e_x
```

There is no distribution assumption, learned threshold, sample recall promise, or independence assumption. This simple bound is deliberately conservative.

For each query:

1. Scan all eligible int8 rows, retaining an upper bound per row and the best `max(4k,32)` approximate scores as seeds.
2. Score the seeds with the reference computation. This supplies a valid current kth score.
3. Visit remaining rows in file order. Skip a row only if its upper bound is **strictly below** the current kth score. Otherwise read and score its original vector.
4. Return the exact heap in deterministic order.

Every skipped row is below a cutoff that can only increase. Strict comparison keeps exact ties eligible. If there are fewer than k eligible rows, they all survive. The seed count affects performance, never correctness.

## Floating-point treatment

The theorem above uses real arithmetic; the implementation uses IEEE floating-point arithmetic. With normalized inputs and the dimension cap, the standard accumulation error bound `gamma_n = n*u/(1-n*u)` stays small (`u = 2^-53`). A stored float32 product is representable in float64 before summation. The int8 accumulator is exact: even `65536 * 128 * 128` fits signed int32.

The implementation adds `g(d) = 64 * (d + 1) * f64::EPSILON` to each computed residual norm and vector norm, and again to the final score upper bound. At the maximum dimension this is approximately `9.31e-10`. This exceeds the usual dot/norm accumulation allowance for normalized vectors and covers reconstruction, subtraction, squaring, square-root and scale-product rounding conservatively. Rust is compiled without fast-math reassociation. The code is not a machine-checked floating-point proof or an interval-arithmetic library.

Tests compare every sampled bound to actual float64 scores on isotropic, anisotropic, near-identical and extreme finite inputs; SIMD outputs are checked against integer scalar arithmetic. Exhaustive top-k tests cover odd dimensions, ties, filters, k larger than the corpus, and independent NumPy calculations. Future compiler/architecture ports must retain these checks and the stated arithmetic contract.

`certified: true` reports that this algorithm completed under the valid-index contract. The result also includes the largest upper bound of any pruned row and the final cutoff, useful for audit. These are diagnostic witnesses, not a cryptographic or independently complete proof of the scan.

## SIMD

ARM uses runtime-detected SDOT, with a widening NEON fallback. SDOT is emitted through a small inline-assembly block because its Rust intrinsic is unstable on the minimum supported compiler. Loads cover only complete 32-byte blocks; the scalar tail handles the remainder. The widening path uses 16-bit products and 32-bit accumulators.

AVX2 sign-extends int8 lanes to int16 and uses `madd_epi16`. It intentionally avoids `maddubs_epi16`, whose saturating intermediate can silently corrupt extreme signed products. All supported paths compute the same integer sum. ARM execution is locally verified; x86 execution is covered by CI when available.

## Persistence and trust

`CAIRN001` has a 256-byte header, then aligned sections for 32-byte row metadata, int8 codes, packed sign bits, and normalized float32 originals. Header integers and floats are little endian; section starts align to 64 bytes. Row metadata contains three float64 values (scale, residual, norm) and a uint64 equality tag. Packed signs use the most significant bit first, and unused trailing bits are zero.

Open recomputes the only legal layout, rejects unsupported flags and checks lengths, offsets and finite metadata. It does not touch all original vectors. `verify` checks a SHA-256 digest of the payload. `verify --deep` additionally recomputes the codes, norms, residuals and sign bits for every row, establishing the mathematical invariants of externally produced files. A checksum alone does not validate an adversarial producer's math.

Build uses a private `create_new` temporary file, writes and syncs the complete index, then creates a hard link to the final name. The hard link publishes atomically without replacing an existing file; the parent directory is synced on Unix. Failure before publication removes only the temporary file created by that build. Filesystems must support same-directory hard links. Updates create a new snapshot with a new name.

Memory mapping requires that nobody write or truncate the backing file while it is open. The Rust constructor is explicitly unsafe and documents this lifetime contract. The CLI and Python layer enforce no-clobber publication for their own writes, but cannot prevent an unrelated process from violating it.

## Resource accounting

The quantized scan touches approximately `N * (d + 32)` bytes. Certified scratch includes `8N` bytes for bounds plus query and heap storage. The original-vector section uses `4Nd` disk bytes and is accessed through the OS page cache. Packed binary baselines add `N * ceil(d/8)` bytes. A 1024-dimensional index therefore occupies about 5.16 bytes per original coordinate, rather than reducing total disk size.

`logical_float_bytes` counts row bytes requested by exact scoring, not physical disk traffic. Page faults can fetch more than a row; repeated queries can keep much of the original section resident. The tool does not impose a hard memory budget. Benchmark memory measurements must name process scope, query count and platform units.

The `scratch_bytes` diagnostic is an accounting estimate for principal arrays and heaps, not a peak allocator or process measurement. Temporary sorting capacity, allocation bookkeeping and the Python client are outside that figure.
