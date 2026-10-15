use crate::{
    kernel::{dot_f64, dot_i8},
    quantize::{Quantized, guard, hamming, normalize, pack_signs},
    storage::Index,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub id: usize,
    pub score: f64,
}
impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.total_cmp(&other.score).is_eq()
    }
}
impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct TopK {
    heap: BinaryHeap<Reverse<Hit>>,
    k: usize,
}
impl TopK {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k),
            k,
        }
    }
    fn add(&mut self, id: usize, score: f64) {
        let hit = Hit { id, score };
        if self.heap.len() < self.k {
            self.heap.push(Reverse(hit));
        } else if hit > self.heap.peek().unwrap().0 {
            *self.heap.peek_mut().unwrap() = Reverse(hit);
        }
    }
    fn cutoff(&self) -> f64 {
        if self.heap.len() < self.k {
            f64::NEG_INFINITY
        } else {
            self.heap.peek().unwrap().0.score
        }
    }
    fn sorted(self) -> Vec<Hit> {
        let mut hits: Vec<_> = self.heap.into_iter().map(|h| h.0).collect();
        hits.sort_unstable_by(|a, b| b.cmp(a));
        hits
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Certified,
    Exhaustive,
    Binary,
    Int8,
}
impl std::str::FromStr for Mode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "certified" => Ok(Self::Certified),
            "exhaustive" => Ok(Self::Exhaustive),
            "binary" => Ok(Self::Binary),
            "int8" => Ok(Self::Int8),
            _ => anyhow::bail!("mode must be certified, exhaustive, binary or int8"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchOptions {
    pub k: usize,
    pub mode: Mode,
    pub oversample: usize,
    pub tag: Option<u64>,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 10,
            mode: Mode::Certified,
            oversample: 4,
            tag: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub mode: Mode,
    pub eligible: usize,
    pub int8_scored: usize,
    pub binary_scored: usize,
    pub exact_scored: usize,
    pub pruned: usize,
    pub scratch_bytes: usize,
    pub logical_code_bytes: usize,
    pub logical_float_bytes: usize,
    pub elapsed_ms: f64,
    pub certified: bool,
    pub cutoff: Option<f64>,
    /// Largest upper bound among skipped rows; not an independently checkable proof.
    pub max_pruned_upper_bound: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub hits: Vec<Hit>,
    pub stats: Stats,
}

impl Index {
    /// Exact means identical top-k and tie ordering to Exhaustive for the stored
    /// normalized f32 vectors and a normalized f32 query, scored in fixed-order f64.
    /// Certified mode assumes the index's quantization invariants have not changed.
    pub fn search(&self, raw_query: &[f32], options: &SearchOptions) -> Result<SearchResult> {
        ensure!(
            raw_query.len() == self.layout.dimensions,
            "query dimension mismatch"
        );
        ensure!(options.k > 0, "k must be positive");
        ensure!(options.oversample > 0, "oversample must be positive");
        let start = Instant::now();
        let q = normalize(raw_query)?;
        let k = options.k.min(self.layout.rows);
        let mut exact = TopK::new(k);
        let mut stats = Stats {
            mode: options.mode,
            eligible: 0,
            int8_scored: 0,
            binary_scored: 0,
            exact_scored: 0,
            pruned: 0,
            scratch_bytes: q.len() * 4,
            logical_code_bytes: 0,
            logical_float_bytes: 0,
            elapsed_ms: 0.0,
            certified: matches!(options.mode, Mode::Certified | Mode::Exhaustive),
            cutoff: None,
            max_pruned_upper_bound: None,
        };
        let eligible = |id| options.tag.is_none_or(|tag| self.tag(id) == tag);
        if options.mode == Mode::Exhaustive {
            for id in 0..self.layout.rows {
                if !eligible(id) {
                    continue;
                }
                stats.eligible += 1;
                exact.add(id, self.exact_score(&q, id)?);
                stats.exact_scored += 1;
            }
        } else {
            let quant = Quantized::new(&q);
            let bits = pack_signs(&q);
            let candidates = if options.mode == Mode::Certified {
                k.saturating_mul(4).max(32)
            } else {
                k.saturating_mul(options.oversample)
            }
            .min(self.layout.rows);
            let mut seeds = TopK::new(candidates);
            let mut upper = if options.mode == Mode::Certified {
                vec![f64::NEG_INFINITY; self.layout.rows]
            } else {
                vec![]
            };
            stats.scratch_bytes += upper.len() * 8
                + quant.codes.len()
                + bits.len()
                + candidates * std::mem::size_of::<Hit>();
            // Approximate modes do not allocate the O(rows) bound array.
            #[allow(clippy::needless_range_loop)]
            for id in 0..self.layout.rows {
                if !eligible(id) {
                    continue;
                }
                stats.eligible += 1;
                let score = if options.mode == Mode::Binary {
                    stats.binary_scored += 1;
                    -(hamming(&bits, self.binary(id)) as f64)
                } else {
                    let (scale, residual, norm) = self.meta(id);
                    let score = dot_i8(&quant.codes, self.codes(id)) as f64 * quant.scale * scale;
                    stats.int8_scored += 1;
                    if options.mode == Mode::Certified {
                        // q.x - qhat.xhat = (q-qhat).x + qhat.(x-xhat).
                        // Triangle + Cauchy-Schwarz bound both terms. Strict pruning
                        // preserves exact ties; guard includes final score accumulation.
                        upper[id] = score
                            + quant.residual * norm
                            + quant.reconstructed_norm * residual
                            + guard(q.len());
                    }
                    score
                };
                seeds.add(id, score);
            }
            let mut ids: Vec<_> = seeds.sorted().into_iter().map(|h| h.id).collect();
            // Ascending IDs improve locality for disk-backed original vectors.
            ids.sort_unstable();
            for &id in &ids {
                exact.add(id, self.exact_score(&q, id)?);
                stats.exact_scored += 1;
            }
            if options.mode == Mode::Certified {
                for (id, &bound) in upper.iter().enumerate() {
                    if bound == f64::NEG_INFINITY || ids.binary_search(&id).is_ok() {
                        continue;
                    }
                    if bound < exact.cutoff() {
                        stats.pruned += 1;
                        stats.max_pruned_upper_bound =
                            Some(stats.max_pruned_upper_bound.map_or(bound, |b| b.max(bound)));
                    } else {
                        exact.add(id, self.exact_score(&q, id)?);
                        stats.exact_scored += 1;
                    }
                }
            } else {
                stats.pruned = stats.eligible - stats.exact_scored;
            }
        }
        stats.logical_code_bytes = stats.int8_scored * self.layout.dimensions
            + stats.binary_scored * self.layout.dimensions.div_ceil(8);
        stats.logical_float_bytes = stats.exact_scored * self.layout.dimensions * 4;
        stats.scratch_bytes += k * std::mem::size_of::<Hit>();
        let hits = exact.sorted();
        stats.cutoff = hits.last().map(|h| h.score);
        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(SearchResult { hits, stats })
    }
    fn exact_score(&self, q: &[f32], id: usize) -> Result<f64> {
        let score = dot_f64(q, self.vector(id));
        ensure!(
            score.is_finite(),
            "non-finite stored vector at row {id}; run verify"
        );
        Ok(score)
    }
}
