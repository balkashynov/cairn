//! Symmetric per-vector scalar quantization, distinct from Cohere's API int8 format.
use anyhow::{Result, ensure};

pub const MAX_DIM: usize = 65_536;

/// Reserve substantially more than the standard gamma_n f64 accumulation bound.
/// Inputs to every bound are normalized and dimension is limited by MAX_DIM.
pub fn guard(dim: usize) -> f64 {
    64.0 * (dim as f64 + 1.0) * f64::EPSILON
}

pub fn normalize(v: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !v.is_empty() && v.len() <= MAX_DIM,
        "dimension must be in 1..={MAX_DIM}"
    );
    ensure!(
        v.iter().all(|x| x.is_finite()),
        "vectors must contain only finite values"
    );
    // f32 squares (including subnormals) and their sum fit safely in f64.
    let norm = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    ensure!(norm > 0.0, "zero vectors have no cosine direction");
    Ok(v.iter().map(|&x| (x as f64 / norm) as f32).collect())
}

#[derive(Clone, Debug)]
pub struct Quantized {
    pub codes: Vec<i8>,
    pub scale: f64,
    pub residual: f64,
    pub norm: f64,
    pub reconstructed_norm: f64,
}

impl Quantized {
    /// Accepts finite, normalized f32 values from normalize().
    pub fn new(v: &[f32]) -> Self {
        let g = guard(v.len());
        let scale = v.iter().map(|&x| (x as f64).abs()).fold(0.0, f64::max) / 127.0;
        let codes: Vec<i8> = v
            .iter()
            .map(|&x| ((x as f64 / scale).round().clamp(-127.0, 127.0)) as i8)
            .collect();
        let residual = v
            .iter()
            .zip(&codes)
            .map(|(&x, &c)| (x as f64 - c as f64 * scale).powi(2))
            .sum::<f64>()
            .sqrt()
            + g;
        let norm = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt() + g;
        let reconstructed_norm = codes
            .iter()
            .map(|&c| (c as f64 * scale).powi(2))
            .sum::<f64>()
            .sqrt()
            + g;
        Self {
            codes,
            scale,
            residual,
            norm,
            reconstructed_norm,
        }
    }
}

pub fn pack_signs(v: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0_u8; v.len().div_ceil(8)];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            bytes[i / 8] |= 1 << (7 - i % 8);
        }
    }
    bytes
}

pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let mut n = 0;
    let mut aa = a.chunks_exact(8);
    let mut bb = b.chunks_exact(8);
    for (x, y) in aa.by_ref().zip(bb.by_ref()) {
        n += (u64::from_ne_bytes(x.try_into().unwrap())
            ^ u64::from_ne_bytes(y.try_into().unwrap()))
        .count_ones();
    }
    n + aa
        .remainder()
        .iter()
        .zip(bb.remainder())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum::<u32>()
}
