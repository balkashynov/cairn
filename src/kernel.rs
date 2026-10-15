//! Architecture-specific integer dot products. Every kernel computes the same i32.
//! Input dimensions are capped by the storage layer to prevent integer overflow.

pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// The dimension must be at most 65,536; codes are in [-127, 127].
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    assert!(a.len() <= 65_536);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: feature detection above; both slices have identical length.
            return unsafe { neon_dotprod(a, b) };
        }
        // SAFETY: NEON is baseline on aarch64; loads stay within the slices.
        unsafe { neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above; loads stay within both slices.
            return unsafe { avx2(a, b) };
        }
        dot_i8_scalar(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    dot_i8_scalar(a, b)
}

pub fn name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            "aarch64-neon-sdot"
        } else {
            "aarch64-neon-widening"
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            "x86_64-avx2"
        } else {
            "scalar"
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        "scalar"
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn neon_dotprod(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let mut s0 = vdupq_n_s32(0);
    let mut s1 = vdupq_n_s32(0);
    let mut i = 0;
    while i + 32 <= a.len() {
        // SAFETY: the loop bounds cover both 16-byte loads in each slice.
        unsafe {
            let x0 = vld1q_s8(a.as_ptr().add(i));
            let y0 = vld1q_s8(b.as_ptr().add(i));
            let x1 = vld1q_s8(a.as_ptr().add(i + 16));
            let y1 = vld1q_s8(b.as_ptr().add(i + 16));
            // The stdarch dotprod intrinsic is unstable on the minimum Rust version.
            // This instruction has no memory effects and only updates its accumulator.
            std::arch::asm!("sdot {s:v}.4s, {x:v}.16b, {y:v}.16b", s = inout(vreg) s0,
                x = in(vreg) x0, y = in(vreg) y0, options(pure, nomem, nostack));
            std::arch::asm!("sdot {s:v}.4s, {x:v}.16b, {y:v}.16b", s = inout(vreg) s1,
                x = in(vreg) x1, y = in(vreg) y1, options(pure, nomem, nostack));
        }
        i += 32;
    }
    vaddvq_s32(vaddq_s32(s0, s1)) + dot_i8_scalar(&a[i..], &b[i..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let mut s = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= a.len() {
        // SAFETY: i + 16 <= len for both slices.
        let (x, y) = unsafe { (vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i))) };
        s = vaddq_s32(s, vpaddlq_s16(vmull_s8(vget_low_s8(x), vget_low_s8(y))));
        s = vaddq_s32(s, vpaddlq_s16(vmull_high_s8(x, y)));
        i += 16;
    }
    vaddvq_s32(s) + dot_i8_scalar(&a[i..], &b[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let mut s = _mm256_setzero_si256();
    let mut i = 0;
    while i + 16 <= a.len() {
        // Widen first: maddubs saturates on signed extreme pairs and is unsafe here.
        // SAFETY: the loop bounds cover the 16-byte loads.
        let (x, y) = unsafe {
            (
                _mm_loadu_si128(a.as_ptr().add(i).cast()),
                _mm_loadu_si128(b.as_ptr().add(i).cast()),
            )
        };
        s = _mm256_add_epi32(
            s,
            _mm256_madd_epi16(_mm256_cvtepi8_epi16(x), _mm256_cvtepi8_epi16(y)),
        );
        i += 16;
    }
    let mut lanes = [0_i32; 8];
    // SAFETY: lanes has exactly 32 writable bytes; unaligned store is permitted.
    unsafe {
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), s);
    }
    lanes.into_iter().sum::<i32>() + dot_i8_scalar(&a[i..], &b[i..])
}

/// Fixed-order f64 accumulation of f32 products is Cairn's reference score.
pub fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0, |s, (&x, &y)| s + x as f64 * y as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispatched_matches_scalar_including_tails_and_extremes() {
        for n in [1, 7, 15, 16, 17, 31, 32, 33, 256, 768, 1024, 2048, 65536] {
            let a: Vec<i8> = (0..n)
                .map(|i| ((i * 71) % 255) as i16 - 127)
                .map(|v| v as i8)
                .collect();
            let b: Vec<i8> = (0..n)
                .map(|i| ((i * 29 + 17) % 255) as i16 - 127)
                .map(|v| v as i8)
                .collect();
            assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b));
            let extreme = vec![-128; n];
            assert_eq!(
                dot_i8(&extreme, &extreme),
                dot_i8_scalar(&extreme, &extreme)
            );
            #[cfg(target_arch = "aarch64")]
            assert_eq!(unsafe { neon(&a, &b) }, dot_i8_scalar(&a, &b));
        }
    }
}
