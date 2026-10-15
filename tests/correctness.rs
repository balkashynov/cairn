use cairn::{
    Index, Mode, SearchOptions, build, kernel,
    quantize::{Quantized, guard, normalize},
};
use std::{fs, io::Write, path::Path};
use tempfile::TempDir;

fn fbin(path: &Path, rows: &[Vec<f32>]) {
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&(rows.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(rows[0].len() as u32).to_le_bytes()).unwrap();
    for r in rows {
        for x in r {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}
fn fixture(rows: &[Vec<f32>], tags: Option<Vec<u64>>) -> (TempDir, Index) {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("input.fbin");
    let output = dir.path().join("index.cairn");
    fbin(&input, rows);
    let tp = tags.map(|tags| {
        let p = dir.path().join("tags.u64");
        fs::write(
            &p,
            tags.into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        p
    });
    build(&input, &output, tp.as_deref()).unwrap();
    // SAFETY: test owns this immutable file; writes only after dropping this mapping.
    let index = unsafe { Index::open(&output) }.unwrap();
    (dir, index)
}
fn random(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
}

#[test]
fn certified_matches_reference_with_tails_filters_k_and_seed_variations() {
    let mut rng = 722;
    for d in [1, 3, 8, 17, 31, 128, 257, 768, 2048] {
        let rows: Vec<Vec<f32>> = (0..131)
            .map(|_| (0..d).map(|_| random(&mut rng)).collect())
            .collect();
        let (_dir, index) = fixture(&rows, Some((0..rows.len()).map(|i| i as u64 % 3).collect()));
        index.verify().unwrap();
        index.validate_bounds().unwrap();
        for _ in 0..4 {
            let q: Vec<_> = (0..d).map(|_| random(&mut rng)).collect();
            for k in [1, 10, 150] {
                for tag in [None, Some(0), Some(2), Some(999)] {
                    let opts = SearchOptions {
                        k,
                        mode: Mode::Exhaustive,
                        oversample: 1,
                        tag,
                    };
                    let expected = index.search(&q, &opts).unwrap();
                    let actual = index
                        .search(
                            &q,
                            &SearchOptions {
                                mode: Mode::Certified,
                                ..opts
                            },
                        )
                        .unwrap();
                    assert_eq!(actual.hits, expected.hits, "d={d} k={k} tag={tag:?}");
                    assert_eq!(
                        actual.stats.eligible,
                        actual.stats.pruned + actual.stats.exact_scored
                    );
                    if let Some(upper) = actual.stats.max_pruned_upper_bound {
                        assert!(upper < actual.stats.cutoff.unwrap());
                    }
                    for hit in &actual.hits {
                        if let Some(tag) = tag {
                            assert_eq!(index.tag(hit.id), tag);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn residual_bound_contains_actual_scores_on_anisotropic_and_near_tied_data() {
    let mut rng = 4;
    for d in [3, 31, 128, 768, 2048, 65536] {
        for trial in 0..20 {
            let mut x: Vec<_> = (0..d).map(|_| random(&mut rng)).collect();
            let mut q: Vec<_> = (0..d).map(|_| random(&mut rng)).collect();
            if trial % 3 == 0 {
                x[0] = 10000.0;
                q[0] = -10000.0;
            }
            if trial % 3 == 1 {
                q = x.iter().map(|v| v + random(&mut rng) * 1e-5).collect();
            }
            let x = normalize(&x).unwrap();
            let q = normalize(&q).unwrap();
            let cx = Quantized::new(&x);
            let cq = Quantized::new(&q);
            let approx = kernel::dot_i8(&cx.codes, &cq.codes) as f64 * cx.scale * cq.scale;
            let radius = cq.residual * cx.norm + cq.reconstructed_norm * cx.residual + guard(d);
            let actual = kernel::dot_f64(&q, &x);
            assert!(
                (actual - approx).abs() <= radius,
                "d={d} actual={actual} approx={approx} radius={radius}"
            );
        }
    }
}

#[test]
fn binary_candidate_failure_is_recovered_by_certified_search() {
    // All signs are identical. The true nearest vector has the largest ID and is
    // excluded from a fixed binary candidate set, even though Hamming search is exact.
    let mut rows = vec![vec![0.001, 1.0]; 100];
    rows[99] = vec![1.0, 0.001];
    let (_dir, index) = fixture(&rows, None);
    let q = vec![1.0, 0.001];
    let binary = index
        .search(
            &q,
            &SearchOptions {
                k: 1,
                mode: Mode::Binary,
                oversample: 4,
                tag: None,
            },
        )
        .unwrap();
    assert_ne!(binary.hits[0].id, 99);
    assert!(!binary.stats.certified);
    let certified = index
        .search(
            &q,
            &SearchOptions {
                k: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(certified.hits[0].id, 99);
    assert!(certified.stats.certified);
}

#[test]
fn exact_ties_and_ambiguous_bounds_fall_back_without_losing_results() {
    let rows = vec![vec![1.0, 0.3, -0.2]; 200];
    let (_dir, index) = fixture(&rows, None);
    let result = index
        .search(
            &rows[0],
            &SearchOptions {
                k: 5,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        result.hits.iter().map(|h| h.id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(result.stats.exact_scored, rows.len());
}

#[test]
fn malformed_inputs_fail_without_publishing_partial_index() {
    for bad in [
        vec![0.0, 0.0],
        vec![f32::NAN, 1.0],
        vec![f32::INFINITY, 1.0],
        vec![f32::NEG_INFINITY, 1.0],
    ] {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in.fbin");
        let out = dir.path().join("out.cairn");
        fbin(&input, &[vec![1.0, 0.0], bad]);
        assert!(build(&input, &out, None).is_err());
        assert!(!out.exists());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}

#[test]
fn extreme_finite_inputs_normalize_and_invalid_queries_fail() {
    let rows = vec![vec![f32::MAX, f32::MIN], vec![f32::from_bits(1), 0.0]];
    let (_dir, index) = fixture(&rows, None);
    index.validate_bounds().unwrap();
    for bad in [vec![], vec![1.0], vec![0.0, 0.0], vec![f32::NAN, 0.0]] {
        assert!(index.search(&bad, &Default::default()).is_err());
    }
    assert!(
        index
            .search(
                &[1.0, 0.0],
                &SearchOptions {
                    k: 0,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(
        index
            .search(
                &[1.0, 0.0],
                &SearchOptions {
                    oversample: 0,
                    ..Default::default()
                }
            )
            .is_err()
    );
}

#[test]
fn no_clobber_and_corruption_checks() {
    let rows = vec![vec![1.0, 0.1], vec![-0.2, 1.0]];
    let (dir, index) = fixture(&rows, None);
    let p = dir.path().join("index.cairn");
    let bytes = fs::read(&p).unwrap();
    assert!(build(&dir.path().join("input.fbin"), &p, None).is_err());
    assert_eq!(fs::read(&p).unwrap(), bytes);
    let codes = index.layout().codes_offset;
    drop(index);
    for at in [0, 8, 16, 24, 32, 40, 48, 56, 64, 128] {
        let mut corrupt = bytes.clone();
        corrupt[at] ^= 0xff;
        fs::write(&p, &corrupt).unwrap();
        assert!(unsafe { Index::open(&p) }.is_err(), "offset {at}");
    }
    let mut corrupt = bytes.clone();
    corrupt[codes] ^= 1;
    fs::write(&p, &corrupt).unwrap();
    let idx = unsafe { Index::open(&p) }.unwrap();
    assert!(idx.verify().is_err());
    assert!(idx.validate_bounds().is_err());
    drop(idx);
    for len in [0, 7, 255, bytes.len() - 1] {
        fs::write(&p, &bytes[..len]).unwrap();
        assert!(unsafe { Index::open(&p) }.is_err());
    }
}

#[test]
fn index_is_shareable_for_parallel_readers() {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<Index>();
    let (_dir, index) = fixture(&vec![vec![0.4, 0.5, -0.3]; 80], None);
    std::thread::scope(|s| {
        for _ in 0..4 {
            let index = &index;
            s.spawn(move || {
                assert_eq!(
                    index
                        .search(&[0.4, 0.5, -0.3], &Default::default())
                        .unwrap()
                        .hits[0]
                        .id,
                    0
                );
            });
        }
    });
}

#[test]
fn int8_oversampling_cannot_guarantee_exactness_for_near_ties() {
    // All rows round to the same [127,0] codes. The row-dependent scale makes
    // the true nearest row score slightly worse in quantized space.
    let mut rows = vec![vec![1.0, 0.001]; 100];
    rows[99] = vec![1.0, 0.003];
    let (_dir, index) = fixture(&rows, None);
    let q = rows[99].clone();
    let fixed = index
        .search(
            &q,
            &SearchOptions {
                k: 1,
                mode: Mode::Int8,
                oversample: 4,
                tag: None,
            },
        )
        .unwrap();
    assert_ne!(fixed.hits[0].id, 99);
    let exact = index
        .search(
            &q,
            &SearchOptions {
                k: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(exact.hits[0].id, 99);
    assert_eq!(exact.stats.exact_scored, 100);
}
