//! Immutable, versioned, little-endian index with atomic no-clobber publication.
use crate::quantize::{MAX_DIM, Quantized, normalize, pack_signs};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const MAGIC: &[u8; 8] = b"CAIRN001";
const HEADER: usize = 256;
const META: usize = 32;
const HASH_AT: usize = 96;

fn align(n: usize) -> Result<usize> {
    Ok(n.checked_add(63).context("index size overflow")? / 64 * 64)
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).context("index size overflow")
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("index size overflow")
}
fn get_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn get_f64(b: &[u8], at: usize) -> f64 {
    f64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn usize_at(b: &[u8], at: usize) -> Result<usize> {
    usize::try_from(get_u64(b, at)).context("index value exceeds address space")
}
fn put_u64(b: &mut [u8], at: usize, v: usize) {
    b[at..at + 8].copy_from_slice(&(v as u64).to_le_bytes());
}

#[derive(Clone, Debug, Serialize)]
pub struct Layout {
    pub rows: usize,
    pub dimensions: usize,
    pub meta_offset: usize,
    pub codes_offset: usize,
    pub binary_offset: usize,
    pub vectors_offset: usize,
    pub file_bytes: usize,
}
impl Layout {
    pub fn new(rows: usize, dimensions: usize) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= u32::MAX as usize,
            "row count must be in 1..=u32::MAX"
        );
        ensure!(
            dimensions > 0 && dimensions <= MAX_DIM,
            "unsupported dimension"
        );
        let meta_offset = HEADER;
        let codes_offset = align(add(meta_offset, mul(rows, META)?)?)?;
        let binary_offset = align(add(codes_offset, mul(rows, dimensions)?)?)?;
        let vectors_offset = align(add(binary_offset, mul(rows, dimensions.div_ceil(8))?)?)?;
        let file_bytes = add(vectors_offset, mul(mul(rows, dimensions)?, 4)?)?;
        Ok(Self {
            rows,
            dimensions,
            meta_offset,
            codes_offset,
            binary_offset,
            vectors_offset,
            file_bytes,
        })
    }
    fn header(&self) -> [u8; HEADER] {
        let mut b = [0; HEADER];
        b[..8].copy_from_slice(MAGIC);
        for (i, v) in [
            self.rows,
            self.dimensions,
            self.meta_offset,
            self.codes_offset,
            self.binary_offset,
            self.vectors_offset,
            self.file_bytes,
        ]
        .into_iter()
        .enumerate()
        {
            put_u64(&mut b, 8 + i * 8, v);
        }
        b
    }
}

#[derive(Serialize)]
pub struct IndexInfo {
    pub layout: Layout,
    pub format: &'static str,
    pub kernel: &'static str,
    pub quantized_scan_bytes: usize,
    pub float_vector_bytes: usize,
    pub payload_sha256: String,
}

/// An immutable mapping with validated, non-mutable section boundaries.
///
/// ```compile_fail
/// fn corrupt_offsets(index: &mut cairn::storage::Index) {
///     index.layout().vectors_offset = usize::MAX;
/// }
/// ```
pub struct Index {
    map: Mmap,
    pub(crate) layout: Layout,
}
impl Index {
    /// Validated offsets are immutable for the mapping lifetime. Exposing a
    /// mutable layout would let safe callers invalidate the bounds of raw views.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Open an immutable Cairn index. Structural checks do not hash the full payload.
    /// Call verify() for byte integrity and validate_bounds() for untrusted producers.
    ///
    /// # Safety
    /// The backing file must not be written to or truncated while this mapping lives.
    /// Build publishes a new file without replacing existing ones. Use new filenames
    /// for new snapshots; external writers must follow the same immutability contract.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        ensure!(
            cfg!(target_endian = "little"),
            "big-endian hosts are not supported"
        );
        let f = File::open(path.as_ref()).context("open index")?;
        ensure!(f.metadata()?.len() >= HEADER as u64, "truncated header");
        // SAFETY: caller guarantees backing-file immutability for the mapping lifetime.
        let map = unsafe { Mmap::map(&f)? };
        ensure!(
            &map[..8] == MAGIC,
            "invalid Cairn magic or unsupported format version"
        );
        let layout = Layout::new(usize_at(&map, 8)?, usize_at(&map, 16)?)?;
        ensure!(
            map.len() == layout.file_bytes,
            "index size mismatch (truncated or trailing bytes)"
        );
        for (at, v) in [
            (24, layout.meta_offset),
            (32, layout.codes_offset),
            (40, layout.binary_offset),
            (48, layout.vectors_offset),
            (56, layout.file_bytes),
        ] {
            ensure!(usize_at(&map, at)? == v, "invalid index section offset");
        }
        ensure!(
            map[64..96]
                .iter()
                .chain(map[128..HEADER].iter())
                .all(|&x| x == 0),
            "unsupported header flags"
        );
        let index = Self { map, layout };
        for i in 0..index.layout.rows {
            let (scale, residual, norm) = index.meta(i);
            ensure!(
                scale.is_finite()
                    && scale > 0.0
                    && scale <= 1.0
                    && residual.is_finite()
                    && (0.0..=2.0).contains(&residual)
                    && norm.is_finite()
                    && (0.99..=1.01).contains(&norm),
                "invalid quantization metadata at row {i}"
            );
        }
        Ok(index)
    }
    pub fn meta(&self, id: usize) -> (f64, f64, f64) {
        assert!(id < self.layout.rows);
        let at = self.layout.meta_offset + id * META;
        (
            get_f64(&self.map, at),
            get_f64(&self.map, at + 8),
            get_f64(&self.map, at + 16),
        )
    }
    pub fn tag(&self, id: usize) -> u64 {
        assert!(id < self.layout.rows);
        get_u64(&self.map, self.layout.meta_offset + id * META + 24)
    }
    pub fn codes(&self, id: usize) -> &[i8] {
        assert!(id < self.layout.rows);
        let at = self.layout.codes_offset + id * self.layout.dimensions;
        // SAFETY: i8 has alignment 1 and all bit patterns are valid; span checked by layout.
        unsafe {
            std::slice::from_raw_parts(self.map.as_ptr().add(at).cast(), self.layout.dimensions)
        }
    }
    pub fn binary(&self, id: usize) -> &[u8] {
        assert!(id < self.layout.rows);
        let stride = self.layout.dimensions.div_ceil(8);
        let at = self.layout.binary_offset + id * stride;
        &self.map[at..at + stride]
    }
    pub fn vector(&self, id: usize) -> &[f32] {
        assert!(id < self.layout.rows);
        let at = self.layout.vectors_offset + id * self.layout.dimensions * 4;
        // SAFETY: section starts at a 64-byte boundary, row stride is a multiple of 4,
        // all f32 bit patterns are valid, host is little endian, and layout bounds it.
        unsafe {
            std::slice::from_raw_parts(self.map.as_ptr().add(at).cast(), self.layout.dimensions)
        }
    }
    pub fn info(&self) -> IndexInfo {
        IndexInfo {
            layout: self.layout.clone(),
            format: "CAIRN001",
            kernel: crate::kernel::name(),
            quantized_scan_bytes: self.layout.rows * (self.layout.dimensions + META),
            float_vector_bytes: self.layout.rows * self.layout.dimensions * 4,
            payload_sha256: self.map[HASH_AT..HASH_AT + 32]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        }
    }
    pub fn verify(&self) -> Result<()> {
        let digest = Sha256::digest(&self.map[HEADER..]);
        ensure!(
            digest[..] == self.map[HASH_AT..HASH_AT + 32],
            "payload checksum mismatch"
        );
        Ok(())
    }
    /// Recompute quantization from every stored vector. This establishes the bound
    /// invariants even when the file was produced by a different implementation.
    pub fn validate_bounds(&self) -> Result<()> {
        for id in 0..self.layout.rows {
            let v = self.vector(id);
            ensure!(
                v.iter().all(|x| x.is_finite()),
                "non-finite vector at row {id}"
            );
            let quant = Quantized::new(v);
            let (scale, residual, norm) = self.meta(id);
            ensure!(
                scale == quant.scale && residual >= quant.residual && norm >= quant.norm,
                "invalid bounds at row {id}"
            );
            ensure!(
                self.codes(id) == quant.codes,
                "invalid int8 codes at row {id}"
            );
            ensure!(
                self.binary(id) == pack_signs(v),
                "invalid binary codes at row {id}"
            );
        }
        Ok(())
    }
}

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// fbin: little-endian u32 rows, u32 dimensions, then row-major f32 values.
/// Optional tags: one little-endian u64 per row. Tags are equality filters, not ACLs.
pub fn build(input: &Path, output: &Path, tags: Option<&Path>) -> Result<Layout> {
    ensure!(
        !output.exists(),
        "destination exists; build a new snapshot instead"
    );
    let mut source = BufReader::new(File::open(input)?);
    let mut hdr = [0; 8];
    source.read_exact(&mut hdr).context("read fbin header")?;
    let rows = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[4..].try_into().unwrap()) as usize;
    let layout = Layout::new(rows, dim)?;
    ensure!(
        source.get_ref().metadata()?.len() == (8 + rows * dim * 4) as u64,
        "fbin size mismatch"
    );
    let mut tag_reader = tags.map(File::open).transpose()?.map(BufReader::new);
    if let Some(t) = &tag_reader {
        ensure!(
            t.get_ref().metadata()?.len() == (rows * 8) as u64,
            "tag file size mismatch"
        );
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = output
        .file_name()
        .context("destination must be a file")?
        .to_string_lossy();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let tmp_path = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), stamp));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&tmp_path)?;
    let tmp = TempFile(tmp_path);
    file.set_len(layout.file_bytes as u64)?;
    file.write_all(&layout.header())?;
    let mut bytes = vec![0; dim * 4];
    for id in 0..rows {
        source.read_exact(&mut bytes)?;
        let raw: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let v = normalize(&raw).with_context(|| format!("invalid input vector at row {id}"))?;
        let q = Quantized::new(&v);
        let mut meta = [0_u8; META];
        for (i, value) in [q.scale, q.residual, q.norm].into_iter().enumerate() {
            meta[i * 8..i * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        if let Some(t) = &mut tag_reader {
            t.read_exact(&mut meta[24..32])?;
        }
        file.seek(SeekFrom::Start((layout.meta_offset + id * META) as u64))?;
        file.write_all(&meta)?;
        file.seek(SeekFrom::Start((layout.codes_offset + id * dim) as u64))?;
        // i8 -> u8 preserves the stored two's-complement byte pattern.
        let codes: Vec<u8> = q.codes.iter().map(|&c| c as u8).collect();
        file.write_all(&codes)?;
        file.seek(SeekFrom::Start(
            (layout.binary_offset + id * dim.div_ceil(8)) as u64,
        ))?;
        file.write_all(&pack_signs(&v))?;
        for (b, x) in bytes.chunks_exact_mut(4).zip(v) {
            b.copy_from_slice(&x.to_le_bytes());
        }
        file.seek(SeekFrom::Start(
            (layout.vectors_offset + id * dim * 4) as u64,
        ))?;
        file.write_all(&bytes)?;
    }
    file.seek(SeekFrom::Start(HEADER as u64))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    file.seek(SeekFrom::Start(HASH_AT as u64))?;
    file.write_all(&hash.finalize())?;
    file.sync_all()?;
    // A hard link publishes atomically and refuses an existing destination. Both
    // names are in the same directory/filesystem. Drop removes only the temp name.
    fs::hard_link(&tmp.0, output).context("atomic no-clobber publication")?;
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(layout)
}

/// Streaming reader for query files; corpus-sized query sets need not fit in RAM.
pub struct FbinReader {
    file: BufReader<File>,
    remaining: usize,
    dimensions: usize,
    bytes: Vec<u8>,
}
impl FbinReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let mut h = [0; 8];
        file.read_exact(&mut h)?;
        let n = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(h[4..].try_into().unwrap()) as usize;
        if d == 0 || d > MAX_DIM {
            bail!("invalid query dimension");
        }
        ensure!(
            file.get_ref().metadata()?.len() == add(8, mul(mul(n, d)?, 4)?)? as u64,
            "query fbin size mismatch"
        );
        Ok(Self {
            file,
            remaining: n,
            dimensions: d,
            bytes: vec![0; d * 4],
        })
    }
}
impl Iterator for FbinReader {
    type Item = Result<Vec<f32>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        if let Err(e) = self.file.read_exact(&mut self.bytes) {
            self.remaining = 0;
            return Some(Err(e.into()));
        }
        let mut row = Vec::with_capacity(self.dimensions);
        for b in self.bytes.chunks_exact(4) {
            row.push(f32::from_le_bytes(b.try_into().unwrap()));
        }
        Some(Ok(row))
    }
}

pub fn read_fbin(path: &Path) -> Result<Vec<Vec<f32>>> {
    FbinReader::open(path)?.collect()
}
