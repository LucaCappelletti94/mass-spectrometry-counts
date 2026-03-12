use std::io::{self, Write};
use std::path::Path;

use npyz::WriterBuilder;
use serde::Serialize;

use crate::bucketing::FixedWidthBucketing;
use crate::cooccurrence::CooccurrenceMatrix;

#[derive(Debug, Serialize)]
pub struct Metadata {
    pub min_mz: f64,
    pub max_mz: f64,
    pub bin_width: f64,
    pub num_buckets: usize,
    pub num_spectra: u64,
    pub num_nonzero_entries: usize,
}

/// Zip file options with ZIP64 enabled for large matrices.
fn zip_options() -> npyz::zip::write::FileOptions {
    npyz::zip::write::FileOptions::default().large_file(true)
}

/// Write the co-occurrence matrix as a scipy-compatible sparse CSR NPZ file.
///
/// The matrix is `num_buckets x num_buckets`, upper triangle only (i <= j).
/// Diagonal entries (i == i) are per-bucket counts.
///
/// Uses ZIP64 extensions to support matrices exceeding the 4 GB ZIP32 limit.
/// Loadable in Python with `scipy.sparse.load_npz("cooccurrence.npz")`.
pub fn write_npz(
    path: &Path,
    cooc: &CooccurrenceMatrix,
    num_buckets: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let sorted = cooc.iter_sorted();
    let nnz = sorted.len();

    // Build CSR arrays from sorted (row, col, value) triples
    let mut indptr = vec![0i64; num_buckets + 1];
    let mut indices: Vec<i32> = Vec::with_capacity(nnz);
    let mut data: Vec<i64> = Vec::with_capacity(nnz);

    for (i, j, count) in &sorted {
        indices.push(*j as i32);
        data.push(*count as i64);
        indptr[*i as usize + 1] += 1;
    }

    // Convert counts to cumulative offsets
    for r in 1..=num_buckets {
        indptr[r] += indptr[r - 1];
    }

    let mut npz = npyz::npz::NpzWriter::create(path)?;
    let opts = zip_options();

    // format
    npz.array::<[u8]>("format", opts)?
        .dtype(npyz::DType::Plain("|S3".parse().unwrap()))
        .shape(&[])
        .begin_nd()?
        .push(b"csr")?;

    // shape
    npz.array::<i64>("shape", opts)?
        .default_dtype()
        .shape(&[2])
        .begin_nd()?
        .extend([num_buckets as i64, num_buckets as i64])?;

    // indices (column indices, i32 since num_buckets <= 200_000)
    npz.array::<i32>("indices", opts)?
        .default_dtype()
        .shape(&[nnz as u64])
        .begin_nd()?
        .extend(indices.iter().copied())?;

    // indptr (row pointers)
    npz.array::<i64>("indptr", opts)?
        .default_dtype()
        .shape(&[indptr.len() as u64])
        .begin_nd()?
        .extend(indptr.iter().copied())?;

    // data (co-occurrence counts)
    npz.array::<i64>("data", opts)?
        .default_dtype()
        .shape(&[nnz as u64])
        .begin_nd()?
        .extend(data.iter().copied())?;

    // Explicitly finalize the zip archive
    npz.zip_writer().finish().map_err(io::Error::from)?;

    Ok(())
}

/// Write metadata as JSON.
pub fn write_metadata<W: Write>(
    writer: &mut W,
    bucketing: &FixedWidthBucketing,
    num_spectra: u64,
    num_nonzero_entries: usize,
) -> std::io::Result<()> {
    let metadata = Metadata {
        min_mz: bucketing.min_mz(),
        max_mz: bucketing.max_mz(),
        bin_width: bucketing.bin_width(),
        num_buckets: bucketing.num_buckets(),
        num_spectra,
        num_nonzero_entries,
    };
    let json = serde_json::to_string_pretty(&metadata).map_err(std::io::Error::other)?;
    writer.write_all(json.as_bytes())?;
    writer.write_all(b"\n")?;
    Ok(())
}

/// Write all output files to a directory.
pub fn write_output(
    output_dir: &Path,
    bucketing: &FixedWidthBucketing,
    cooc: &CooccurrenceMatrix,
    num_spectra: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    std::fs::create_dir_all(output_dir)?;

    let npz_path = output_dir.join("cooccurrence.npz");
    write_npz(&npz_path, cooc, bucketing.num_buckets())?;

    let meta_path = output_dir.join("metadata.json");
    let mut meta_file = std::io::BufWriter::new(std::fs::File::create(&meta_path)?);
    write_metadata(&mut meta_file, bucketing, num_spectra, cooc.num_nonzero())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::Spectrum;

    #[test]
    fn test_write_npz_roundtrip() {
        let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
        let mut cooc = CooccurrenceMatrix::new();
        cooc.add_spectrum(
            &Spectrum {
                peaks: vec![(5.0, 1.0), (15.0, 1.0)],
            },
            &bucketing,
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.npz");
        write_npz(&path, &cooc, bucketing.num_buckets()).unwrap();

        // Verify the file exists and is non-empty
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(file_size > 0);

        // Verify we can read it back with npyz
        let mut npz = npyz::npz::NpzArchive::open(&path).unwrap();
        let csr = npyz::sparse::Csr::<i64>::from_npz(&mut npz).unwrap();
        assert_eq!(csr.shape, [10, 10]);
        assert_eq!(csr.data.len(), 3); // (0,0), (0,1), (1,1)
        assert!(csr.data.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_write_metadata() {
        let bucketing = FixedWidthBucketing::new(0.0, 2000.0, 0.1).unwrap();
        let mut buf = Vec::new();
        write_metadata(&mut buf, &bucketing, 1000, 500).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(json["num_buckets"], 20000);
        assert_eq!(json["num_spectra"], 1000);
        assert_eq!(json["num_nonzero_entries"], 500);
    }
}
