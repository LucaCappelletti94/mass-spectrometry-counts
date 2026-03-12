use std::path::Path;

use hdf5::File as H5File;
use ndarray::Array3;

use super::Spectrum;

/// Parser for GeMS HDF5 format.
///
/// The `/spectrum` dataset has shape `(num_spectra, 2, 128)` float64:
/// - `spectrum[i, 0, :]` = m/z values (zero-padded to 128)
/// - `spectrum[i, 1, :]` = intensity values (zero-padded to 128)
pub struct Hdf5Parser {
    file: H5File,
    num_spectra: usize,
    peak_len: usize,
    batch_size: usize,
    current_offset: usize,
    current_batch: Vec<Spectrum>,
    batch_cursor: usize,
}

impl Hdf5Parser {
    pub fn open(
        path: &Path,
        batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = H5File::open(path)?;
        let dataset = file.dataset("spectrum")?;
        let shape = dataset.shape();
        if shape.len() != 3 || shape[1] != 2 {
            return Err(format!(
                "unexpected /spectrum shape: {:?}, expected (N, 2, 128)",
                shape
            )
            .into());
        }
        let num_spectra = shape[0];
        let peak_len = shape[2];

        Ok(Self {
            file,
            num_spectra,
            peak_len,
            batch_size,
            current_offset: 0,
            current_batch: Vec::new(),
            batch_cursor: 0,
        })
    }

    pub fn num_spectra(&self) -> usize {
        self.num_spectra
    }

    fn load_next_batch(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let remaining = self.num_spectra - self.current_offset;
        if remaining == 0 {
            self.current_batch.clear();
            self.batch_cursor = 0;
            return Ok(());
        }

        let count = remaining.min(self.batch_size);
        let dataset = self.file.dataset("spectrum")?;

        // Read slice [current_offset..current_offset+count, 0..2, 0..peak_len] -> Array3
        let data: Array3<f64> = dataset.read_slice(ndarray::s![
            self.current_offset..self.current_offset + count,
            ..,
            ..
        ])?;
        // data shape: (count, 2, peak_len)

        self.current_batch.clear();
        self.current_batch.reserve(count);

        let peak_len = self.peak_len;
        for i in 0..count {
            let mut peaks = Vec::new();
            for j in 0..peak_len {
                let mz = data[[i, 0, j]];
                let intensity = data[[i, 1, j]];
                if mz == 0.0 && intensity == 0.0 {
                    break;
                }
                peaks.push((mz, intensity));
            }
            peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            self.current_batch.push(Spectrum { peaks });
        }

        self.current_offset += count;
        self.batch_cursor = 0;
        Ok(())
    }
}

impl Iterator for Hdf5Parser {
    type Item = Result<Spectrum, Box<dyn std::error::Error + Send + Sync>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_cursor >= self.current_batch.len() {
            if self.current_offset >= self.num_spectra {
                return None;
            }
            if let Err(e) = self.load_next_batch() {
                return Some(Err(e));
            }
            if self.current_batch.is_empty() {
                return None;
            }
        }

        let spectrum = self.current_batch[self.batch_cursor].clone();
        self.batch_cursor += 1;
        Some(Ok(spectrum))
    }
}
