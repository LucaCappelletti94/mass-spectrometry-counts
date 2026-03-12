pub mod hdf5;

/// A single spectrum: a list of (m/z, intensity) pairs sorted by m/z.
#[derive(Debug, Clone)]
pub struct Spectrum {
    pub peaks: Vec<(f64, f64)>,
}

/// Trait for iterating over spectra from any source.
pub trait SpectrumSource:
    Iterator<Item = Result<Spectrum, Box<dyn std::error::Error + Send + Sync>>>
{
}

impl<T> SpectrumSource for T where
    T: Iterator<Item = Result<Spectrum, Box<dyn std::error::Error + Send + Sync>>>
{
}
