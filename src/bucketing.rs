/// Fixed-width m/z bucketing with uniform bin width.
///
/// Bucket `i` covers `[min_mz + i * bin_width, min_mz + (i+1) * bin_width)`.
#[derive(Debug, Clone)]
pub struct FixedWidthBucketing {
    min_mz: f64,
    bin_width: f64,
    num_buckets: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum BucketingError {
    #[error("bin_width must be positive and finite, got {0}")]
    InvalidBinWidth(f64),
    #[error("min_mz ({min_mz}) must be less than max_mz ({max_mz})")]
    InvalidRange { min_mz: f64, max_mz: f64 },
    #[error("min_mz and max_mz must be finite")]
    NonFinite,
}

impl FixedWidthBucketing {
    /// Create a new bucketing scheme.
    ///
    /// # Errors
    /// Returns an error if `bin_width <= 0`, `min_mz >= max_mz`, or any value is non-finite.
    pub fn new(min_mz: f64, max_mz: f64, bin_width: f64) -> Result<Self, BucketingError> {
        if !bin_width.is_finite() || bin_width <= 0.0 {
            return Err(BucketingError::InvalidBinWidth(bin_width));
        }
        if !min_mz.is_finite() || !max_mz.is_finite() {
            return Err(BucketingError::NonFinite);
        }
        if min_mz >= max_mz {
            return Err(BucketingError::InvalidRange { min_mz, max_mz });
        }
        let num_buckets = ((max_mz - min_mz) / bin_width).ceil() as usize;
        Ok(Self {
            min_mz,
            bin_width,
            num_buckets,
        })
    }

    /// Returns the bucket index for a given m/z value, or `None` if out of range.
    #[inline]
    pub fn bucket_index(&self, mz: f64) -> Option<usize> {
        if mz < self.min_mz {
            return None;
        }
        let idx = ((mz - self.min_mz) / self.bin_width) as usize;
        if idx >= self.num_buckets {
            None
        } else {
            Some(idx)
        }
    }

    /// Total number of buckets.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Returns the center m/z value for a given bucket index.
    ///
    /// # Panics
    /// Panics if `index >= self.num_buckets()`.
    #[inline]
    pub fn bucket_center_mz(&self, index: usize) -> f64 {
        assert!(index < self.num_buckets, "bucket index out of range");
        self.min_mz + (index as f64 + 0.5) * self.bin_width
    }

    /// Returns the (inclusive_min, exclusive_max) m/z range for a bucket.
    ///
    /// # Panics
    /// Panics if `index >= self.num_buckets()`.
    #[inline]
    pub fn bucket_range(&self, index: usize) -> (f64, f64) {
        assert!(index < self.num_buckets, "bucket index out of range");
        let lo = self.min_mz + index as f64 * self.bin_width;
        let hi = lo + self.bin_width;
        (lo, hi)
    }

    pub fn bin_width(&self) -> f64 {
        self.bin_width
    }

    pub fn min_mz(&self) -> f64 {
        self.min_mz
    }

    pub fn max_mz(&self) -> f64 {
        self.min_mz + self.num_buckets as f64 * self.bin_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_bucketing() {
        let b = FixedWidthBucketing::new(0.0, 100.0, 1.0).unwrap();
        assert_eq!(b.num_buckets(), 100);
        assert_eq!(b.bucket_index(0.0), Some(0));
        assert_eq!(b.bucket_index(0.5), Some(0));
        assert_eq!(b.bucket_index(1.0), Some(1));
        assert_eq!(b.bucket_index(99.9), Some(99));
        assert_eq!(b.bucket_index(100.0), None);
        assert_eq!(b.bucket_index(-0.1), None);
    }

    #[test]
    fn test_bucket_center() {
        let b = FixedWidthBucketing::new(0.0, 10.0, 1.0).unwrap();
        assert!((b.bucket_center_mz(0) - 0.5).abs() < 1e-10);
        assert!((b.bucket_center_mz(5) - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_bucket_range() {
        let b = FixedWidthBucketing::new(0.0, 10.0, 0.5).unwrap();
        assert_eq!(b.num_buckets(), 20);
        let (lo, hi) = b.bucket_range(3);
        assert!((lo - 1.5).abs() < 1e-10);
        assert!((hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fractional_buckets() {
        // max_mz not evenly divisible by bin_width -> ceiling
        let b = FixedWidthBucketing::new(0.0, 2000.0, 0.1).unwrap();
        assert_eq!(b.num_buckets(), 20000);
    }

    #[test]
    fn test_invalid_bin_width() {
        assert!(FixedWidthBucketing::new(0.0, 100.0, 0.0).is_err());
        assert!(FixedWidthBucketing::new(0.0, 100.0, -1.0).is_err());
        assert!(FixedWidthBucketing::new(0.0, 100.0, f64::NAN).is_err());
        assert!(FixedWidthBucketing::new(0.0, 100.0, f64::INFINITY).is_err());
    }

    #[test]
    fn test_invalid_range() {
        assert!(FixedWidthBucketing::new(100.0, 0.0, 1.0).is_err());
        assert!(FixedWidthBucketing::new(100.0, 100.0, 1.0).is_err());
        assert!(FixedWidthBucketing::new(f64::NAN, 100.0, 1.0).is_err());
    }
}
