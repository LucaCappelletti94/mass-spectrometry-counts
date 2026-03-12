use crate::bucketing::FixedWidthBucketing;
use crate::parsers::Spectrum;

/// Accumulates per-bucket counts: how many spectra have at least one peak in each bucket.
#[derive(Debug, Clone)]
pub struct BucketCounts {
    counts: Vec<u64>,
}

impl BucketCounts {
    pub fn new(num_buckets: usize) -> Self {
        Self {
            counts: vec![0; num_buckets],
        }
    }

    /// Process a single spectrum, incrementing the count for each bucket that has at least one peak.
    pub fn add_spectrum(&mut self, spectrum: &Spectrum, bucketing: &FixedWidthBucketing) {
        let mut last_bucket: Option<usize> = None;
        for &(mz, _) in &spectrum.peaks {
            if let Some(idx) = bucketing.bucket_index(mz) {
                // Since peaks are sorted by m/z, bucket indices are non-decreasing.
                // Skip duplicates.
                if last_bucket == Some(idx) {
                    continue;
                }
                self.counts[idx] += 1;
                last_bucket = Some(idx);
            }
        }
    }

    /// Merge another BucketCounts into this one (for parallel reduction).
    pub fn merge(&mut self, other: &BucketCounts) {
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    pub fn counts(&self) -> &[u64] {
        &self.counts
    }
}

/// Extract deduplicated bucket indices from a spectrum.
/// Since peaks are sorted by m/z, bucket indices are non-decreasing,
/// so we just skip consecutive duplicates.
pub fn spectrum_bucket_indices(spectrum: &Spectrum, bucketing: &FixedWidthBucketing) -> Vec<u32> {
    let mut indices = Vec::new();
    let mut last: Option<u32> = None;
    for &(mz, _) in &spectrum.peaks {
        if let Some(idx) = bucketing.bucket_index(mz) {
            let idx = idx as u32;
            if last == Some(idx) {
                continue;
            }
            indices.push(idx);
            last = Some(idx);
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_counts() {
        let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
        let mut counts = BucketCounts::new(bucketing.num_buckets());

        // Spectrum with peaks in buckets 0, 1, 5
        let s1 = Spectrum {
            peaks: vec![(5.0, 1.0), (15.0, 1.0), (55.0, 1.0)],
        };
        counts.add_spectrum(&s1, &bucketing);
        assert_eq!(counts.counts()[0], 1);
        assert_eq!(counts.counts()[1], 1);
        assert_eq!(counts.counts()[5], 1);
        assert_eq!(counts.counts()[2], 0);

        // Spectrum with two peaks in the same bucket (should count once)
        let s2 = Spectrum {
            peaks: vec![(5.0, 1.0), (7.0, 1.0)],
        };
        counts.add_spectrum(&s2, &bucketing);
        assert_eq!(counts.counts()[0], 2);
    }

    #[test]
    fn test_spectrum_bucket_indices_dedup() {
        let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
        let spectrum = Spectrum {
            peaks: vec![(5.0, 1.0), (7.0, 1.0), (15.0, 1.0), (55.0, 1.0)],
        };
        let indices = spectrum_bucket_indices(&spectrum, &bucketing);
        // Two peaks in bucket 0, but should only appear once
        assert_eq!(indices, vec![0, 1, 5]);
    }

    #[test]
    fn test_merge() {
        let mut a = BucketCounts::new(5);
        a.counts[0] = 3;
        a.counts[2] = 7;

        let mut b = BucketCounts::new(5);
        b.counts[0] = 2;
        b.counts[3] = 4;

        a.merge(&b);
        assert_eq!(a.counts(), &[5, 0, 7, 4, 0]);
    }
}
