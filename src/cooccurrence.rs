use rustc_hash::FxHashMap;

use crate::bucketing::FixedWidthBucketing;
use crate::counts::spectrum_bucket_indices;
use crate::parsers::Spectrum;

/// Encodes a pair of bucket indices (i, j) where i <= j into a single u64 key.
#[inline]
fn pair_key(i: u32, j: u32) -> u64 {
    (i as u64) << 32 | j as u64
}

/// Decodes a u64 key back into a pair of bucket indices.
#[inline]
pub fn decode_pair_key(key: u64) -> (u32, u32) {
    ((key >> 32) as u32, key as u32)
}

/// Sparse co-occurrence matrix stored as upper-triangle entries in a hash map.
///
/// For each pair (i, j) with i <= j, stores how many spectra have peaks in both buckets.
/// The diagonal entry (i, i) equals the per-bucket count.
#[derive(Debug, Clone)]
pub struct CooccurrenceMatrix {
    entries: FxHashMap<u64, u64>,
}

impl CooccurrenceMatrix {
    pub fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Process a single spectrum, incrementing co-occurrence for all bucket pairs.
    pub fn add_spectrum(&mut self, spectrum: &Spectrum, bucketing: &FixedWidthBucketing) {
        let indices = spectrum_bucket_indices(spectrum, bucketing);
        let n = indices.len();
        for a in 0..n {
            for b in a..n {
                let key = pair_key(indices[a], indices[b]);
                *self.entries.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// Merge another CooccurrenceMatrix into this one (for parallel reduction).
    pub fn merge(&mut self, other: &CooccurrenceMatrix) {
        for (&key, &count) in &other.entries {
            *self.entries.entry(key).or_insert(0) += count;
        }
    }

    /// Get the co-occurrence count for a pair of buckets.
    pub fn get(&self, i: u32, j: u32) -> u64 {
        let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
        self.entries.get(&pair_key(lo, hi)).copied().unwrap_or(0)
    }

    /// Number of non-zero entries.
    pub fn num_nonzero(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over all non-zero entries as (bucket_i, bucket_j, count).
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, u64)> + '_ {
        self.entries.iter().map(|(&key, &count)| {
            let (i, j) = decode_pair_key(key);
            (i, j, count)
        })
    }

    /// Iterate over sorted entries (by bucket_i, then bucket_j).
    pub fn iter_sorted(&self) -> Vec<(u32, u32, u64)> {
        let mut entries: Vec<_> = self.iter().collect();
        entries.sort_by_key(|&(i, j, _)| (i, j));
        entries
    }

    /// Extract per-bucket counts from the diagonal entries.
    pub fn diagonal_counts(&self, num_buckets: usize) -> Vec<u64> {
        let mut counts = vec![0u64; num_buckets];
        for (&key, &count) in &self.entries {
            let (i, j) = decode_pair_key(key);
            if i == j {
                counts[i as usize] = count;
            }
        }
        counts
    }
}

impl Default for CooccurrenceMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooccurrence_basic() {
        let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
        let mut cooc = CooccurrenceMatrix::new();

        // Spectrum with peaks in buckets 0, 1, 5
        let s1 = Spectrum {
            peaks: vec![(5.0, 1.0), (15.0, 1.0), (55.0, 1.0)],
        };
        cooc.add_spectrum(&s1, &bucketing);

        // Diagonal = bucket counts
        assert_eq!(cooc.get(0, 0), 1);
        assert_eq!(cooc.get(1, 1), 1);
        assert_eq!(cooc.get(5, 5), 1);

        // Off-diagonal
        assert_eq!(cooc.get(0, 1), 1);
        assert_eq!(cooc.get(0, 5), 1);
        assert_eq!(cooc.get(1, 5), 1);

        // Symmetry: get(j, i) == get(i, j)
        assert_eq!(cooc.get(1, 0), 1);
        assert_eq!(cooc.get(5, 0), 1);

        // Unused pair
        assert_eq!(cooc.get(2, 3), 0);
    }

    #[test]
    fn test_diagonal_equals_bucket_counts() {
        let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
        let mut cooc = CooccurrenceMatrix::new();
        let mut counts = crate::counts::BucketCounts::new(bucketing.num_buckets());

        let spectra = vec![
            Spectrum {
                peaks: vec![(5.0, 1.0), (15.0, 1.0), (55.0, 1.0)],
            },
            Spectrum {
                peaks: vec![(5.0, 1.0), (7.0, 1.0), (25.0, 1.0)],
            },
            Spectrum {
                peaks: vec![(55.0, 1.0), (65.0, 1.0)],
            },
        ];

        for s in &spectra {
            cooc.add_spectrum(s, &bucketing);
            counts.add_spectrum(s, &bucketing);
        }

        // The diagonal of the co-occurrence matrix must equal the bucket counts
        let diag = cooc.diagonal_counts(bucketing.num_buckets());
        assert_eq!(diag, counts.counts());
    }

    #[test]
    fn test_merge() {
        let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
        let mut a = CooccurrenceMatrix::new();
        let mut b = CooccurrenceMatrix::new();

        a.add_spectrum(
            &Spectrum {
                peaks: vec![(5.0, 1.0), (15.0, 1.0)],
            },
            &bucketing,
        );
        b.add_spectrum(
            &Spectrum {
                peaks: vec![(5.0, 1.0), (55.0, 1.0)],
            },
            &bucketing,
        );

        a.merge(&b);
        assert_eq!(a.get(0, 0), 2); // bucket 0 seen in both spectra
        assert_eq!(a.get(0, 1), 1); // only from first spectrum
        assert_eq!(a.get(0, 5), 1); // only from second spectrum
    }
}
