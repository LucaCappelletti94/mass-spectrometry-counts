use mass_spectrometry_counts::bucketing::FixedWidthBucketing;
use mass_spectrometry_counts::cooccurrence::CooccurrenceMatrix;
use mass_spectrometry_counts::counts::BucketCounts;
use mass_spectrometry_counts::output;
use mass_spectrometry_counts::parsers::Spectrum;

/// Build a small set of synthetic spectra with known bucket assignments.
fn synthetic_spectra() -> Vec<Spectrum> {
    vec![
        // Spectrum 1: peaks in buckets 0, 1, 5 (at bin_width=10, min=0)
        Spectrum {
            peaks: vec![(5.0, 1.0), (15.0, 0.8), (55.0, 0.3)],
        },
        // Spectrum 2: peaks in buckets 0, 2
        Spectrum {
            peaks: vec![(3.0, 0.5), (7.0, 0.9), (25.0, 0.4)],
        },
        // Spectrum 3: peaks in buckets 5, 6
        Spectrum {
            peaks: vec![(55.0, 1.0), (65.0, 0.7)],
        },
        // Spectrum 4: single peak in bucket 9
        Spectrum {
            peaks: vec![(95.0, 1.0)],
        },
    ]
}

#[test]
fn test_end_to_end_counts_and_cooccurrence() {
    let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
    let spectra = synthetic_spectra();

    let mut counts = BucketCounts::new(bucketing.num_buckets());
    let mut cooc = CooccurrenceMatrix::new();

    for s in &spectra {
        counts.add_spectrum(s, &bucketing);
        cooc.add_spectrum(s, &bucketing);
    }

    // Verify bucket counts:
    // Bucket 0: spectra 1, 2 -> 2
    // Bucket 1: spectrum 1 -> 1
    // Bucket 2: spectrum 2 -> 1
    // Bucket 5: spectra 1, 3 -> 2
    // Bucket 6: spectrum 3 -> 1
    // Bucket 9: spectrum 4 -> 1
    assert_eq!(counts.counts()[0], 2);
    assert_eq!(counts.counts()[1], 1);
    assert_eq!(counts.counts()[2], 1);
    assert_eq!(counts.counts()[5], 2);
    assert_eq!(counts.counts()[6], 1);
    assert_eq!(counts.counts()[9], 1);
    // Unused buckets
    assert_eq!(counts.counts()[3], 0);
    assert_eq!(counts.counts()[4], 0);
    assert_eq!(counts.counts()[7], 0);
    assert_eq!(counts.counts()[8], 0);

    // Property: diagonal of co-occurrence == bucket counts
    let diag = cooc.diagonal_counts(bucketing.num_buckets());
    assert_eq!(diag.as_slice(), counts.counts());

    // Verify specific co-occurrences:
    // (0, 1): only spectrum 1 has both -> 1
    assert_eq!(cooc.get(0, 1), 1);
    // (0, 2): only spectrum 2 has both -> 1
    assert_eq!(cooc.get(0, 2), 1);
    // (0, 5): spectrum 1 has both -> 1
    assert_eq!(cooc.get(0, 5), 1);
    // (1, 5): spectrum 1 has both -> 1
    assert_eq!(cooc.get(1, 5), 1);
    // (5, 6): spectrum 3 has both -> 1
    assert_eq!(cooc.get(5, 6), 1);
    // (1, 2): no spectrum has both -> 0
    assert_eq!(cooc.get(1, 2), 0);
    // (0, 9): no spectrum has both -> 0
    assert_eq!(cooc.get(0, 9), 0);

    // Property: co-occurrence is symmetric
    assert_eq!(cooc.get(0, 1), cooc.get(1, 0));
    assert_eq!(cooc.get(0, 5), cooc.get(5, 0));
    assert_eq!(cooc.get(5, 6), cooc.get(6, 5));
}

#[test]
fn test_parallel_equals_sequential() {
    let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
    let spectra = synthetic_spectra();

    // Sequential
    let mut seq_cooc = CooccurrenceMatrix::new();
    let mut seq_counts = BucketCounts::new(bucketing.num_buckets());
    for s in &spectra {
        seq_cooc.add_spectrum(s, &bucketing);
        seq_counts.add_spectrum(s, &bucketing);
    }

    // Parallel
    let (par_cooc, par_counts, num) =
        mass_spectrometry_counts::process_spectra_parallel(spectra.clone(), &bucketing, 2);

    assert_eq!(num, 4);
    assert_eq!(par_counts.counts(), seq_counts.counts());

    // Check all co-occurrence entries match
    for entry in seq_cooc.iter() {
        let (i, j, count) = entry;
        assert_eq!(par_cooc.get(i, j), count, "mismatch at ({}, {})", i, j);
    }
    assert_eq!(par_cooc.num_nonzero(), seq_cooc.num_nonzero());
}

#[test]
fn test_streaming_processing() {
    let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
    let spectra = synthetic_spectra();
    let source = spectra.into_iter().map(Ok);

    let (cooc, num_spectra) =
        mass_spectrometry_counts::process_spectra_streaming(source, &bucketing, 2, 2, None)
            .unwrap();

    assert_eq!(num_spectra, 4);
    assert_eq!(cooc.get(0, 0), 2); // bucket 0 in 2 spectra
    assert_eq!(cooc.get(5, 5), 2); // bucket 5 in 2 spectra
    assert_eq!(cooc.get(9, 9), 1); // bucket 9 in 1 spectrum
}

#[test]
fn test_output_npz_format() {
    let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
    let mut cooc = CooccurrenceMatrix::new();
    for s in &synthetic_spectra() {
        cooc.add_spectrum(s, &bucketing);
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.npz");
    output::write_npz(&path, &cooc, bucketing.num_buckets()).unwrap();

    // Verify we can read it back
    let mut npz = npyz::npz::NpzArchive::open(&path).unwrap();
    let csr = npyz::sparse::Csr::<i64>::from_npz(&mut npz).unwrap();
    assert_eq!(csr.shape, [10, 10]);
    assert_eq!(csr.data.len(), cooc.num_nonzero());
    // All upper triangle: column index >= row index for each entry
    for r in 0..10usize {
        let start = csr.indptr[r];
        let end = csr.indptr[r + 1];
        for pos in start..end {
            assert!(csr.indices[pos] >= r as u64);
        }
    }
}

#[test]
fn test_output_to_directory() {
    let dir = tempfile::tempdir().unwrap();
    let bucketing = FixedWidthBucketing::new(0.0, 100.0, 10.0).unwrap();
    let mut cooc = CooccurrenceMatrix::new();
    for s in &synthetic_spectra() {
        cooc.add_spectrum(s, &bucketing);
    }

    output::write_output(dir.path(), &bucketing, &cooc, 4).unwrap();

    // Check files exist
    assert!(dir.path().join("cooccurrence.npz").exists());
    assert!(dir.path().join("metadata.json").exists());

    // Parse metadata
    let meta_str = std::fs::read_to_string(dir.path().join("metadata.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    assert_eq!(meta["num_buckets"], 10);
    assert_eq!(meta["num_spectra"], 4);
    assert_eq!(meta["bin_width"], 10.0);
    assert!(meta["num_nonzero_entries"].as_u64().unwrap() > 0);
}
