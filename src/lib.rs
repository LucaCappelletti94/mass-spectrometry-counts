pub mod bucketing;
pub mod cooccurrence;
pub mod counts;
pub mod download;
pub mod output;
pub mod parsers;

use bucketing::FixedWidthBucketing;
use cooccurrence::CooccurrenceMatrix;
use counts::BucketCounts;
use parsers::Spectrum;

/// Process spectra in parallel using rayon, accumulating co-occurrence and bucket counts.
///
/// Collects all spectra into memory first, then processes in parallel chunks.
/// For very large datasets, consider streaming with per-thread accumulation.
pub fn process_spectra_parallel(
    spectra: Vec<Spectrum>,
    bucketing: &FixedWidthBucketing,
    num_threads: usize,
) -> (CooccurrenceMatrix, BucketCounts, u64) {
    use rayon::prelude::*;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to create rayon thread pool");

    let num_spectra = spectra.len() as u64;

    pool.install(|| {
        let (cooc, counts) = spectra
            .par_iter()
            .fold(
                || {
                    (
                        CooccurrenceMatrix::new(),
                        BucketCounts::new(bucketing.num_buckets()),
                    )
                },
                |(mut cooc, mut counts), spectrum| {
                    cooc.add_spectrum(spectrum, bucketing);
                    counts.add_spectrum(spectrum, bucketing);
                    (cooc, counts)
                },
            )
            .reduce(
                || {
                    (
                        CooccurrenceMatrix::new(),
                        BucketCounts::new(bucketing.num_buckets()),
                    )
                },
                |(mut cooc_a, mut counts_a), (cooc_b, counts_b)| {
                    cooc_a.merge(&cooc_b);
                    counts_a.merge(&counts_b);
                    (cooc_a, counts_a)
                },
            );

        (cooc, counts, num_spectra)
    })
}

/// Process spectra from an iterator in streaming fashion with parallel processing.
///
/// Collects spectra in batches and processes each batch in parallel.
pub fn process_spectra_streaming(
    source: impl Iterator<Item = Result<Spectrum, Box<dyn std::error::Error + Send + Sync>>>,
    bucketing: &FixedWidthBucketing,
    num_threads: usize,
    batch_size: usize,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<(CooccurrenceMatrix, u64), Box<dyn std::error::Error + Send + Sync>> {
    use rayon::prelude::*;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?;

    let mut global_cooc = CooccurrenceMatrix::new();
    let mut num_spectra: u64 = 0;
    let mut batch = Vec::with_capacity(batch_size);

    for result in source {
        let spectrum = result?;
        batch.push(spectrum);

        if batch.len() >= batch_size {
            let batch_cooc = pool.install(|| {
                batch
                    .par_iter()
                    .fold(CooccurrenceMatrix::new, |mut cooc, spectrum| {
                        cooc.add_spectrum(spectrum, bucketing);
                        cooc
                    })
                    .reduce(CooccurrenceMatrix::new, |mut a, b| {
                        a.merge(&b);
                        a
                    })
            });

            num_spectra += batch.len() as u64;
            global_cooc.merge(&batch_cooc);
            if let Some(pb) = progress {
                pb.set_position(num_spectra);
            }
            batch.clear();
        }
    }

    // Process remaining spectra
    if !batch.is_empty() {
        let batch_cooc = pool.install(|| {
            batch
                .par_iter()
                .fold(CooccurrenceMatrix::new, |mut cooc, spectrum| {
                    cooc.add_spectrum(spectrum, bucketing);
                    cooc
                })
                .reduce(CooccurrenceMatrix::new, |mut a, b| {
                    a.merge(&b);
                    a
                })
        });

        num_spectra += batch.len() as u64;
        global_cooc.merge(&batch_cooc);
        if let Some(pb) = progress {
            pb.set_position(num_spectra);
        }
    }

    Ok((global_cooc, num_spectra))
}
