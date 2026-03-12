use std::path::PathBuf;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use mass_spectrometry_counts::bucketing::FixedWidthBucketing;
use mass_spectrometry_counts::download;
use mass_spectrometry_counts::output::write_output;
use mass_spectrometry_counts::parsers;

#[derive(Parser, Debug)]
#[command(
    name = "mass-spectrometry-counts",
    about = "Compute bucket counts and co-occurrence matrices from mass spectra"
)]
struct Cli {
    /// Use a local HDF5 file instead of downloading GeMS-A10
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Bucket width in Da
    #[arg(long, default_value = "0.1")]
    bin_width: f64,

    /// Minimum m/z value
    #[arg(long, default_value = "0.0")]
    min_mz: f64,

    /// Maximum m/z value
    #[arg(long, default_value = "2000.0")]
    max_mz: f64,

    /// Output directory
    #[arg(short, long, default_value = "results")]
    output_dir: PathBuf,

    /// Number of threads for parallel processing
    #[arg(short, long, default_value = "4")]
    threads: usize,

    /// HDF5 read batch size
    #[arg(long, default_value = "10000")]
    batch_size: usize,

    /// Download cache directory
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    let input_path = match cli.input {
        Some(path) => path,
        None => {
            let cache_dir = cli.cache_dir.unwrap_or_else(download::default_cache_dir);
            download::download_dataset(&cache_dir)?
        }
    };

    let bucketing = FixedWidthBucketing::new(cli.min_mz, cli.max_mz, cli.bin_width)?;

    eprintln!(
        "Bucketing: [{}, {}) with bin_width={}, {} buckets",
        bucketing.min_mz(),
        bucketing.max_mz(),
        bucketing.bin_width(),
        bucketing.num_buckets()
    );
    eprintln!("Threads: {}", cli.threads);
    eprintln!("Input: {}", input_path.display());

    let parser = parsers::hdf5::Hdf5Parser::open(&input_path, cli.batch_size)?;
    let num_spectra_total = parser.num_spectra();

    let pb = ProgressBar::new(num_spectra_total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} spectra ({per_sec}, {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let (cooc, num_spectra) = mass_spectrometry_counts::process_spectra_streaming(
        parser,
        &bucketing,
        cli.threads,
        cli.batch_size,
        Some(&pb),
    )?;

    pb.finish_with_message(format!("{} spectra processed", num_spectra));

    eprintln!(
        "Co-occurrence matrix: {} non-zero entries",
        cooc.num_nonzero()
    );

    write_output(&cli.output_dir, &bucketing, &cooc, num_spectra)?;
    eprintln!("Output written to {}", cli.output_dir.display());

    Ok(())
}
