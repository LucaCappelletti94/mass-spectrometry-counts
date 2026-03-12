use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};

const GEMS_A10_URL: &str =
    "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/GeMS_A/GeMS_A10.hdf5";
const GEMS_A10_FILENAME: &str = "GeMS_A10.hdf5";

/// Returns the default cache directory: `~/.cache/mass-spectrometry-counts/`
pub fn default_cache_dir() -> PathBuf {
    let cache = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".cache")
        });
    cache.join("mass-spectrometry-counts")
}

/// Download the GeMS-A10 dataset to `cache_dir`, returning the path to the file.
///
/// If the file already exists, skips the download.
pub fn download_dataset(
    cache_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    fs::create_dir_all(cache_dir)?;
    let dest = cache_dir.join(GEMS_A10_FILENAME);

    if dest.exists() {
        eprintln!("Dataset already cached at {}", dest.display());
        return Ok(dest);
    }

    eprintln!("Downloading GeMS-A10 dataset from HuggingFace...");
    eprintln!("URL: {}", GEMS_A10_URL);

    let response = reqwest::blocking::Client::new()
        .get(GEMS_A10_URL)
        .send()?
        .error_for_status()?;

    let total_size = response.content_length();

    let pb = if let Some(size) = total_size {
        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})"
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {bytes} downloaded")
                .unwrap(),
        );
        pb
    };

    // Write to a temp file first, then rename to avoid partial files
    let tmp_dest = cache_dir.join(format!(".{}.part", GEMS_A10_FILENAME));
    let mut file = fs::File::create(&tmp_dest)?;

    let mut downloaded: u64 = 0;
    let mut reader = response;
    let mut buf = vec![0u8; 256 * 1024]; // 256 KB buffer

    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    file.flush()?;
    drop(file);
    fs::rename(&tmp_dest, &dest)?;

    pb.finish_with_message("Download complete");
    eprintln!("Saved to {}", dest.display());

    Ok(dest)
}
