//! Transcription (M3): whisper-rs integration scaffold.

use anyhow::{Context, Result};
use pcp_decoder::PcmFrames;
use sha2::{Digest, Sha256};
use std::io::Read;
use tracing::instrument;
use whisper_rs::{FullParams, SamplingStrategy};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Transcript {
    pub text: String,
}

pub struct TranscribeOptions<'a> {
    pub model: &'a str, // tiny|base|small|medium|large-v3
    pub no_gpu: bool,
    pub language: Option<&'a str>, // Some("auto") for autodetect or specific ISO code
    pub gpu_backend: Option<&'a str>, // cpu|metal|opencl (compile-time feature dependent)
}

#[instrument(skip(pcm, opts))]
pub fn transcribe_pcm(pcm: &PcmFrames, opts: &TranscribeOptions) -> Result<Transcript> {
    // Resolve model root and spec
    let cfg = resolve_model_root()?;
    let mm = ModelManager::new(cfg);
    let spec = default_model_spec(opts.model).with_context(|| {
        format!(
            "unknown model '{}': expected tiny|base|small|medium|large-v3",
            opts.model
        )
    })?;
    let _path = mm.download_model(&spec)?;
    let ctx = mm.load_model(spec.name)?;

    // Create a state and run full transcription
    let mut state = ctx.create_state().context("create whisper state")?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(num_cpus::get() as i32);
    // Language: autodetect if "auto" or None
    let resolved_lang = resolve_language_opt(opts.language);
    if let Some(ref lang) = resolved_lang {
        params.set_language(Some(lang));
    } else {
        params.set_language(None);
    }
    params.set_translate(false);
    params.set_no_timestamps(true);
    // GPU backend flag placeholder: effective only if compiled with that backend feature.
    if let Some(backend) = opts.gpu_backend {
        tracing::info!(
            backend,
            no_gpu = opts.no_gpu,
            "transcribe: requested GPU backend"
        );
    }

    state.full(params, &pcm.samples).context("whisper full()")?;

    let mut out = String::new();
    let n = state.full_n_segments().context("segments count")?;
    for i in 0..n {
        let seg = state.full_get_segment_text(i).unwrap_or_default();
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(seg.trim());
    }
    Ok(Transcript { text: out })
}

// -------- Model management API --------

#[derive(Debug, Clone)]
pub struct ModelRootConfig {
    pub root: std::path::PathBuf,
}

impl ModelRootConfig {
    // macOS default: ~/Library/Application Support/pcp_pipe/models
    pub fn default() -> Result<Self> {
        let proj = directories::ProjectDirs::from("com.github", "mwatts", "pcp_pipe")
            .context("cannot resolve ProjectDirs")?;
        let root = proj.data_dir().join("models");
        Ok(Self { root })
    }
}

#[derive(Debug, Clone)]
pub struct ModelSpec<'a> {
    pub name: &'a str, // tiny|base|small|medium|large-v3
    pub url: &'a str,  // GGUF download URL
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model_dir: std::path::PathBuf,
    pub model_file: std::path::PathBuf,
    pub support_dir: std::path::PathBuf,
}

pub struct ModelManager {
    root: std::path::PathBuf,
}

impl ModelManager {
    pub fn new(config: ModelRootConfig) -> Self {
        Self { root: config.root }
    }

    // Layout:
    // <root>/models/<model_name>/model.bin
    // <root>/support/<model_name>/...
    pub fn paths_for(&self, model_name: &str) -> ModelPaths {
        let model_dir = self.root.join("models").join(model_name);
        let support_dir = self.root.join("support").join(model_name);
        let model_file = model_dir.join("model.bin");
        ModelPaths {
            model_dir,
            model_file,
            support_dir,
        }
    }

    pub fn ensure_dirs(&self, model_name: &str) -> Result<ModelPaths> {
        let paths = self.paths_for(model_name);
        std::fs::create_dir_all(&paths.model_dir)
            .with_context(|| format!("create {:?}", paths.model_dir))?;
        std::fs::create_dir_all(&paths.support_dir)
            .with_context(|| format!("create {:?}", paths.support_dir))?;
        Ok(paths)
    }

    pub fn download_model(&self, spec: &ModelSpec) -> Result<std::path::PathBuf> {
        let paths = self.ensure_dirs(spec.name)?;
        // If exists, verify checksum if we have it; if mismatch, re-download
        if paths.model_file.exists() {
            if let Some(stored) = self.read_stored_checksum(&paths.model_file)? {
                let actual = self.compute_checksum(&paths.model_file)?;
                if actual != stored {
                    tracing::warn!("model checksum mismatch; re-downloading");
                } else {
                    return Ok(paths.model_file);
                }
            } else {
                // No stored checksum; compute and store for future verifies
                let actual = self.compute_checksum(&paths.model_file)?;
                self.write_stored_checksum(&paths.model_file, &actual)?;
                return Ok(paths.model_file);
            }
        }

        // Perform the blocking network download on a dedicated OS thread to avoid
        // interacting with a running Tokio runtime.
        let url = spec.url.to_string();
        let tmp_path = paths.model_dir.join("model.bin.download");
        let final_path = paths.model_file.clone();
        let checksum_path = Self::checksum_path(&final_path);
        let handle = std::thread::spawn(move || -> Result<()> {
            let mut resp = reqwest::blocking::get(&url).with_context(|| format!("GET {}", url))?;
            let total = resp.content_length();
            let mut hasher = Sha256::new();
            let mut tmp = std::fs::File::create(&tmp_path).context("create temp model file")?;
            let mut downloaded: u64 = 0;
            let mut last_log = std::time::Instant::now();
            let mut buf = [0u8; 1024 * 64];
            loop {
                let n = resp.read(&mut buf).context("read chunk")?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                std::io::Write::write_all(&mut tmp, &buf[..n]).context("write chunk")?;
                downloaded += n as u64;
                if last_log.elapsed().as_millis() > 500 {
                    if let Some(t) = total {
                        let pct = (downloaded as f64 / t as f64) * 100.0;
                        tracing::info!(
                            downloaded,
                            total = t,
                            percent = format!("{pct:.1}"),
                            "downloading model"
                        );
                    } else {
                        tracing::info!(downloaded, "downloading model");
                    }
                    last_log = std::time::Instant::now();
                }
            }
            // Validate size if known
            if let Some(t) = total {
                anyhow::ensure!(downloaded == t, "incomplete download: {downloaded}/{t}");
            }
            let hex = hex::encode(hasher.finalize());
            // Move into place
            std::fs::rename(&tmp_path, &final_path).context("move model into place")?;
            std::fs::write(&checksum_path, format!("{}\n", hex)).context("write checksum file")?;
            Ok(())
        });

        handle
            .join()
            .map_err(|_| anyhow::anyhow!("model download thread panicked"))??;
        Ok(paths.model_file)
    }

    pub fn load_model(&self, model_name: &str) -> Result<whisper_rs::WhisperContext> {
        let paths = self.ensure_dirs(model_name)?;
        let p = paths.model_file;
        let params = whisper_rs::WhisperContextParameters::default();
        let ctx = whisper_rs::WhisperContext::new_with_params(p.to_string_lossy().as_ref(), params)
            .context("load whisper model")?;
        Ok(ctx)
    }

    // Simple update: re-download and replace the model file atomically.
    pub fn update_model(&self, spec: &ModelSpec) -> Result<std::path::PathBuf> {
        let paths = self.ensure_dirs(spec.name)?;
        // Force re-download using the same streaming path as download_model
        // Remove old checksum to avoid confusion
        let _ = std::fs::remove_file(Self::checksum_path(&paths.model_file));
        // Call download_model to perform streaming download+checksum
        self.download_model(spec)
    }

    fn checksum_path(model_file: &std::path::Path) -> std::path::PathBuf {
        model_file.with_extension("bin.sha256")
    }

    fn read_stored_checksum(&self, model_file: &std::path::Path) -> Result<Option<String>> {
        let p = Self::checksum_path(model_file);
        if !p.exists() {
            return Ok(None);
        }
        let s = std::fs::read_to_string(p).context("read checksum file")?;
        Ok(Some(s.trim().to_string()))
    }

    fn write_stored_checksum(&self, model_file: &std::path::Path, hex: &str) -> Result<()> {
        let p = Self::checksum_path(model_file);
        std::fs::write(p, format!("{}\n", hex)).context("write checksum file")
    }

    fn compute_checksum(&self, model_file: &std::path::Path) -> Result<String> {
        use std::io::Read;
        let mut f =
            std::fs::File::open(model_file).with_context(|| format!("open {:?}", model_file))?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 1024 * 64];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex::encode(hasher.finalize()))
    }
}

pub fn default_model_spec(name: &str) -> Option<ModelSpec<'_>> {
    // Using stable ggml models compatible with whisper.cpp/whisper-rs
    let url = match name {
        "tiny" => {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true"
        }
        "base" => {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true"
        }
        "small" => {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true"
        }
        "medium" => {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true"
        }
        "large-v3" => {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true"
        }
        _ => return None,
    };
    Some(ModelSpec { name, url })
}

// Helper to resolve root configuration with optional env override
pub fn resolve_model_root() -> Result<ModelRootConfig> {
    if let Ok(root) = std::env::var("PCP_MODEL_ROOT") {
        return Ok(ModelRootConfig { root: root.into() });
    }
    ModelRootConfig::default()
}

// Small helper to map language option to whisper param value
fn resolve_language_opt(lang_opt: Option<&str>) -> Option<String> {
    match lang_opt {
        Some(s) if !s.eq_ignore_ascii_case("auto") && !s.trim().is_empty() => Some(s.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn checksum_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.bin");
        {
            let mut f = std::fs::File::create(&model_file).unwrap();
            f.write_all(b"hello world").unwrap();
        }
        let mm = ModelManager {
            root: dir.path().to_path_buf(),
        };
        let hex1 = mm.compute_checksum(&model_file).unwrap();
        mm.write_stored_checksum(&model_file, &hex1).unwrap();
        let hex2 = mm.read_stored_checksum(&model_file).unwrap().unwrap();
        assert_eq!(hex1, hex2);

        // Modifying file should change checksum
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&model_file)
                .unwrap();
            f.write_all(b"!").unwrap();
        }
        let hex3 = mm.compute_checksum(&model_file).unwrap();
        assert_ne!(hex1, hex3);
    }

    #[test]
    fn language_autodetect_helper() {
        assert_eq!(super::resolve_language_opt(None), None);
        assert_eq!(super::resolve_language_opt(Some("auto")), None);
        assert_eq!(super::resolve_language_opt(Some("AUTO")), None);
        assert_eq!(super::resolve_language_opt(Some("")), None);
        assert_eq!(
            super::resolve_language_opt(Some("en")),
            Some("en".to_string())
        );
        assert_eq!(
            super::resolve_language_opt(Some(" fr ")),
            Some(" fr ".to_string())
        );
    }
}
