use anyhow::Result;
use clap::Parser;
use pcp_pipeline::run_one;
use pcp_utils::init_tracing;
use std::path::PathBuf;
use tokio::fs;

#[derive(Parser, Debug)]
#[command(name = "pcp", about = "PCP Pipe CLI", version)]
struct Args {
    /// Output directory
    #[arg(long, default_value = "./podcast_output")]
    output_dir: String,

    /// Disable GPU acceleration
    #[arg(long)]
    no_gpu: bool,

    /// Whisper model name (tiny|base|small|medium|large-v3)
    #[arg(long, default_value = "base")]
    whisper_model: String,

    /// Source URL(s) to process
    #[arg(value_name = "URL", num_args = 1..)]
    urls: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    let args = Args::parse();

    for url in &args.urls {
        tracing::info!(url, output_dir = %args.output_dir, "starting pipeline run");
        let result = run_one(url, &args.output_dir).await?;
        tracing::info!(path = %result.audio_file_path, "download complete");
        let json = serde_json::to_string_pretty(&result)?;
        println!("{}", json);
        // Save JSON next to output dir with a stable name derived from saved file stem
        let out_dir = PathBuf::from(&args.output_dir);
        fs::create_dir_all(&out_dir).await.ok();
        let audio_path_buf = PathBuf::from(&result.audio_file_path);
        let stem = audio_path_buf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("result");
        let out_path = out_dir.join(format!("{}_results.json", stem));
        fs::write(out_path, json).await?;
    }

    Ok(())
}
