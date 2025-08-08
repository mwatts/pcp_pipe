use anyhow::Result;
use clap::Parser;
use pcp_pipeline::run_one;
use pcp_utils::init_tracing;

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

    /// Source URL(s)
    urls: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    let args = Args::parse();

    for url in &args.urls {
        let result = run_one(url, &args.output_dir).await?;
        let json = serde_json::to_string_pretty(&result)?;
        println!("{}", json);
    }

    Ok(())
}
