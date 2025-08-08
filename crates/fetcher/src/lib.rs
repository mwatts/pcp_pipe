//! Fetcher: Rust-only downloading and RSS expansion.

use anyhow::{anyhow, bail, Context, Result};
use feed_rs::model::Feed;
use feed_rs::parser;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::{fs, io::AsyncWriteExt};
use tracing::info;
use url::Url;

#[derive(Debug, Clone)]
pub struct FetchResult {
    pub final_url: String,
    pub saved_path: String,
    pub sha256: String,
    pub content_type: Option<String>,
}

/// Download a URL to the output directory. Supports resume if server allows ranges.
pub async fn fetch_to_file(url: &str, out_dir: &str) -> Result<FetchResult> {
    let parsed = Url::parse(url).map_err(|e| anyhow!(e))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        bail!("unsupported scheme: {}", parsed.scheme());
    }
    fs::create_dir_all(out_dir).await.ok();

    let client = reqwest::Client::builder()
        .user_agent("pcp-fetcher/0.1")
        .http2_adaptive_window(true)
        .build()?;

    // HEAD to discover metadata and possible filename
    let head = client.head(parsed.as_str()).send().await?;
    let final_url = head.url().to_string();
    let ctype = head.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    let cd_filename = head
        .headers()
        .get(reqwest::header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_cd_filename);

    let mut filename = filename_from_url(&final_url)?;
    if let Some(name) = cd_filename { filename = name; }
    // Append extension from content-type if missing and obvious
    if Path::new(&filename).extension().is_none() {
        if let Some(ct) = &ctype {
            if ct.contains("audio/mpeg") {
                filename.push_str(".mp3");
            } else if ct.contains("audio/mp4") || ct.contains("audio/aac") {
                filename.push_str(".m4a");
            } else if ct.contains("audio/wav") {
                filename.push_str(".wav");
            } else if ct.contains("audio/flac") {
                filename.push_str(".flac");
            } else if ct.contains("ogg") {
                filename.push_str(".ogg");
            }
        }
    }

    let dest = Path::new(out_dir).join(sanitize_filename(&filename));

    // Try ranged resume
        // ETag sidecar handling
        let etag_path = dest.with_extension(format!("{}etag", dest.extension().and_then(|s| s.to_str()).map(|s| format!("{}.", s)).unwrap_or_default()));
        let sha_path = dest.with_extension(format!("{}sha256", dest.extension().and_then(|s| s.to_str()).map(|s| format!("{}.", s)).unwrap_or_default()));

        // Try ranged resume and conditional GET via If-None-Match
        let mut builder = client.get(final_url.clone());
        let mut offset: u64 = 0;
        if dest.exists() {
            if let Ok(meta) = dest.metadata() {
                offset = meta.len();
                if offset > 0 {
                    builder = builder.header(reqwest::header::RANGE, format!("bytes={}-", offset));
                }
            }
            if let Ok(etag) = fs::read_to_string(&etag_path).await {
                let etag = etag.trim();
                if !etag.is_empty() {
                    builder = builder.header(reqwest::header::IF_NONE_MATCH, etag.to_string());
                }
            }
        }
    let resp = builder.send().await?;
    if resp.status() == reqwest::StatusCode::NOT_MODIFIED {
        // Nothing to do, keep existing file and compute checksum if missing
        let sha256 = if let Ok(s) = fs::read_to_string(&sha_path).await { s.trim().to_string() } else { String::new() };
        return Ok(FetchResult { final_url, saved_path: dest.display().to_string(), sha256, content_type: ctype });
    }
    resp.error_for_status_ref()?;

    // If resuming, include existing bytes in hash calculation
    let mut hasher = Sha256::new();
    if offset > 0 {
        if let Ok(existing) = fs::read(&dest).await {
            hasher.update(&existing);
        }
    }

    let mut file = if offset > 0 {
        info!(path=?dest, offset, "resuming download");
        fs::OpenOptions::new().append(true).open(&dest).await?
    } else {
        fs::File::create(&dest).await?
    };
    // Capture ETag before consuming response
    let etag_received = resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await.transpose().context("stream error")? {
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
    }

    let sha256 = hex::encode(hasher.finalize());
    // Persist sidecars
    if let Some(etag) = etag_received {
        let _ = fs::write(&etag_path, etag).await;
    }
    let _ = fs::write(&sha_path, &sha256).await;
    Ok(FetchResult {
        final_url,
        saved_path: dest.display().to_string(),
        sha256,
        content_type: ctype,
    })
}

use futures_util::StreamExt;

fn filename_from_url(url: &str) -> Result<String> {
    let parsed = Url::parse(url)?;
    if let Some(seg) = parsed.path_segments().and_then(|s| s.last()).filter(|s| !s.is_empty()) {
        return Ok(seg.to_string());
    }
    Ok("download".to_string())
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| if matches!(c, '/' | '\\' | '"' | '\n' | '\r' | '\t' | '<' | '>' | '|' | ':' | '*') { '_' } else { c })
        .collect()
}

fn parse_cd_filename(raw: &str) -> Option<String> {
    // Very small parser for: attachment; filename="foo.mp3"; filename*=UTF-8''bar.mp3
    // Prefer RFC 5987 filename* if present
    for part in raw.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("filename*=") {
            // e.g., UTF-8''bar.mp3
            let rest = rest.trim_matches('"');
            let pieces: Vec<&str> = rest.splitn(2, "''").collect();
            if pieces.len() == 2 { return Some(pieces[1].to_string()); }
        }
    }
    for part in raw.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("filename=") {
            return Some(rest.trim_matches('"').to_string());
        }
    }
    None
}

/// Parse an RSS/Atom feed and return enclosure audio URLs.
pub fn extract_feed_audio_urls(xml: &str) -> Result<Vec<String>> {
    let feed: Feed = parser::parse(xml.as_bytes()).context("failed to parse feed")?;
    let mut urls = Vec::new();
    for entry in feed.entries {
        for link in entry.links {
            if let Some(ct) = &link.media_type {
                if ct.starts_with("audio/") {
                    urls.push(link.href);
                }
            }
        }
    }
    if urls.is_empty() {
        bail!("no audio enclosures found in feed")
    }
    Ok(urls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use tokio::fs;

    #[tokio::test]
    async fn resume_with_range_206() {
        let mut server = Server::new_async().await;
        let path = "/audio.mp3";
        // HEAD response
        let _m1 = server.mock("HEAD", path)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();

        // Expect GET with Range header starting at 6
        let _m2 = server.mock("GET", path)
            .match_header("range", "bytes=6-")
            .with_status(206)
            .with_body("world")
            .create();

        let url = format!("{}{}", server.url(), path);
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path();
        // Pre-create partial file "hello "
        let pre = out_dir.join("audio.mp3");
        fs::write(&pre, b"hello ").await.unwrap();

        let res = fetch_to_file(&url, out_dir.to_str().unwrap()).await.unwrap();
        let data = fs::read(&res.saved_path).await.unwrap();
        assert_eq!(data, b"hello world");
    }

    #[tokio::test]
    async fn not_modified_304_with_etag() {
        let mut server = Server::new_async().await;
        let path = "/file";
        // HEAD for both runs
        let _h = server.mock("HEAD", path)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();
        // First GET returns content and ETag
        let _g1 = server.mock("GET", path)
            .with_status(200)
            .with_header("etag", "\"abc\"")
            .with_body("xyz")
            .create();

        let url = format!("{}{}", server.url(), path);
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path();
        let res1 = fetch_to_file(&url, out_dir.to_str().unwrap()).await.unwrap();
        let sha1 = res1.sha256.clone();
        assert!(!sha1.is_empty());

        // Second GET responds 304 when If-None-Match is provided
        let _g2 = server.mock("GET", path)
            .match_header("if-none-match", "\"abc\"")
            .with_status(304)
            .create();

        let res2 = fetch_to_file(&url, out_dir.to_str().unwrap()).await.unwrap();
        assert_eq!(res2.sha256, sha1);
    }

    #[tokio::test]
    async fn filename_from_content_disposition_and_type() {
        let mut server = Server::new_async().await;
        // Case 1: filename from Content-Disposition
        let p1 = "/download";
        let _h1 = server.mock("HEAD", p1)
            .with_status(200)
            .with_header("content-disposition", "attachment; filename=\"foo.mp3\"")
            .create();
        let _g1 = server.mock("GET", p1).with_status(200).create();
        let url1 = format!("{}{}", server.url(), p1);
        let dir1 = tempfile::tempdir().unwrap();
        let res1 = fetch_to_file(&url1, dir1.path().to_str().unwrap()).await.unwrap();
        assert!(res1.saved_path.ends_with("foo.mp3"));

        // Case 2: extension from Content-Type
        let p2 = "/file";
        let _h2 = server.mock("HEAD", p2)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();
        let _g2 = server.mock("GET", p2).with_status(200).create();
        let url2 = format!("{}{}", server.url(), p2);
        let dir2 = tempfile::tempdir().unwrap();
        let res2 = fetch_to_file(&url2, dir2.path().to_str().unwrap()).await.unwrap();
        assert!(res2.saved_path.ends_with("file.mp3"));
    }

    #[test]
    fn test_filename_sanitize() {
        assert_eq!(sanitize_filename("a/b:c*"), "a_b_c_");
    }

    #[test]
    fn test_extract_feed_audio_urls() {
        let xml = r#"<?xml version='1.0'?>
        <feed xmlns='http://www.w3.org/2005/Atom'>
          <title>Test</title>
          <entry>
            <title>Ep1</title>
            <link rel='enclosure' href='https://example.com/foo.mp3' type='audio/mpeg'/>
          </entry>
        </feed>"#;
        let urls = extract_feed_audio_urls(xml).unwrap();
        assert_eq!(urls, vec!["https://example.com/foo.mp3"]);
    }
}
