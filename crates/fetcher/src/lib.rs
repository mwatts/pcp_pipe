//! Fetcher: Rust-only downloading and RSS expansion.

use anyhow::{Context, Result, anyhow, bail};
use feed_rs::model::Feed;
use feed_rs::parser;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
};
use tracing::info;
use url::Url;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FetchResult {
    pub source_url: String,
    pub final_url: String,
    pub saved_path: String,
    pub sha256: String,
    pub content_type: Option<String>,
    pub original_filename: Option<String>,
    pub new_filename: String,
    pub title: Option<String>,
    pub etag: Option<String>,
    pub content_length: Option<u64>,
    pub last_modified: Option<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub bytes_written: Option<u64>,
    pub canonical_url: Option<String>,
    pub headers_head: Option<HashMap<String, String>>,
    pub headers_get: Option<HashMap<String, String>>,
}

/// Download a URL to the output directory. Supports resume if server allows ranges.
pub async fn fetch_to_file(url: &str, out_dir: &str) -> Result<FetchResult> {
    let parsed = Url::parse(url).map_err(|e| anyhow!(e))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        bail!("unsupported scheme: {}", parsed.scheme());
    }
    fs::create_dir_all(out_dir).await?;

    let client = reqwest::Client::builder()
        .user_agent("pcp-fetcher/0.1")
        .http2_adaptive_window(true)
        .redirect(reqwest::redirect::Policy::limited(10))
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // HEAD to discover metadata and possible filename
    let head = client.head(parsed.as_str()).send().await?;
    let mut final_url = head.url().to_string();
    let mut ctype = head
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let head_headers = Some(headers_map(head.headers()));
    let head_etag = head
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let head_content_length = head
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let head_last_modified = head
        .headers()
        .get(reqwest::header::LAST_MODIFIED)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let cd_filename = head
        .headers()
        .get(reqwest::header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_cd_filename);

    // Optional title captured during resolver parsing
    let mut page_title: Option<String> = None;
    let canonical_url: Option<String> = None;

    // If this looks non-audio (e.g., text/html or application/xml), fetch and attempt recursive resolution
    if let Some(ref ct) = ctype {
        if ct.starts_with("text/html")
            || ct.contains("xml")
            || ct.contains("rss")
            || ct.contains("atom")
        {
            let (resolved_url, pt, discovered_ct) =
                resolve_from_document(&client, &final_url).await?;
            if let Some(u) = resolved_url {
                final_url = u;
            }
            if let Some(ct2) = discovered_ct {
                ctype = Some(ct2);
            }
            // Capture page title for filename and metadata
            page_title = pt;
            // Avoid re-fetching the (potentially large) final URL just to infer canonical.
            // If needed later, we can extend the resolver to return canonical explicitly.
        }
    }

    // Build a filename with preference order:
    // 1) Content-Disposition filename
    // 2) Page <title> (if present)
    // 3) Derived from final URL
    let mut filename = if let Some(name) = &cd_filename {
        name.clone()
    } else if let Some(t) = &page_title {
        slugify(t)
    } else {
        filename_from_url(&final_url)?
    };
    // Append extension from content-type if missing and obvious
    if Path::new(&filename).extension().is_none() {
        if let Some(ext) = infer_audio_extension(ctype.as_deref(), &final_url)
            .or_else(|| ext_from_url_path(&final_url))
        {
            filename.push('.');
            filename.push_str(&ext);
        }
    }

    let original_filename = Some(filename.clone());
    let dest = Path::new(out_dir).join(sanitize_filename(&filename));

    // Try ranged resume
    // ETag sidecar handling
    let etag_path = dest.with_extension(format!(
        "{}etag",
        dest.extension()
            .and_then(|s| s.to_str())
            .map(|s| format!("{}.", s))
            .unwrap_or_default()
    ));
    let sha_path = dest.with_extension(format!(
        "{}sha256",
        dest.extension()
            .and_then(|s| s.to_str())
            .map(|s| format!("{}.", s))
            .unwrap_or_default()
    ));

    // If file and sha256 sidecar exist, verify and early return without GET
    if dest.exists() && fs::metadata(&sha_path).await.is_ok() {
        info!(path=?dest, sha_path=?sha_path, "fetcher: existing file and sha256 sidecar found; verifying");
        if let Ok(expected) = fs::read_to_string(&sha_path).await {
            let expected = expected.trim().to_string();
            if !expected.is_empty() {
                let actual = compute_sha256_file(&dest).await?;
                if actual == expected {
                    info!(path=?dest, sha256=%actual, "fetcher: checksum verified; skipping GET");
                    let bytes_written = dest.metadata().ok().map(|m| m.len());
                    let now = chrono::Utc::now().to_rfc3339();
                    let new_filename = dest.file_name().unwrap().to_string_lossy().into_owned();
                    return Ok(FetchResult {
                        source_url: url.to_string(),
                        final_url,
                        saved_path: dest.display().to_string(),
                        sha256: actual,
                        content_type: ctype,
                        original_filename,
                        new_filename,
                        title: page_title,
                        etag: head_etag,
                        content_length: head_content_length,
                        last_modified: head_last_modified,
                        started_at: Some(now.clone()),
                        completed_at: Some(now),
                        bytes_written,
                        canonical_url,
                        headers_head: head_headers,
                        headers_get: None,
                    });
                } else {
                    // Corrupt or incomplete file; remove and restart download fresh (no resume/If-None-Match)
                    info!(path=?dest, expected=%expected, actual=%actual, "fetcher: checksum mismatch; deleting and re-downloading");
                    let _ = fs::remove_file(&dest).await;
                    let _ = fs::remove_file(&sha_path).await;
                    let _ = fs::remove_file(&etag_path).await;
                }
            }
        }
    }

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
    let started_at = Some(chrono::Utc::now().to_rfc3339());
    let resp = builder.send().await?;
    if resp.status() == reqwest::StatusCode::NOT_MODIFIED {
        // Nothing to do, keep existing file; compute checksum if missing
        let sha256 = if let Ok(s) = fs::read_to_string(&sha_path).await {
            info!(path=?dest, "fetcher: 304 Not Modified; checksum sidecar present");
            s.trim().to_string()
        } else {
            info!(path=?dest, "fetcher: 304 Not Modified; checksum missing; recomputing");
            let calc = compute_sha256_file(&dest).await?;
            let _ = fs::write(&sha_path, &calc).await;
            info!(sha_path=?sha_path, sha256=%calc, "fetcher: wrote sha256 sidecar after 304");
            calc
        };
        let bytes_written = dest.metadata().ok().map(|m| m.len());
        return Ok(FetchResult {
            source_url: url.to_string(),
            final_url,
            saved_path: dest.display().to_string(),
            sha256,
            content_type: ctype,
            original_filename: original_filename.clone(),
            new_filename: dest.file_name().unwrap().to_string_lossy().into(),
            title: page_title,
            etag: head_etag,
            content_length: head_content_length,
            last_modified: head_last_modified,
            started_at,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
            bytes_written,
            canonical_url,
            headers_head: head_headers,
            headers_get: None,
        });
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
    let get_headers = Some(headers_map(resp.headers()));
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
    let final_etag = etag_received.clone().or(head_etag);
    if let Some(etag) = &final_etag {
        let _ = fs::write(&etag_path, etag).await;
    }
    let _ = fs::write(&sha_path, &sha256).await;
    let new_filename = dest.file_name().unwrap().to_string_lossy().into_owned();
    // Write metadata sidecar JSON
    let completed_at = Some(chrono::Utc::now().to_rfc3339());
    let bytes_written = dest.metadata().ok().map(|m| m.len());
    let meta = FetchResult {
        source_url: url.to_string(),
        final_url: final_url.clone(),
        saved_path: dest.display().to_string(),
        sha256: sha256.clone(),
        content_type: ctype.clone(),
        original_filename: original_filename.clone(),
        new_filename: new_filename.clone(),
        title: page_title.clone(),
        etag: final_etag.clone(),
        content_length: head_content_length,
        last_modified: head_last_modified.clone(),
        started_at: started_at.clone(),
        completed_at: completed_at.clone(),
        bytes_written,
        canonical_url: canonical_url.clone(),
        headers_head: head_headers.clone(),
        headers_get: get_headers.clone(),
    };
    let _ = fs::write(
        dest.with_extension(format!(
            "{}json",
            dest.extension()
                .and_then(|s| s.to_str())
                .map(|s| format!("{}.", s))
                .unwrap_or_default()
        )),
        serde_json::to_vec_pretty(&meta)?,
    )
    .await;

    Ok(FetchResult {
        source_url: url.to_string(),
        final_url,
        saved_path: dest.display().to_string(),
        sha256,
        content_type: ctype,
        original_filename,
        new_filename,
        title: page_title,
        etag: final_etag,
        content_length: head_content_length,
        last_modified: head_last_modified,
        started_at,
        completed_at,
        bytes_written,
        canonical_url,
        headers_head: head_headers,
        headers_get: get_headers,
    })
}

use futures_util::StreamExt;

fn filename_from_url(url: &str) -> Result<String> {
    let parsed = Url::parse(url)?;
    if let Some(seg) = parsed
        .path_segments()
        .and_then(|s| s.last())
        .filter(|s| !s.is_empty())
    {
        return Ok(seg.to_string());
    }
    Ok("download".to_string())
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if matches!(
                c,
                '/' | '\\' | '"' | '\n' | '\r' | '\t' | '<' | '>' | '|' | ':' | '*'
            ) {
                '_'
            } else {
                c
            }
        })
        .collect()
}

fn infer_audio_extension(ctype: Option<&str>, url: &str) -> Option<String> {
    // Try content-type first
    if let Some(ct) = ctype {
        let ct = ct.to_ascii_lowercase();
        if ct.starts_with("audio/") {
            if ct.contains("mpeg") {
                return Some("mp3".into());
            }
            if ct.contains("mp4") || ct.contains("aac") || ct.contains("x-m4a") {
                return Some("m4a".into());
            }
            if ct.contains("wav") || ct.contains("x-wav") {
                return Some("wav".into());
            }
            if ct.contains("flac") {
                return Some("flac".into());
            }
            if ct.contains("ogg") || ct.contains("vorbis") {
                return Some("ogg".into());
            }
            if ct.contains("opus") {
                return Some("opus".into());
            }
        }
    }
    // Fallback: guess from URL path
    let guess = mime_guess::from_path(url).first_or_octet_stream();
    if guess.type_() == mime::AUDIO {
        let sub = guess.subtype();
        if sub == mime::MPEG {
            return Some("mp3".into());
        }
        if sub == mime::MP4 || sub.as_str() == "aac" || sub.as_str() == "x-m4a" {
            return Some("m4a".into());
        }
        if sub.as_str() == "wav" || sub.as_str() == "x-wav" {
            return Some("wav".into());
        }
        if sub.as_str() == "flac" {
            return Some("flac".into());
        }
        if sub.as_str() == "ogg" || sub.as_str() == "vorbis" {
            return Some("ogg".into());
        }
        if sub.as_str() == "opus" {
            return Some("opus".into());
        }
    }
    None
}

fn ext_from_url_path(url: &str) -> Option<String> {
    if let Ok(u) = Url::parse(url) {
        if let Some(last) = u.path_segments().and_then(|s| s.last()) {
            if let Some((_, ext)) = last.rsplit_once('.') {
                let ext = ext.to_ascii_lowercase();
                let known = [
                    "mp3", "m4a", "m4b", "aac", "wav", "flac", "ogg", "opus", "mp4",
                ];
                if known.contains(&ext.as_str()) {
                    return Some(ext);
                }
            }
        }
    }
    None
}

fn slugify(s: &str) -> String {
    // Basic, dependency-free slug: lowercase, map non-alnum to hyphen, collapse repeats, trim, limit length
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if c.is_ascii() {
            // Map spaces, dashes, underscores and most punctuation to hyphen
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        } else {
            // For non-ASCII, replace with hyphen separator as a safe default
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        }
    }
    // Trim leading/trailing hyphens
    while out.starts_with('-') {
        out.remove(0);
    }
    while out.ends_with('-') {
        out.pop();
    }
    // Collapse multiple dashes already ensured; enforce max length
    if out.len() > 120 {
        out.truncate(120);
    }
    if out.is_empty() {
        "download".to_string()
    } else {
        out
    }
}

fn headers_map(h: &reqwest::header::HeaderMap) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (k, v) in h.iter() {
        if let Ok(val) = v.to_str() {
            map.insert(k.as_str().to_string(), val.to_string());
        }
    }
    map
}

async fn resolve_from_document(
    client: &reqwest::Client,
    url: &str,
) -> Result<(Option<String>, Option<String>, Option<String>)> {
    let mut current = url.to_string();
    let mut title_acc: Option<String> = None;
    let mut last_ct: Option<String> = None;
    for _ in 0..4 {
        let res = client.get(&current).send().await?.error_for_status()?;
        let ct = res
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        last_ct = ct.clone();
        let body = res.text().await?;
        // Try RSS/Atom first
        if body.contains("<rss") || body.contains("<feed") {
            if let Ok(urls) = extract_feed_audio_urls(&body) {
                if let Some(u) = urls.first() {
                    return Ok((Some(u.clone()), title_acc, Some("audio/*".to_string())));
                }
            }
        }
        // HTML parse
        let doc = scraper::Html::parse_document(&body);
        if let Ok(sel) = scraper::Selector::parse("title") {
            if let Some(el) = doc.select(&sel).next() {
                let t = el.text().collect::<String>().trim().to_string();
                if !t.is_empty() && title_acc.is_none() {
                    title_acc = Some(t);
                }
            }
        }
        if let Ok(sel) = scraper::Selector::parse("meta[property=og:title], meta[name=og:title]") {
            if let Some(el) = doc.select(&sel).next() {
                if let Some(c) = el.value().attr("content") {
                    let t = c.trim();
                    if !t.is_empty() && title_acc.is_none() {
                        title_acc = Some(t.to_string());
                    }
                }
            }
        }
        // Meta audio
        if let Ok(sel) = scraper::Selector::parse(
            "meta[property=og:audio], meta[name=og:audio], meta[property=og:audio:secure_url]",
        ) {
            for el in doc.select(&sel) {
                if let Some(c) = el.value().attr("content") {
                    if let Ok(resolved) = Url::parse(&current).and_then(|base| base.join(c)) {
                        let href = resolved.to_string();
                        let guess = mime_guess::from_path(&href).first_or_octet_stream();
                        if guess.type_() == mime::AUDIO
                            || href.ends_with(".mp3")
                            || href.ends_with(".m4a")
                            || href.ends_with(".wav")
                            || href.ends_with(".flac")
                            || href.ends_with(".ogg")
                        {
                            return Ok((
                                Some(href),
                                title_acc,
                                Some(guess.essence_str().to_string()),
                            ));
                        }
                    }
                }
            }
        }
        // <audio src> and audio>source
        if let Ok(sel_audio) = scraper::Selector::parse("audio") {
            for el in doc.select(&sel_audio) {
                if let Some(src) = el.value().attr("src") {
                    if let Ok(resolved) = Url::parse(&current).and_then(|base| base.join(src)) {
                        let href = resolved.to_string();
                        let guess = mime_guess::from_path(&href).first_or_octet_stream();
                        if guess.type_() == mime::AUDIO
                            || href.ends_with(".mp3")
                            || href.ends_with(".m4a")
                            || href.ends_with(".wav")
                            || href.ends_with(".flac")
                            || href.ends_with(".ogg")
                        {
                            return Ok((
                                Some(href),
                                title_acc,
                                Some(guess.essence_str().to_string()),
                            ));
                        }
                    }
                }
            }
            if let Ok(sel_source) = scraper::Selector::parse("audio source") {
                for el in doc.select(&sel_source) {
                    if let Some(src) = el.value().attr("src") {
                        if let Ok(resolved) = Url::parse(&current).and_then(|base| base.join(src)) {
                            let href = resolved.to_string();
                            let guess = mime_guess::from_path(&href).first_or_octet_stream();
                            if guess.type_() == mime::AUDIO
                                || href.ends_with(".mp3")
                                || href.ends_with(".m4a")
                                || href.ends_with(".wav")
                                || href.ends_with(".flac")
                                || href.ends_with(".ogg")
                            {
                                return Ok((
                                    Some(href),
                                    title_acc,
                                    Some(guess.essence_str().to_string()),
                                ));
                            }
                        }
                    }
                }
            }
        }
        // <a href> heuristic
        if let Ok(sel) = scraper::Selector::parse("a") {
            for el in doc.select(&sel) {
                if let Some(href_attr) = el.value().attr("href") {
                    if let Ok(resolved) = Url::parse(&current).and_then(|base| base.join(href_attr))
                    {
                        let href = resolved.to_string();
                        let guess = mime_guess::from_path(&href).first_or_octet_stream();
                        if guess.type_() == mime::AUDIO
                            || href.ends_with(".mp3")
                            || href.ends_with(".m4a")
                            || href.ends_with(".wav")
                            || href.ends_with(".flac")
                            || href.ends_with(".ogg")
                        {
                            return Ok((
                                Some(href),
                                title_acc,
                                Some(guess.essence_str().to_string()),
                            ));
                        }
                    }
                }
            }
        }
        // Follow canonical/og:url
        if let Some(next) = find_canonical_url(&doc, &current) {
            if next != current {
                tracing::debug!(from=%current, to=%next, "following canonical/og:url");
                current = next;
                continue;
            }
        }
        break; // couldn't resolve further
    }
    Ok((None, title_acc, last_ct))
}

fn find_canonical_url(doc: &scraper::Html, base: &str) -> Option<String> {
    // Try various selectors to be resilient
    let selectors = [
        "link[rel=canonical]",
        "link[rel~=canonical]",
        "meta[property=og:url]",
        "meta[name=og:url]",
    ];
    for sel_str in selectors {
        if let Ok(sel) = scraper::Selector::parse(sel_str) {
            for el in doc.select(&sel) {
                let href = el
                    .value()
                    .attr("href")
                    .or_else(|| el.value().attr("content"));
                if let Some(h) = href {
                    if let Ok(resolved) = Url::parse(base).and_then(|b| b.join(h)) {
                        return Some(resolved.to_string());
                    }
                }
            }
        }
    }
    None
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
            if pieces.len() == 2 {
                return Some(pieces[1].to_string());
            }
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

async fn compute_sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use sha2::{Digest, Sha256};
    use tokio::fs;

    #[tokio::test]
    async fn resume_with_range_206() {
        let mut server = Server::new_async().await;
        let path = "/audio.mp3";
        // HEAD response
        let _m1 = server
            .mock("HEAD", path)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();

        // Expect GET with Range header starting at 6
        let _m2 = server
            .mock("GET", path)
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

        let res = fetch_to_file(&url, out_dir.to_str().unwrap())
            .await
            .unwrap();
        let data = fs::read(&res.saved_path).await.unwrap();
        assert_eq!(data, b"hello world");
    }

    #[tokio::test]
    async fn not_modified_304_with_etag() {
        let mut server = Server::new_async().await;
        let path = "/file";
        // HEAD for both runs
        let _h = server
            .mock("HEAD", path)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();
        // First GET returns content and ETag
        let _g1 = server
            .mock("GET", path)
            .with_status(200)
            .with_header("etag", "\"abc\"")
            .with_body("xyz")
            .create();

        let url = format!("{}{}", server.url(), path);
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path();
        let res1 = fetch_to_file(&url, out_dir.to_str().unwrap())
            .await
            .unwrap();
        let sha1 = res1.sha256.clone();
        assert!(!sha1.is_empty());

        // Second GET responds 304 when If-None-Match is provided
        let _g2 = server
            .mock("GET", path)
            .match_header("if-none-match", "\"abc\"")
            .with_status(304)
            .create();

        let res2 = fetch_to_file(&url, out_dir.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(res2.sha256, sha1);
    }

    #[tokio::test]
    async fn filename_from_content_disposition_and_type() {
        let mut server = Server::new_async().await;
        // Case 1: filename from Content-Disposition
        let p1 = "/download";
        let _h1 = server
            .mock("HEAD", p1)
            .with_status(200)
            .with_header("content-disposition", "attachment; filename=\"foo.mp3\"")
            .create();
        let _g1 = server.mock("GET", p1).with_status(200).create();
        let url1 = format!("{}{}", server.url(), p1);
        let dir1 = tempfile::tempdir().unwrap();
        let res1 = fetch_to_file(&url1, dir1.path().to_str().unwrap())
            .await
            .unwrap();
        assert!(res1.saved_path.ends_with("foo.mp3"));

        // Case 2: extension from Content-Type
        let p2 = "/file";
        let _h2 = server
            .mock("HEAD", p2)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();
        let _g2 = server.mock("GET", p2).with_status(200).create();
        let url2 = format!("{}{}", server.url(), p2);
        let dir2 = tempfile::tempdir().unwrap();
        let res2 = fetch_to_file(&url2, dir2.path().to_str().unwrap())
            .await
            .unwrap();
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

    #[test]
    fn sample_html_canonical_is_found() {
        // Verify we can extract canonical from the provided sample HTML fixture
        let html =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/sample.html"))
                .unwrap();
        let doc = scraper::Html::parse_document(&html);
        let base = "https://overcast.fm/+ZSKful01Q";
        let canon = super::find_canonical_url(&doc, base);
        assert_eq!(canon.as_deref(), Some("https://vftb.net/?p=10888"));
    }

    #[tokio::test]
    async fn html_title_preferred_and_audio_resolution() {
        let mut server = Server::new_async().await;
        // Serve an HTML resolver page using sample.html contents, but append an <audio> tag to ensure discoverable audio
        let mut html =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/sample.html"))
                .unwrap();
        html.push_str(&format!(
            "\n<audio src=\"{}/audio.mp3\"></audio>",
            server.url()
        ));

        let path = "/resolver";
        let _h = server
            .mock("HEAD", path)
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .create();
        let _g = server
            .mock("GET", path)
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .with_body(html)
            .create();

        // The audio itself
        let _ha = server
            .mock("HEAD", "/audio.mp3")
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();
        let _ga = server
            .mock("GET", "/audio.mp3")
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .with_body("ABC")
            .create();

        let url = format!("{}{}", server.url(), path);
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path();
        let res = fetch_to_file(&url, out_dir.to_str().unwrap())
            .await
            .unwrap();
        // File should be named based on the HTML title and have .mp3 extension
        assert!(res.new_filename.ends_with(".mp3"));
        assert!(res.title.is_some());
        let saved = fs::read(&res.saved_path).await.unwrap();
        assert_eq!(saved, b"ABC");
        // Metadata sidecar should contain title
        let meta_path = std::path::Path::new(&res.saved_path).with_extension("mp3.json");
        let meta: FetchResult =
            serde_json::from_slice(&fs::read(meta_path).await.unwrap()).unwrap();
        assert!(meta.title.is_some());
    }

    #[tokio::test]
    async fn html_follow_canonical_then_resolve_audio() {
        let mut server = Server::new_async().await;
        // Rewrite sample.html canonical to point to a second page on our mock server
        let sample =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/sample.html"))
                .unwrap();
        let rewritten = sample.replace(
            "https://vftb.net/?p=10888",
            &format!("{}/page2", server.url()),
        );

        let _h1 = server
            .mock("HEAD", "/page1")
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .create();
        let _g1 = server
            .mock("GET", "/page1")
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .with_body(rewritten)
            .create();

        // Page2 contains an <a> to audio
        let body2 = format!(
            "<html><head><title>Page 2</title></head><body><a href=\"{}/song.m4a\">Download</a></body></html>",
            server.url()
        );
        let _h2 = server
            .mock("HEAD", "/page2")
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .create();
        let _g2 = server
            .mock("GET", "/page2")
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .with_body(body2)
            .create();

        // Audio endpoint
        let _ha = server
            .mock("HEAD", "/song.m4a")
            .with_status(200)
            .with_header("content-type", "audio/mp4")
            .create();
        let _ga = server
            .mock("GET", "/song.m4a")
            .with_status(200)
            .with_header("content-type", "audio/mp4")
            .with_body("DATA")
            .create();

        let url = format!("{}/page1", server.url());
        let dir = tempfile::tempdir().unwrap();
        let res = fetch_to_file(&url, dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert!(
            res.final_url.ends_with(".m4a"),
            "final_url was {}",
            res.final_url
        );
        assert!(
            res.new_filename.ends_with(".m4a"),
            "new_filename was {} with ctype {:?}",
            res.new_filename,
            res.content_type
        );
        let saved = fs::read(&res.saved_path).await.unwrap();
        assert_eq!(saved, b"DATA");
    }

    #[tokio::test]
    async fn skip_download_when_sha_sidecar_matches() {
        let mut server = Server::new_async().await;
        let path = "/song.mp3";
        // Provide HEAD for metadata discovery
        let _h = server
            .mock("HEAD", path)
            .with_status(200)
            .with_header("content-type", "audio/mpeg")
            .create();
        // Intentionally do NOT provide a GET mock; early return must avoid GET

        let url = format!("{}{}", server.url(), path);
        let dir = tempfile::tempdir().unwrap();
        let out_dir = dir.path();

        // Prepare existing file and matching sha256 sidecar
        let dest = out_dir.join("song.mp3");
        fs::write(&dest, b"OK").await.unwrap();
        let expected = {
            let mut hasher = Sha256::new();
            hasher.update(b"OK");
            hex::encode(hasher.finalize())
        };
        let sha_path = dest.with_extension("mp3.sha256");
        fs::write(&sha_path, &expected).await.unwrap();

        let res = fetch_to_file(&url, out_dir.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(res.sha256, expected);
        assert!(res.saved_path.ends_with("song.mp3"));
    }
}
