//! Decoder: symphonia-based decoding to PCM frames.

use anyhow::{Context, Result};
use std::fs::File;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::ProbeResult;
use tracing::{debug, info};

pub struct PcmFrames {
    pub samples: Vec<f32>, // interleaved mono for now
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn decode_to_pcm(path: &str) -> Result<PcmFrames> {
    // Open media source
    info!(%path, "decoder: opening file");
    let file = File::open(path).with_context(|| format!("open file: {}", path))?;
    let msrc = MediaSourceStream::new(Box::new(file), Default::default());

    // Probe container
    debug!("decoder: probing container");
    let probed: ProbeResult = symphonia::default::get_probe()
        .format(
            &Default::default(),
            msrc,
            &Default::default(),
            &Default::default(),
        )
        .context("probe format")?;
    let mut format = probed.format;

    // Select default audio track
    let track = format
        .default_track()
        .context("no default audio track")?
        .id;
    debug!(track_id=%track, "decoder: selected default audio track");

    // Create decoder
    let params = format
        .tracks()
        .iter()
        .find(|t| t.id == track)
        .unwrap()
        .codec_params
        .clone();
    debug!(
        sample_rate=?params.sample_rate,
        channels=?params.channels.map(|c| c.count()),
        codec=?params.codec,
        "decoder: creating codec"
    );
    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions { verify: false })
        .context("make decoder")?;

    // Collect samples (note: not resampled yet)
    let mut samples = Vec::<f32>::new();
    let mut sample_rate = params.sample_rate.unwrap_or(16_000) as u32;
    let _channels_src = params.channels.map(|c| c.count() as u16).unwrap_or(1);

    let mut pkt_count: u64 = 0;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => {
                pkt_count += 1;
                if pkt_count % 200 == 0 {
                    debug!(packets=pkt_count, "decoder: reading packets");
                }
                p
            }
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(e).context("read packet")?,
        };
        if packet.track_id() != track {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // audio_buf is an AudioBufferRef (Planar or Interleaved, various sample types)
                match audio_buf {
                    AudioBufferRef::F32(buf) => {
                        debug!(frames=buf.frames(), "decoder: f32 buffer");
                        let spec = *buf.spec();
                        sample_rate = spec.rate;
                        let _ = spec.channels.count();
                        let chans = spec.channels.count();
                        let frames = buf.frames();
                        for i in 0..frames {
                            let mut acc = 0.0f32;
                            for ch in 0..chans {
                                acc += buf.chan(ch)[i];
                            }
                            samples.push(acc / chans as f32);
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        debug!(frames=buf.frames(), "decoder: f64 buffer");
                        let spec = *buf.spec();
                        sample_rate = spec.rate;
                        let _ = spec.channels.count();
                        let chans = spec.channels.count();
                        let frames = buf.frames();
                        for i in 0..frames {
                            let mut acc = 0.0f32;
                            for ch in 0..chans {
                                acc += buf.chan(ch)[i] as f32;
                            }
                            samples.push(acc / chans as f32);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        debug!(frames=buf.frames(), "decoder: s16 buffer");
                        let spec = *buf.spec();
                        sample_rate = spec.rate;
                        let _ = spec.channels.count();
                        let chans = spec.channels.count();
                        let frames = buf.frames();
                        for i in 0..frames {
                            let mut acc = 0.0f32;
                            for ch in 0..chans {
                                acc += (buf.chan(ch)[i] as f32) / i16::MAX as f32;
                            }
                            samples.push(acc / chans as f32);
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        debug!(frames=buf.frames(), "decoder: u8 buffer");
                        let spec = *buf.spec();
                        sample_rate = spec.rate;
                        let _ = spec.channels.count();
                        let chans = spec.channels.count();
                        let frames = buf.frames();
                        for i in 0..frames {
                            let mut acc = 0.0f32;
                            for ch in 0..chans {
                                acc += (buf.chan(ch)[i] as f32 - 128.0) / 128.0;
                            }
                            samples.push(acc / chans as f32);
                        }
                    }
                    _ => {}
                }
            }
            Err(Error::DecodeError(_)) => {
                // Non-fatal; skip corrupt packet
                continue
            }
            Err(e) => return Err(e).context("decode packet")?,
        }
    }

    // Resample to 16 kHz mono for Whisper compatibility
    debug!(from=sample_rate, to=16_000, "decoder: resampling linear");
    let samples = resample_linear(&samples, sample_rate, 16_000);
    info!(frames=samples.len(), "decoder: done");
    Ok(PcmFrames {
        samples,
        sample_rate: 16_000,
        channels: 1,
    })
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return input.to_vec();
    }
    let step = src_rate as f64 / dst_rate as f64; // src samples per dst sample
    let n_out = ((input.len() as f64) / step).floor().max(0.0) as usize;
    let mut out = Vec::with_capacity(n_out);
    for j in 0..n_out {
        let pos = j as f64 * step;
        let i = pos.floor() as usize;
        let frac = (pos - i as f64) as f32;
        if i + 1 < input.len() {
            let a = input[i];
            let b = input[i + 1];
            out.push(a * (1.0 - frac) + b * frac);
        } else {
            out.push(input[i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_compiles() {
        // This just ensures the function can be called; no fixture yet.
        let _ = decode_to_pcm("/nonexistent").err();
    }

    fn write_wav_mono_i16(path: &std::path::Path, sr: u32, data: &[i16]) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        let byte_rate = sr * 2;
        let block_align = 2u16;
        let bits_per_sample = 16u16;
        let data_size = (data.len() * 2) as u32;
        let riff_size = 36 + data_size;
        // RIFF header
        f.write_all(b"RIFF").unwrap();
        f.write_all(&riff_size.to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        // fmt chunk
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap(); // PCM chunk size
        f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
        f.write_all(&1u16.to_le_bytes()).unwrap(); // channels
        f.write_all(&sr.to_le_bytes()).unwrap();
        f.write_all(&byte_rate.to_le_bytes()).unwrap();
        f.write_all(&block_align.to_le_bytes()).unwrap();
        f.write_all(&bits_per_sample.to_le_bytes()).unwrap();
        // data chunk
        f.write_all(b"data").unwrap();
        f.write_all(&data_size.to_le_bytes()).unwrap();
        // samples
        let mut buf = Vec::with_capacity(data.len() * 2);
        for &s in data {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        f.write_all(&buf).unwrap();
    }

    #[test]
    fn decode_wav_and_resample_to_16k() {
        // Generate a small 11_025 Hz mono sine wave WAV and ensure decoder returns 16k mono
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone.wav");
        let sr = 11_025u32;
        let dur_secs = 1.0f32 / 5.0; // 0.2s
        let n = (sr as f32 * dur_secs) as usize;
        let freq = 440.0f32;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let v = (2.0 * std::f32::consts::PI * freq * t).sin();
            data.push((v * 0.5 * i16::MAX as f32) as i16);
        }
        write_wav_mono_i16(&path, sr, &data);

        let pcm = decode_to_pcm(path.to_str().unwrap()).unwrap();
        assert_eq!(pcm.sample_rate, 16_000);
        assert_eq!(pcm.channels, 1);
        assert!(!pcm.samples.is_empty());
        // Length ratio approximately 16000 / 11025
        let expected = (n as f64 * (16_000.0 / sr as f64)) as usize;
        let diff = pcm.samples.len().abs_diff(expected);
        assert!(
            diff < 50,
            "len={} expected~{} diff={}",
            pcm.samples.len(),
            expected,
            diff
        );
    }
}
