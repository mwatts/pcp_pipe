//! Decoder: placeholder crate for symphonia-based decoding to PCM.

pub struct PcmFrames;

pub fn decode_to_pcm(_path: &str) -> anyhow::Result<PcmFrames> {
    // Stub to keep workspace building. Implementation deferred to M2.
    Ok(PcmFrames)
}
