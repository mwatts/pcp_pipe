use anyhow::{anyhow, Result};
use tracing::instrument;
use objc2_speech::{SFSpeechRecognizer, SFSpeechRecognizerAuthorizationStatus};



#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Transcript {
    pub text: String,
}

/// Options for native macOS transcription via Speech framework.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NativeTranscribeOptions {
    /// BCP-47/ISO locale identifier, e.g., "en-US". If None, system default is used.
    pub locale: Option<String>,
    /// Prefer on-device recognition when available.
    pub on_device: bool,
    /// Report partial results while recognition is in progress (not currently surfaced).
    pub report_partials: bool,
}

/// Transcribe an audio file directly using macOS Speech (SFSpeechRecognizer).
///
/// Notes:
/// - Requires microphone/speech recognition permission (TCC). The first run will
///   prompt for access in a GUI session. For headless runs, pre-authorize as needed.
#[instrument(skip(opts))]
pub async fn transcribe_file_async(path: &std::path::Path, opts: &NativeTranscribeOptions) -> Result<Transcript> {
    if !path.exists() {
        return Err(anyhow!("input file not found: {}", path.display()));
    }

    tracing::info!("macOS native transcription requested for: {}", path.display());
    tracing::info!("Options: locale={:?}, on_device={}, report_partials={}",
                  opts.locale, opts.on_device, opts.report_partials);

    // Step 1: Check authorization status
    let auth_status = unsafe { SFSpeechRecognizer::authorizationStatus() };
    tracing::info!("Speech recognition authorization status: {:?}", auth_status);

    match auth_status {
        SFSpeechRecognizerAuthorizationStatus::NotDetermined => {
            tracing::info!("Speech recognition authorization not determined - requesting permission...");
            
            // For now, provide helpful instructions to the user
            return Err(anyhow!(
                "Speech recognition permission needed. Please:\n\
                1. Open System Preferences > Privacy & Security > Speech Recognition\n\
                2. Add this application to the allowed list\n\
                3. Or run this command from a GUI application that can request permission\n\
                \n\
                Current status: Not Determined"
            ));
        }
        SFSpeechRecognizerAuthorizationStatus::Denied => {
            return Err(anyhow!(
                "Speech recognition permission denied. Please enable it in:\n\
                System Preferences > Privacy & Security > Speech Recognition"
            ));
        }
        SFSpeechRecognizerAuthorizationStatus::Restricted => {
            return Err(anyhow!(
                "Speech recognition is restricted on this device (parental controls or device management)."
            ));
        }
        SFSpeechRecognizerAuthorizationStatus::Authorized => {
            tracing::info!("Speech recognition is authorized, proceeding...");
        }
        _ => {
            return Err(anyhow!(
                "Unknown speech recognition authorization status: {:?}", auth_status
            ));
        }
    }

    // Implementation plan:
    // 1. ✅ Check authorization status  
    // 2. ⚠️  Create SFSpeechRecognizer with locale
    // 3. TODO: Create NSURL from file path
    // 4. TODO: Create SFSpeechURLRecognitionRequest
    // 5. TODO: Configure request options
    // 6. TODO: Perform recognition and wait for results
    
    // Step 2: Create SFSpeechRecognizer  
    tracing::info!("Creating SFSpeechRecognizer with default locale (locale customization coming next)");
    let recognizer = unsafe { SFSpeechRecognizer::new() };
    
    // TODO: Add locale support:
    // if let Some(locale_str) = &opts.locale {
    //     tracing::info!("Will create SFSpeechRecognizer with locale: {}", locale_str);
    // }

    // Check if recognizer was created successfully
    tracing::info!("SFSpeechRecognizer created successfully");
    
    // Check if recognizer is available
    let is_available = unsafe { recognizer.isAvailable() };
    if !is_available {
        return Err(anyhow!("SFSpeechRecognizer is not available on this device"));
    }
    tracing::info!("SFSpeechRecognizer is available");
    
    // For now, return enhanced placeholder that shows recognizer creation is working
    tracing::warn!("Using enhanced placeholder implementation - NSURL creation coming next");
    
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    
    let locale_info = opts.locale
        .as_ref()
        .map(|l| format!(" (locale: {})", l))
        .unwrap_or_default();
    
    let device_info = if opts.on_device { " [on-device]" } else { " [cloud]" };
    
    Ok(Transcript {
        text: format!("✅ Auth+Recognizer OK: {}{}{}", file_name, locale_info, device_info)
    })
}


#[cfg(test)]
mod tests {
    // Test module for future unit tests
}
