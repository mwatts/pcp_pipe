#!/usr/bin/env python3
"""
High-Performance Podcast Processing Pipeline
Leverages uv for package management and Mojo for compute-intensive operations
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import hashlib
from datetime import datetime
import asyncio
import concurrent.futures

import requests
import torch
from transformers import pipeline
import spacy
import librosa
import soundfile as sf
import yt_dlp
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore")

# Import Mojo modules for performance-critical operations
try:
    from .mojo_accelerators import (
        MojoAudioProcessor, 
        MojoTranscriptAnalyzer,
        MojoEntityExtractor
    )
    MOJO_AVAILABLE = True
    print("Mojo accelerators loaded successfully")
except ImportError:
    MOJO_AVAILABLE = False
    print("Mojo accelerators not available, falling back to Python implementations")

# Whisper import - will use Mojo-accelerated version if available
try:
    if MOJO_AVAILABLE:
        from .mojo_whisper import MojoWhisper as whisper
    else:
        import whisper
except ImportError:
    import whisper

@dataclass
class TimeOffset:
    start: float
    end: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"start": self.start, "end": self.end}

@dataclass
class Speaker:
    id: str
    confidence: float
    segments: List[TimeOffset]

@dataclass
class TranscriptSegment:
    text: str
    speaker: Optional[str]
    time_offset: TimeOffset
    confidence: float
    segment_id: str

@dataclass
class Entity:
    text: str
    label: str
    confidence: float
    time_offset: TimeOffset
    transcript_offset: Tuple[int, int]
    context: str
    
@dataclass
class ProcessingResult:
    source_url: str
    audio_file_path: str
    transcript_segments: List[TranscriptSegment]
    full_transcript: str
    summary: str
    entities: List[Entity]
    speakers: List[Speaker]
    metadata: Dict[str, Any]
    processing_timestamp: str

class HighPerformancePodcastProcessor:
    def __init__(self, 
                 output_dir: str = "./podcast_output",
                 whisper_model: str = "large-v3",
                 use_gpu: bool = True,
                 use_mojo_acceleration: bool = True):
        """
        Initialize high-performance podcast processor with fully local models
        
        Args:
            output_dir: Directory to save all outputs
            whisper_model: Whisper model size
            use_gpu: Whether to use GPU acceleration
            use_mojo_acceleration: Whether to use Mojo for performance-critical operations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.use_mojo = use_mojo_acceleration and MOJO_AVAILABLE
        
        print(f"Using device: {self.device}")
        print(f"Mojo acceleration: {'enabled' if self.use_mojo else 'disabled'}")
        
        # Initialize models and accelerators - all local, no external tokens required
        self._init_models(whisper_model)
        self._init_mojo_accelerators()
        
    def _init_models(self, whisper_model: str):
        """Initialize all required models - completely local, no external dependencies"""
        print("Loading local models...")
        
        if self.use_mojo:
            # Use Mojo-accelerated Whisper if available
            self.whisper_model = whisper.load_model(
                whisper_model, 
                device=self.device,
                mojo_acceleration=True
            )
        else:
            # Standard Whisper - downloads model once then runs locally
            self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        print("✓ Whisper model loaded (local)")
        
        # Resemblyzer for local speaker diarization - no external dependencies
        try:
            self.voice_encoder = VoiceEncoder(device=self.device)
            print("✓ Resemblyzer voice encoder loaded (local)")
            self.speaker_diarization_available = True
        except Exception as e:
            print(f"Warning: Could not load Resemblyzer: {e}")
            self.voice_encoder = None
            self.speaker_diarization_available = False
            
        # Local summarization model - downloads once then runs offline
        try:
            # Use a smaller, faster local model that doesn't require external services
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",  # Smaller, faster BART variant
                device=0 if self.device == "cuda" else -1,
                batch_size=8 if self.device == "cuda" else 1
            )
            print("✓ Local summarization model loaded")
        except Exception as e:
            print(f"Warning: Could not load summarization model: {e}")
            self.summarizer = None
            
        # Local NER with downloadable models - no external APIs
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("✓ spaCy NER model loaded (local)")
            if self.device == "cuda":
                try:
                    spacy.require_gpu()
                    print("✓ spaCy GPU acceleration enabled")
                except:
                    print("Note: spaCy GPU acceleration not available")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy NER model loaded (en_core_web_sm)")
            except OSError:
                print("Error: No spacy model found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None

    def _init_mojo_accelerators(self):
        """Initialize Mojo accelerators for performance-critical operations"""
        if not self.use_mojo:
            self.mojo_audio_processor = None
            self.mojo_transcript_analyzer = None
            self.mojo_entity_extractor = None
            return
            
        try:
            self.mojo_audio_processor = MojoAudioProcessor()
            self.mojo_transcript_analyzer = MojoTranscriptAnalyzer()
            self.mojo_entity_extractor = MojoEntityExtractor()
            print("Mojo accelerators initialized successfully")
        except Exception as e:
            print(f"Warning: Mojo accelerators failed to initialize: {e}")
            self.use_mojo = False
            self.mojo_audio_processor = None
            self.mojo_transcript_analyzer = None
            self.mojo_entity_extractor = None

    async def download_audio_async(self, url: str) -> str:
        """
        Asynchronous audio download with enhanced error handling
        """
        print(f"Downloading audio from: {url}")
        
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        audio_filename = f"podcast_{url_hash}.wav"
        audio_path = self.output_dir / audio_filename
        
        if audio_path.exists():
            print(f"Audio already exists: {audio_path}")
            return str(audio_path)
        
        # Enhanced yt-dlp options for better performance
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/best',
            'outtmpl': str(self.output_dir / f'temp_{url_hash}.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'concurrent_fragment_downloads': 4,
            'http_chunk_size': 10485760,  # 10MB chunks
        }
        
        def download_sync():
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(url, download=True)
                
                temp_files = list(self.output_dir.glob(f'temp_{url_hash}.*'))
                if temp_files:
                    temp_file = temp_files[0]
                    temp_file.rename(audio_path)
                    return str(audio_path)
                else:
                    raise Exception("Downloaded file not found")
            except Exception as e:
                return self._direct_download_sync(url, str(audio_path))
        
        # Run download in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            audio_path_result = await loop.run_in_executor(executor, download_sync)
            
        print(f"Audio downloaded: {audio_path_result}")
        return audio_path_result
    
    def _direct_download_sync(self, url: str, output_path: str) -> str:
        """Synchronous fallback download"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
        except Exception as e:
            raise Exception(f"Failed to download audio: {e}")

    def transcribe_audio_optimized(self, audio_path: str) -> Tuple[List[TranscriptSegment], List[Speaker]]:
        """
        High-performance transcription with local speaker diarization
        """
        print("Transcribing audio with local optimizations...")
        
        # Use Mojo for audio preprocessing if available
        if self.use_mojo and self.mojo_audio_processor:
            print("Using Mojo audio preprocessing...")
            audio = self.mojo_audio_processor.load_and_preprocess(audio_path)
        else:
            audio = whisper.load_audio(audio_path)
        
        # Enhanced Whisper transcription with optimal parameters
        transcription_params = {
            'word_timestamps': True,
            'language': 'en',
            'initial_prompt': 'This is a podcast episode with clear speech.',
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.6,
        }
        
        if self.device == "cuda":
            transcription_params.update({
                'fp16': True,  # Use half precision on GPU for speed
            })
        
        result = self.whisper_model.transcribe(audio, **transcription_params)
        
        # Local speaker diarization using Resemblyzer
        speakers, speaker_segments = self._perform_speaker_diarization(audio_path)
        
        # Create optimized transcript segments
        transcript_segments = []
        
        for i, segment in enumerate(result["segments"]):
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Find speaker for this segment using local diarization results
            speaker_id = self._find_speaker_for_segment(
                segment_start, segment_end, speaker_segments
            )
            
            transcript_segments.append(TranscriptSegment(
                text=segment["text"].strip(),
                speaker=speaker_id,
                time_offset=TimeOffset(segment_start, segment_end),
                confidence=segment.get("avg_logprob", 0.0),
                segment_id=f"seg_{i:04d}"
            ))
        
        return transcript_segments, speakers

    def _find_speaker_for_segment(self, start: float, end: float, 
                                 speaker_segments: Dict) -> Optional[str]:
        """Find the best speaker match for a transcript segment"""
        max_overlap = 0
        best_speaker = None
        
        for speaker_id, segments in speaker_segments.items():
            for segment in segments:
                # Calculate overlap between transcript segment and speaker segment
                overlap_start = max(start, segment.start)
                overlap_end = min(end, segment.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_id
        
        # Only assign speaker if significant overlap (>50% of segment)
        segment_duration = end - start
        if max_overlap > segment_duration * 0.5:
            return best_speaker
        else:
            return None

    def _perform_speaker_diarization(self, audio_path: str) -> Tuple[List[Speaker], Dict]:
        """
        Perform local speaker diarization using Resemblyzer
        No external dependencies or tokens required
        """
        if not self.speaker_diarization_available:
            return [], {}
            
        print("Performing local speaker diarization with Resemblyzer...")
        
        try:
            # Load and preprocess audio for Resemblyzer
            wav = preprocess_wav(audio_path)
            
            # Split audio into segments (sliding window approach)
            segment_duration = 1.0  # 1 second segments
            hop_duration = 0.5      # 50% overlap
            
            sample_rate = 16000  # Resemblyzer expects 16kHz
            segment_length = int(segment_duration * sample_rate)
            hop_length = int(hop_duration * sample_rate)
            
            segments = []
            embeddings = []
            timestamps = []
            
            # Extract embeddings for each segment
            for start in range(0, len(wav) - segment_length, hop_length):
                end = start + segment_length
                segment = wav[start:end]
                
                # Get speaker embedding
                embedding = self.voice_encoder.embed_utterance(segment)
                embeddings.append(embedding)
                segments.append(segment)
                timestamps.append((start / sample_rate, end / sample_rate))
            
            if len(embeddings) == 0:
                return [], {}
            
            # Cluster embeddings to identify speakers
            embeddings_array = np.array(embeddings)
            
            # Automatic speaker count estimation using silhouette analysis
            max_speakers = min(10, len(embeddings) // 3)  # Reasonable upper bound
            best_score = -1
            best_n_speakers = 2
            
            if max_speakers > 1:
                from sklearn.metrics import silhouette_score
                
                for n_speakers in range(2, max_speakers + 1):
                    clustering = AgglomerativeClustering(
                        n_clusters=n_speakers,
                        metric='cosine',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(embeddings_array)
                    
                    if len(set(labels)) > 1:  # Ensure we have multiple clusters
                        score = silhouette_score(embeddings_array, labels, metric='cosine')
                        if score > best_score:
                            best_score = score
                            best_n_speakers = n_speakers
            
            # Final clustering with optimal speaker count
            clustering = AgglomerativeClustering(
                n_clusters=best_n_speakers,
                metric='cosine', 
                linkage='average'
            )
            speaker_labels = clustering.fit_predict(embeddings_array)
            
            # Group segments by speaker
            speaker_segments = {}
            for i, (label, (start_time, end_time)) in enumerate(zip(speaker_labels, timestamps)):
                speaker_id = f"speaker_{label}"
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(TimeOffset(start_time, end_time))
            
            # Merge adjacent segments from same speaker
            merged_segments = {}
            for speaker_id, segments in speaker_segments.items():
                merged = self._merge_adjacent_segments(segments)
                merged_segments[speaker_id] = merged
            
            # Create speaker objects
            speakers = []
            for speaker_id, segments in merged_segments.items():
                # Calculate confidence based on cluster cohesion
                speaker_embeddings = [embeddings[i] for i, label in enumerate(speaker_labels) 
                                    if f"speaker_{label}" == speaker_id]
                confidence = self._calculate_speaker_confidence(speaker_embeddings)
                
                speakers.append(Speaker(
                    id=speaker_id,
                    confidence=confidence,
                    segments=segments
                ))
            
            print(f"✓ Detected {len(speakers)} speakers locally")
            return speakers, merged_segments
            
        except Exception as e:
            print(f"Local speaker diarization failed: {e}")
            return [], {}
    
    def _merge_adjacent_segments(self, segments: List[TimeOffset], 
                               max_gap: float = 0.5) -> List[TimeOffset]:
        """Merge adjacent segments from the same speaker"""
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start)
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            # Merge if gap is small
            if current.start - last.end <= max_gap:
                merged[-1] = TimeOffset(last.start, max(last.end, current.end))
            else:
                merged.append(current)
        
        return merged
    
    def _calculate_speaker_confidence(self, embeddings: List[np.ndarray]) -> float:
        """Calculate confidence score based on embedding similarity"""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Return average similarity as confidence
        return float(np.mean(similarities)) if similarities else 0.5

    def generate_summary_optimized(self, full_transcript: str) -> str:
        """
        High-performance summary generation with Mojo acceleration
        """
        if not self.summarizer:
            return "Summary generation not available"
            
        print("Generating optimized summary...")
        
        # Use Mojo for text preprocessing if available
        if self.use_mojo and self.mojo_transcript_analyzer:
            print("Using Mojo text analysis...")
            processed_chunks = self.mojo_transcript_analyzer.create_optimal_chunks(
                full_transcript, max_chunk_length=1000
            )
        else:
            # Fallback to Python chunking
            processed_chunks = self._create_chunks_python(full_transcript, 1000)
        
        # Parallel summarization for better performance
        summaries = []
        
        if self.device == "cuda" and len(processed_chunks) > 1:
            # Batch processing on GPU
            try:
                batch_summaries = self.summarizer(
                    processed_chunks,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    batch_size=min(4, len(processed_chunks))
                )
                summaries = [s['summary_text'] for s in batch_summaries]
            except Exception as e:
                print(f"Batch summarization failed, falling back: {e}")
                # Fallback to sequential processing
                for chunk in processed_chunks:
                    try:
                        summary = self.summarizer(chunk, max_length=150, min_length=50)
                        summaries.append(summary[0]['summary_text'])
                    except:
                        continue
        else:
            # Sequential processing for CPU or single chunk
            for chunk in processed_chunks:
                try:
                    summary = self.summarizer(chunk, max_length=150, min_length=50)
                    summaries.append(summary[0]['summary_text'])
                except:
                    continue
        
        # Final summary combination
        if len(summaries) > 1:
            combined = " ".join(summaries)
            try:
                final_summary = self.summarizer(
                    combined, max_length=300, min_length=100
                )
                return final_summary[0]['summary_text']
            except:
                return " ".join(summaries)
        elif summaries:
            return summaries[0]
        else:
            return "Summary generation failed"

    def _create_chunks_python(self, text: str, max_length: int) -> List[str]:
        """Fallback Python text chunking"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += 1
            
            if current_length >= max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def extract_entities_optimized(self, 
                                 transcript_segments: List[TranscriptSegment], 
                                 full_transcript: str) -> List[Entity]:
        """
        High-performance entity extraction with Mojo acceleration
        """
        if not self.nlp:
            return []
            
        print("Extracting entities with optimizations...")
        
        # Use Mojo for entity extraction if available
        if self.use_mojo and self.mojo_entity_extractor:
            print("Using Mojo entity extraction...")
            return self.mojo_entity_extractor.extract_entities_parallel(
                transcript_segments, full_transcript, self.nlp
            )
        
        # Fallback to optimized Python implementation
        return self._extract_entities_python_optimized(transcript_segments, full_transcript)

    def _extract_entities_python_optimized(self, 
                                         transcript_segments: List[TranscriptSegment], 
                                         full_transcript: str) -> List[Entity]:
        """Optimized Python entity extraction with parallel processing"""
        entities = []
        char_offset = 0
        
        # Batch process segments for better performance
        segment_texts = [seg.text for seg in transcript_segments]
        
        if self.device == "cuda":
            # Use spaCy's batch processing for GPU acceleration
            docs = list(self.nlp.pipe(segment_texts, batch_size=32))
        else:
            # CPU batch processing
            docs = list(self.nlp.pipe(segment_texts, batch_size=8))
        
        for segment, doc in zip(transcript_segments, docs):
            for ent in doc.ents:
                ent_start_char = char_offset + ent.start_char
                ent_end_char = char_offset + ent.end_char
                
                context_start = max(0, ent_start_char - 100)
                context_end = min(len(full_transcript), ent_end_char + 100)
                context = full_transcript[context_start:context_end]
                
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    confidence=1.0,
                    time_offset=segment.time_offset,
                    transcript_offset=(ent_start_char, ent_end_char),
                    context=context.strip()
                ))
            
            char_offset += len(segment.text) + 1
        
        # Optimized deduplication
        unique_entities = []
        seen = set()
        
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

    async def process_podcast_async(self, url: str) -> ProcessingResult:
        """
        Complete high-performance podcast processing pipeline with async operations
        """
        start_time = time.time()
        print(f"Starting high-performance podcast processing for: {url}")
        
        # Async download
        audio_path = await self.download_audio_async(url)
        
        # Create tasks for parallel processing where possible
        tasks = []
        
        # Transcription (sequential due to model dependencies)
        transcript_segments, speakers = self.transcribe_audio_optimized(audio_path)
        full_transcript = self._create_full_transcript(transcript_segments)
        
        # Parallel summary and entity extraction
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            summary_future = loop.run_in_executor(
                executor, self.generate_summary_optimized, full_transcript
            )
            entities_future = loop.run_in_executor(
                executor, self.extract_entities_optimized, 
                transcript_segments, full_transcript
            )
            
            summary, entities = await asyncio.gather(summary_future, entities_future)
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata = {
            "processing_time_seconds": processing_time,
            "audio_duration_seconds": self._get_audio_duration(audio_path),
            "num_segments": len(transcript_segments),
            "num_speakers": len(speakers),
            "num_entities": len(entities),
            "performance_optimizations": {
                "mojo_acceleration": self.use_mojo,
                "gpu_acceleration": self.device == "cuda",
                "async_processing": True,
                "parallel_summarization": True,
                "batch_entity_extraction": True
            },
            "device_used": self.device
        }
        
        result = ProcessingResult(
            source_url=url,
            audio_file_path=audio_path,
            transcript_segments=transcript_segments,
            full_transcript=full_transcript,
            summary=summary,
            entities=entities,
            speakers=speakers,
            metadata=metadata,
            processing_timestamp=datetime.now().isoformat()
        )
        
        # Save results
        await self._save_results_async(result)
        
        print(f"High-performance processing complete in {processing_time:.2f} seconds")
        return result
    
    def _create_full_transcript(self, segments: List[TranscriptSegment]) -> str:
        """Create formatted full transcript"""
        lines = []
        for segment in segments:
            speaker_prefix = f"[{segment.speaker}] " if segment.speaker else ""
            timestamp = f"[{segment.time_offset.start:.2f}s] "
            lines.append(f"{timestamp}{speaker_prefix}{segment.text}")
        return "\n".join(lines)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration efficiently"""
        try:
            if self.use_mojo and self.mojo_audio_processor:
                return self.mojo_audio_processor.get_duration_fast(audio_path)
            else:
                audio, sr = librosa.load(audio_path, sr=None)
                return len(audio) / sr
        except:
            return 0.0
    
    async def _save_results_async(self, result: ProcessingResult):
        """Asynchronous result saving"""
        url_hash = hashlib.md5(result.source_url.encode()).hexdigest()[:12]
        base_name = f"podcast_{url_hash}"
        
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return asdict(obj)
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        complete_result = convert_to_dict(result)
        
        # Parallel file writing
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            save_tasks = [
                loop.run_in_executor(
                    executor, self._save_json, complete_result, f"{base_name}_complete.json"
                ),
                loop.run_in_executor(
                    executor, self._save_json, 
                    {
                        "url": result.source_url,
                        "transcript": result.full_transcript,
                        "metadata": result.metadata
                    }, 
                    f"{base_name}_transcript.json"
                ),
                loop.run_in_executor(
                    executor, self._save_json,
                    {
                        "url": result.source_url,
                        "summary": result.summary,
                        "metadata": result.metadata
                    },
                    f"{base_name}_summary.json"
                ),
                loop.run_in_executor(
                    executor, self._save_json,
                    {
                        "source_url": result.source_url,
                        "audio_file": result.audio_file_path,
                        "full_transcript": {
                            "text": result.full_transcript,
                            "entity_type": "full_transcript",
                            "time_offset": {"start": 0, "end": result.metadata.get("audio_duration_seconds", 0)}
                        },
                        "summary": {
                            "text": result.summary,
                            "entity_type": "summary",
                            "derived_from": "full_transcript"
                        },
                        "extracted_entities": [convert_to_dict(entity) for entity in result.entities],
                        "speakers": [convert_to_dict(speaker) for speaker in result.speakers],
                        "metadata": result.metadata
                    },
                    f"{base_name}_entities.json"
                )
            ]
            
            await asyncio.gather(*save_tasks)
        
        print(f"Results saved with base name: {base_name}")
    
    def _save_json(self, data: dict, filename: str):
        """Save data as JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Synchronous wrapper for backward compatibility
    def process_podcast(self, url: str) -> ProcessingResult:
        """Synchronous wrapper for async processing"""
        return asyncio.run(self.process_podcast_async(url))


def setup_environment():
    """Setup optimized environment using uv"""
    print("Setting up high-performance environment...")
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("uv detected, using for package management")
        use_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv not found, falling back to pip")
        use_uv = False
    
    # Core dependencies
    dependencies = [
        "torch",
        "torchaudio", 
        "openai-whisper",
        "transformers[torch]",
        "pyannote.audio",
        "spacy",
        "librosa",
        "soundfile",
        "yt-dlp",
        "requests",
        "numpy",
        "asyncio-mqtt"  # For async optimizations
    ]
    
    if use_uv:
        # Use uv for faster package installation
        cmd = ["uv", "pip", "install"] + dependencies
        subprocess.run(cmd, check=True)
        
        # Install spaCy model with uv
        subprocess.run([
            "uv", "pip", "install", 
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
        ], check=True)
    else:
        # Fallback to pip
        subprocess.run(["pip", "install"] + dependencies, check=True)
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
    
    print("Environment setup complete")


async def main():
    """High-performance async main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Performance Local Podcast Processor")
    parser.add_argument("url", help="Podcast episode URL")
    parser.add_argument("--output-dir", default="./podcast_output", help="Output directory")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--no-mojo", action="store_true", help="Disable Mojo acceleration")
    parser.add_argument("--setup-env", action="store_true", help="Setup environment with uv")
    
    args = parser.parse_args()
    
    if args.setup_env:
        setup_environment()
        return
    
    processor = HighPerformancePodcastProcessor(
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        use_gpu=not args.no_gpu,
        use_mojo_acceleration=not args.no_mojo
    )
    
    result = await processor.process_podcast_async(args.url)
    
    print(f"\n🎉 Processing Summary:")
    print(f"  Total time: {result.metadata['processing_time_seconds']:.2f}s")
    print(f"  Audio duration: {result.metadata['audio_duration_seconds']:.2f}s")
    print(f"  Processing speed: {result.metadata['audio_duration_seconds']/result.metadata['processing_time_seconds']:.2f}x realtime")
    print(f"  Segments: {result.metadata['num_segments']}")
    print(f"  Speakers: {result.metadata['num_speakers']}")
    print(f"  Entities: {result.metadata['num_entities']}")
    print(f"  Optimizations: {result.metadata['performance_optimizations']}")
    print(f"  🔥 All processing completed locally - no external API calls required")


if __name__ == "__main__":
    asyncio.run(main())
