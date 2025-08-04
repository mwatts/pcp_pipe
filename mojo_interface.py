"""
Mojo Integration Interface for Podcast Processing Pipeline

This module provides a Python interface to Mojo accelerators.
Currently serves as a placeholder until Mojo Python packaging is available.

To enable Mojo acceleration:
1. Install Modular CLI: curl -s https://get.modular.com | sh -
2. Install Mojo: modular install mojo
3. Add to PATH: export PATH=$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH
4. Compile accelerators: ./build_mojo.sh
5. Wait for Mojo Python packaging features (in development)
6. Update imports when packaging becomes available

Current Status:
- Mojo source code is ready and syntactically correct
- Python fallback implementations provide full functionality
- Mojo acceleration will be enabled when packaging is ready
"""

import os
import sys
from typing import Optional, List, Any, Dict

class MojoInterface:
    """Interface for Mojo accelerator integration"""
    
    def __init__(self):
        self.available = False
        self.audio_processor = None
        self.transcript_analyzer = None
        self.entity_extractor = None
        
    def check_availability(self) -> bool:
        """Check if Mojo accelerators are available"""
        # Check for Mojo compiler
        mojo_compiler_available = False
        try:
            import subprocess
            result = subprocess.run(['mojo', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                mojo_compiler_available = True
                print(f"Mojo compiler found: {result.stdout.strip()}")
        except FileNotFoundError:
            pass
            
        # Check for Mojo source
        mojo_source = os.path.join(os.path.dirname(__file__), 'mojo_accelerators.mojo')
        if os.path.exists(mojo_source):
            print(f"Mojo source found at: {mojo_source}")
            if mojo_compiler_available:
                print("To build: ./build_mojo.sh")
                print("Note: Python packaging for Mojo is in development")
            else:
                print("Install Mojo to enable acceleration")
            return False
        return False
    
    def initialize_accelerators(self):
        """Initialize Mojo accelerators if available"""
        if not self.check_availability():
            return False
            
        try:
            # This would import compiled Mojo modules
            # from mojo_accelerators_compiled import (
            #     MojoAudioProcessor,
            #     MojoTranscriptAnalyzer, 
            #     MojoEntityExtractor
            # )
            # 
            # self.audio_processor = MojoAudioProcessor()
            # self.transcript_analyzer = MojoTranscriptAnalyzer()
            # self.entity_extractor = MojoEntityExtractor()
            # self.available = True
            # return True
            pass
        except ImportError as e:
            print(f"Failed to import Mojo accelerators: {e}")
            return False
            
        return False

# Fallback implementations (Python-only)
class PythonAudioProcessor:
    """Python fallback for MojoAudioProcessor"""
    
    def __init__(self):
        print("Using Python audio processing (no Mojo acceleration)")
    
    def load_and_preprocess(self, audio_path: str):
        """Fallback audio preprocessing"""
        import whisper
        return whisper.load_audio(audio_path)
    
    def get_duration_fast(self, audio_path: str) -> float:
        """Fallback duration calculation"""
        try:
            import librosa
            return float(librosa.get_duration(path=audio_path))
        except:
            return 0.0

class PythonTranscriptAnalyzer:
    """Python fallback for MojoTranscriptAnalyzer"""
    
    def __init__(self):
        print("Using Python text analysis (no Mojo acceleration)")
    
    def create_optimal_chunks(self, transcript: str, max_chunk_length: int) -> List[str]:
        """Fallback text chunking"""
        words = transcript.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += 1
            
            if current_length >= max_chunk_length:
                # Check for sentence boundaries
                if word.endswith(('.', '!', '?')):
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                elif current_length >= max_chunk_length * 1.2:  # Hard limit
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class PythonEntityExtractor:
    """Python fallback for MojoEntityExtractor"""
    
    def __init__(self):
        print("Using Python entity extraction (no Mojo acceleration)")
    
    def extract_entities_parallel(self, transcript_segments, full_transcript: str, nlp_model) -> List[Dict]:
        """Fallback entity extraction"""
        entities = []
        char_offset = 0
        
        # Batch process segments
        segment_texts = [seg.text for seg in transcript_segments]
        docs = list(nlp_model.pipe(segment_texts, batch_size=8))
        
        for segment, doc in zip(transcript_segments, docs):
            for ent in doc.ents:
                ent_start_char = char_offset + ent.start_char
                ent_end_char = char_offset + ent.end_char
                
                context_start = max(0, ent_start_char - 100)
                context_end = min(len(full_transcript), ent_end_char + 100)
                context = full_transcript[context_start:context_end]
                
                entity_dict = {
                    "text": ent.text,
                    "label": ent.label_,
                    "confidence": 1.0,
                    "time_offset": segment.time_offset,
                    "transcript_offset": (ent_start_char, ent_end_char),
                    "context": context.strip()
                }
                entities.append(entity_dict)
            
            char_offset += len(segment.text) + 1
        
        # Deduplicate
        unique_entities = []
        seen = set()
        
        for entity in entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

# Global interface instance
mojo_interface = MojoInterface()

def get_accelerators():
    """Get available accelerators (Mojo if available, Python fallbacks otherwise)"""
    if mojo_interface.initialize_accelerators():
        return (
            mojo_interface.audio_processor,
            mojo_interface.transcript_analyzer,
            mojo_interface.entity_extractor
        )
    else:
        return (
            PythonAudioProcessor(),
            PythonTranscriptAnalyzer(),
            PythonEntityExtractor()
        )