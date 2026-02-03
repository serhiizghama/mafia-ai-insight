"""
transcriber.py

Hybrid transcriber with pluggable backends (Faster-Whisper and MLX-Whisper).
Supports both CPU and GPU-accelerated transcription.
"""

import sys
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTranscriber(ABC):
    """
    Abstract base class for transcription backends.
    All backends must implement the transcribe method with consistent output format.
    """

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, any]]:
        """
        Transcribe an audio file to text with timestamps.

        Args:
            audio_path: Path to the audio file

        Returns:
            List[Dict]: List of segment dictionaries with keys:
                - start: float, segment start time in seconds
                - end: float, segment end time in seconds
                - text: str, transcribed text
        """
        pass


class FasterWhisperBackend(BaseTranscriber):
    """
    CPU-optimized transcription backend using faster-whisper.
    Works on any platform (macOS, Linux, Windows).
    """

    def __init__(self, model_size: str = "medium", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the Faster-Whisper backend.

        Args:
            model_size: Whisper model size (e.g., "medium", "large-v3")
            device: Device to use for inference ("cpu" for cross-platform compatibility)
            compute_type: Quantization type ("int8" for optimized performance)
        """
        from faster_whisper import WhisperModel

        print(f"[Faster-Whisper Backend] Loading model: {model_size} (device={device}, compute_type={compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[Faster-Whisper Backend] Model loaded successfully.")

    def transcribe(self, audio_path: str) -> List[Dict[str, any]]:
        """
        Transcribe audio using Faster-Whisper (CPU-optimized).

        Args:
            audio_path: Path to the audio file

        Returns:
            List[Dict]: List of segments with start, end, text
        """
        print(f"[Faster-Whisper Backend] Starting transcription: {audio_path}")

        # Mafia-specific context for better recognition of game terminology
        initial_prompt = (
            "Игра Мафия. Ведущий объявляет: город засыпает, просыпается Дон, Шериф, Мафия. "
            "Голосование, фол, перестрелка, автокатастрофа, попил стола, победа города, победа мафии, Левша, 22."
        )

        # Transcribe with Russian language, beam search, and quality improvements
        segments, info = self.model.transcribe(
            audio_path,
            language="ru",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            repetition_penalty=1.2,
            initial_prompt=initial_prompt,
        )

        # Log detected language information
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # Process segments
        results = []
        for segment in segments:
            # Print progress
            print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text.strip()}")

            # Append to results
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })

        print(f"[Faster-Whisper Backend] Transcription completed. Total segments: {len(results)}")
        return results


class MlxWhisperBackend(BaseTranscriber):
    """
    GPU-optimized transcription backend using MLX-Whisper.
    Only works on Apple Silicon (Mac M1/M2/M3/M4).
    """

    def __init__(self, model_size: str = "medium"):
        """
        Initialize the MLX-Whisper backend.

        Args:
            model_size: Whisper model size (e.g., "medium", "large-v3", "small")
        """
        # Check if running on macOS
        if sys.platform != "darwin":
            raise RuntimeError(
                "MLX backend is only supported on macOS with Apple Silicon. "
                f"Current platform: {sys.platform}. "
                "Please set TRANSCRIPTION_BACKEND=faster_whisper in your .env file."
            )

        # Lazy import to avoid import errors on non-macOS systems
        try:
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is not installed. Install it with: pip install mlx-whisper"
            ) from e

        # MLX community model naming: mlx-community/whisper-{size}-mlx
        self.model_path = f"mlx-community/whisper-{model_size}-mlx"
        self.model_size = model_size

        print(f"[MLX Backend] Initialized with model: {self.model_path} (GPU Mode)")

    def transcribe(self, audio_path: str) -> List[Dict[str, any]]:
        """
        Transcribe audio using MLX-Whisper (GPU-accelerated on Apple Silicon).

        Args:
            audio_path: Path to the audio file

        Returns:
            List[Dict]: List of segments with start, end, text
        """
        print(f"[MLX Backend] Starting transcription: {audio_path}")

        # Mafia-specific context for better recognition of game terminology
        initial_prompt = (
            "Игра Мафия. Ведущий объявляет: город засыпает, просыпается Дон, Шериф, Мафия. "
            "Голосование, фол, перестрелка, автокатастрофа, попил стола, победа города, победа мафии, Левша, 22."
        )

        # Transcribe using MLX with quality settings
        result = self.mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=self.model_path,
            language="ru",
            initial_prompt=initial_prompt,
        )

        # Extract segments from MLX result
        # MLX output format: {"segments": [...], "text": "..."}
        segments = result.get("segments", [])

        # Map to our standard format
        results = []
        for segment in segments:
            # Print progress
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            text = segment.get("text", "").strip()

            print(f"[{start:.1f}s -> {end:.1f}s] {text}")

            # Append to results
            results.append({
                "start": start,
                "end": end,
                "text": text
            })

        print(f"[MLX Backend] Transcription completed. Total segments: {len(results)}")
        return results


def get_transcriber(backend_type: str, model_size: str = "medium") -> BaseTranscriber:
    """
    Factory function to create the appropriate transcriber backend.

    Args:
        backend_type: Backend type ("mlx" or "faster_whisper")
        model_size: Whisper model size (e.g., "medium", "large-v3", "small")

    Returns:
        BaseTranscriber: Configured transcriber instance

    Raises:
        ValueError: If backend_type is not recognized
        RuntimeError: If MLX is selected on non-macOS platform
        ImportError: If required backend library is not installed
    """
    backend_type = backend_type.lower().strip()

    if backend_type == "faster_whisper":
        print("Initializing Faster-Whisper Backend (CPU Mode)")
        return FasterWhisperBackend(model_size=model_size, device="cpu", compute_type="int8")

    elif backend_type == "mlx":
        print("Initializing MLX Backend (GPU Mode)")
        return MlxWhisperBackend(model_size=model_size)

    else:
        raise ValueError(
            f"Unknown transcription backend: '{backend_type}'. "
            f"Supported backends: 'mlx' (Apple Silicon GPU) or 'faster_whisper' (CPU, any OS)"
        )


# Backward compatibility: keep MafiaTranscriber as alias to FasterWhisperBackend
# This allows existing code to continue working without changes
MafiaTranscriber = FasterWhisperBackend
