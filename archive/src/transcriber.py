"""
transcriber.py

Wrapper for faster-whisper to transcribe audio files.
Optimized for Apple Silicon (M4) with CPU and Int8 quantization.
"""

from faster_whisper import WhisperModel


class MafiaTranscriber:
    """
    Mafia game audio transcriber using faster-whisper.

    Attributes:
        model: WhisperModel instance for transcription
    """

    def __init__(self, model_size: str = "medium", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the transcriber with the specified model configuration.

        Args:
            model_size: Whisper model size (e.g., "medium", "large-v3")
            device: Device to use for inference ("cpu" for Apple Silicon)
            compute_type: Quantization type ("int8" for optimized performance on M4)
        """
        print(f"Loading Whisper model: {model_size} (device={device}, compute_type={compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded successfully.")

    def transcribe(self, audio_path: str) -> list[dict]:
        """
        Transcribe an audio file to text with timestamps.

        Args:
            audio_path: Path to the audio file

        Returns:
            list[dict]: List of segment dictionaries with keys:
                - start: float, segment start time in seconds
                - end: float, segment end time in seconds
                - text: str, transcribed text
        """
        print(f"Starting transcription: {audio_path}")

        # Transcribe with Russian language and beam search
        segments, info = self.model.transcribe(
            audio_path,
            language="ru",
            beam_size=5,
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

        print(f"Transcription completed. Total segments: {len(results)}")
        return results
