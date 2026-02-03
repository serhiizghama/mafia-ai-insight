"""
main.py

Entry point for the archive module (Stage 1).
Batch processor: reads URLs from urls.txt, transcribes, and saves to JSON.
Fully automated - no interactive prompts.
"""

import json
import os
import time
from datetime import datetime
from typing import List

from dotenv import load_dotenv

from src.downloader import download_audio
from src.transcriber import get_transcriber

# Get the base directory of the archive package (always points to archive/ folder)
ARCHIVE_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(ARCHIVE_BASE_DIR, "audio")
DATA_DIR = os.path.join(ARCHIVE_BASE_DIR, "data")
URLS_FILE = os.path.join(ARCHIVE_BASE_DIR, "urls.txt")


def load_urls() -> List[str]:
    """
    Load YouTube URLs from urls.txt.

    Returns:
        List[str]: List of valid YouTube URLs (non-empty, non-comment lines)
    """
    if not os.path.exists(URLS_FILE):
        return []

    urls = []
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                urls.append(line)

    return urls


def save_result(data: dict, filename: str = "games_archive.json") -> None:
    """
    Append game data to the JSON archive file.

    Args:
        data: Game data dictionary to append
        filename: Filename (not path) for the JSON archive file in archive/data/
    """
    # Construct full path relative to archive/data/
    filepath = os.path.join(DATA_DIR, filename)

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load existing archive or initialize empty list
    archive = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                archive = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}. Starting fresh archive.")
            archive = []

    # Append new game data
    archive.append(data)

    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(archive, f, ensure_ascii=False, indent=4)

    print(f"✓ Game data saved to: {filepath}")


def process_video(url: str, transcriber, index: int, total: int) -> bool:
    """
    Process a single YouTube video: download, transcribe, and save.

    Args:
        url: YouTube video URL
        transcriber: Transcriber instance to use
        index: Current video index (1-based)
        total: Total number of videos in queue

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    print()
    print("=" * 80)
    print(f"Processing {index}/{total}: {url}")
    print("=" * 80)

    # Step 1: Download audio
    print()
    print(f"[{index}/{total}] STEP 1: Downloading audio...")
    audio_path, video_title = download_audio(url, output_dir=AUDIO_DIR)

    if audio_path is None:
        print(f"✗ Failed to download audio from: {url}")
        return False

    print(f"✓ Download complete: {video_title}")

    # Step 2: Transcribe
    print()
    print(f"[{index}/{total}] STEP 2: Transcribing audio...")
    try:
        transcript_segments = transcriber.transcribe(audio_path)
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        # Clean up audio even on failure
        try:
            os.remove(audio_path)
        except Exception:
            pass
        return False

    print(f"✓ Transcription complete: {len(transcript_segments)} segments")

    # Step 3: Save results
    print()
    print(f"[{index}/{total}] STEP 3: Saving results...")

    # Auto-generate Game ID from timestamp
    game_id = f"game_{int(time.time())}"
    processed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct game data
    game_data = {
        "id": game_id,
        "title": video_title,
        "url": url,
        "meta": {
            "mafia_roles": None,  # To be determined by LLM
            "winner": None,       # To be determined by LLM
            "processed_date": processed_date
        },
        "transcript": transcript_segments
    }

    # Save to JSON archive
    save_result(game_data)

    # Step 4: Cleanup
    print()
    print(f"[{index}/{total}] STEP 4: Cleaning up...")
    try:
        os.remove(audio_path)
        print(f"✓ Deleted temporary audio file: {audio_path}")
    except Exception as e:
        print(f"⚠ Could not delete audio file: {e}")

    print()
    print("=" * 80)
    print(f"✓ Video {index}/{total} processed successfully!")
    print("=" * 80)

    return True


def main():
    """
    Main batch processing workflow.
    Reads URLs from urls.txt and processes each video sequentially.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Read transcription configuration
    backend_type = os.getenv("TRANSCRIPTION_BACKEND", "faster_whisper")
    model_size = os.getenv("WHISPER_MODEL_SIZE", "medium")

    # Display banner
    print()
    print("=" * 80)
    print("MAFIA AI INSIGHT - ARCHIVE MODULE v1.0 (Batch Processor)")
    print("Digital Archive: YouTube -> Audio -> Transcription -> JSON")
    print("=" * 80)
    print(f"Backend: {backend_type} | Model: {model_size}")
    print("=" * 80)
    print()

    # Load URLs from file
    urls = load_urls()

    if not urls:
        print(f"⚠ No URLs found in: {URLS_FILE}")
        print()
        print("Instructions:")
        print(f"1. Add YouTube URLs to: {URLS_FILE}")
        print("2. One URL per line")
        print("3. Lines starting with # are treated as comments")
        print()
        return

    print(f"✓ Loaded {len(urls)} URL(s) from {URLS_FILE}")
    print()

    # Initialize transcriber once (reuse for all videos)
    print("Initializing transcription backend...")
    try:
        transcriber = get_transcriber(backend_type=backend_type, model_size=model_size)
    except Exception as e:
        print(f"✗ Failed to initialize transcriber: {e}")
        return

    print()

    # Process each URL
    success_count = 0
    failure_count = 0

    for i, url in enumerate(urls, start=1):
        success = process_video(url, transcriber, i, len(urls))
        if success:
            success_count += 1
        else:
            failure_count += 1

    # Final summary
    print()
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Total videos: {len(urls)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
