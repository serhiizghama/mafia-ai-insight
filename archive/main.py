"""
main.py

Entry point for the archive module (Stage 1).
Batch processor: reads URLs from urls.txt, transcribes, and saves to JSON.
Fully automated - no interactive prompts.
"""

import glob
import json
import os
import time
from datetime import datetime
from typing import List, Set

from dotenv import load_dotenv

from src.downloader import download_audio
from src.transcriber import get_transcriber


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration (e.g., "2m 35s", "1h 15m 23s", "42s")
    """
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    secs = seconds % 60

    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"


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


def load_processed_urls() -> Set[str]:
    """
    Scan data directory for existing game JSON files and extract processed URLs.
    Enables resume functionality by skipping already processed videos.

    Returns:
        Set[str]: Set of URLs that have been processed
    """
    processed = set()

    if not os.path.exists(DATA_DIR):
        return processed

    # Find all .json files in data directory
    json_files = glob.glob(os.path.join(DATA_DIR, "game_*.json"))

    if not json_files:
        return processed

    print(f"Scanning {len(json_files)} existing file(s) in {DATA_DIR}...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                url = data.get('url')
                if url:
                    processed.add(url)
        except Exception as e:
            print(f"⚠ Warning: Could not read {os.path.basename(file_path)}: {e}")
            continue

    return processed


def save_result(data: dict) -> None:
    """
    Save game data to an individual JSON file.
    Uses the game ID as the filename (e.g., game_1770125240.json).

    Args:
        data: Game data dictionary with 'id' field
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Use game ID as filename
    game_id = data.get('id', f"game_{int(time.time())}")
    filename = f"{game_id}.json"
    filepath = os.path.join(DATA_DIR, filename)

    # Write to individual file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✓ Game data saved to: {filepath}")


def process_video(url: str, transcriber, index: int, total: int, already_processed: int = 0) -> bool:
    """
    Process a single YouTube video: download, transcribe, and save.

    Args:
        url: YouTube video URL
        transcriber: Transcriber instance to use
        index: Current video index in this session (1-based)
        total: Total number of videos in urls.txt
        already_processed: Number of videos already in archive

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    k = already_processed + index

    print()
    print("=" * 80)
    print(f"=== Processing [{k} / {total}]: {url} ===")
    print("=" * 80)

    # Step 1: Download audio
    print()
    print(f"[{k}/{total}] STEP 1: Downloading audio...")
    audio_path, video_title = download_audio(url, output_dir=AUDIO_DIR)

    if audio_path is None:
        print(f"✗ Failed to download audio from: {url}")
        return False

    print(f"✓ Download complete: {video_title}")

    # Step 2: Transcribe
    print()
    print(f"[{k}/{total}] STEP 2: Transcribing audio...")

    # Start timer
    transcription_start = time.time()

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

    # Calculate transcription time
    transcription_duration = time.time() - transcription_start
    duration_str = format_duration(transcription_duration)

    print(f"✓ Transcription complete: {len(transcript_segments)} segments in {duration_str}")

    # Step 3: Save results
    print()
    print(f"[{k}/{total}] STEP 3: Saving results...")

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

    # Save to individual JSON file
    save_result(game_data)

    # Step 4: Cleanup
    print()
    print(f"[{k}/{total}] STEP 4: Cleaning up...")
    try:
        os.remove(audio_path)
        print(f"✓ Deleted temporary audio file: {audio_path}")
    except Exception as e:
        print(f"⚠ Could not delete audio file: {e}")

    print()
    print("=" * 80)
    print(f"✓ Video {k}/{total} processed successfully!")
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

    # Check for already processed URLs (scan existing JSON files)
    processed_urls = load_processed_urls()
    urls_to_process = [url for url in urls if url not in processed_urls]

    print(f"✓ Loaded {len(urls)} URL(s) from {URLS_FILE}")
    print()

    if processed_urls:
        print(f"✓ Found {len(processed_urls)} already processed video(s)")
        print(f"✓ {len(urls_to_process)} remaining URL(s) to process")
    else:
        print(f"✓ No previous progress found. Processing all {len(urls)} URL(s)")

    if not urls_to_process:
        print()
        print("=" * 80)
        print("✓ All URLs have been processed!")
        print("=" * 80)
        return

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

    for i, url in enumerate(urls_to_process, start=1):
        success = process_video(url, transcriber, i, len(urls), len(processed_urls))
        if success:
            success_count += 1
        else:
            failure_count += 1

    # Final summary
    print()
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Total URLs in queue: {len(urls)}")
    print(f"Already processed: {len(processed_urls)}")
    print(f"Processed this session: {success_count + failure_count}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
