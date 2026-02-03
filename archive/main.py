"""
main.py

Entry point for the archive module (Stage 1).
Downloads audio from YouTube, transcribes it, and saves structured game data.
"""

import json
import os
import time
from datetime import datetime

from src.downloader import download_audio
from src.transcriber import MafiaTranscriber

# Get the base directory of the archive package (always points to archive/ folder)
ARCHIVE_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(ARCHIVE_BASE_DIR, "audio")
DATA_DIR = os.path.join(ARCHIVE_BASE_DIR, "data")


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

    print(f"\nGame data saved to: {filepath}")


def main():
    """
    Main CLI workflow for the archive module.
    """
    # Display banner
    print("=" * 60)
    print("MAFIA AI INSIGHT - ARCHIVE MODULE (Stage 1)")
    print("Digital Archive: YouTube -> Audio -> Transcription -> JSON")
    print("=" * 60)
    print()

    # Prompt for YouTube URL
    youtube_url = input("Enter YouTube Video URL: ").strip()
    if not youtube_url:
        print("Error: YouTube URL is required.")
        return

    # Prompt for Game ID (optional, auto-generate if empty)
    game_id = input("Game ID (e.g., 'Game_01'): ").strip()
    if not game_id:
        game_id = f"game_{int(time.time())}"
        print(f"Auto-generated Game ID: {game_id}")

    # Prompt for Mafia roles
    mafia_roles = input("Who was Mafia? (e.g., 'Player 1, Player 5'): ").strip()

    # Prompt for winner
    winner = input("Who won? (Town/Mafia): ").strip()

    print()
    print("-" * 60)
    print("STEP 1: Downloading audio from YouTube...")
    print("-" * 60)

    # Download audio (use absolute path to archive/audio/)
    audio_path, video_title = download_audio(youtube_url, output_dir=AUDIO_DIR)

    if audio_path is None:
        print("Error: Failed to download audio. Aborting.")
        return

    print()
    print("-" * 60)
    print("STEP 2: Transcribing audio...")
    print("-" * 60)

    # Initialize transcriber and transcribe
    transcriber = MafiaTranscriber(model_size="medium", device="cpu", compute_type="int8")
    transcript_segments = transcriber.transcribe(audio_path)

    print()
    print("-" * 60)
    print("STEP 3: Saving results...")
    print("-" * 60)

    # Get current timestamp
    processed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct game data
    game_data = {
        "id": game_id,
        "title": video_title,
        "url": youtube_url,
        "meta": {
            "mafia_roles": mafia_roles,
            "winner": winner,
            "processed_date": processed_date
        },
        "transcript": transcript_segments
    }

    # Save to JSON archive
    save_result(game_data)

    print()
    print("=" * 60)
    print("ARCHIVE COMPLETE!")
    print(f"Game ID: {game_id}")
    print(f"Transcript segments: {len(transcript_segments)}")
    print("=" * 60)

    # Optional: Clean up audio file
    cleanup = input("\nDelete downloaded audio file? (y/N): ").strip().lower()
    if cleanup == 'y':
        try:
            os.remove(audio_path)
            print(f"Deleted: {audio_path}")
        except Exception as e:
            print(f"Could not delete audio file: {e}")


if __name__ == "__main__":
    main()
