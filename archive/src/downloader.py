"""
downloader.py

Wrapper for yt-dlp to download audio from YouTube videos.
"""

import os
from typing import Optional, Tuple

import yt_dlp


def download_audio(youtube_url: str, output_dir: str = "audio") -> Tuple[Optional[str], Optional[str]]:
    """
    Download audio from a YouTube video.

    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save the audio file (default: "audio")

    Returns:
        Tuple[Optional[str], Optional[str]]: (filepath, title) on success, (None, None) on error
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Configure yt-dlp options
    # player_client: try android/mweb to reduce HTTP 403 (YouTube often blocks default client)
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {'player_client': ['android', 'mweb']},
        },
    }

    try:
        print(f"Downloading audio from: {youtube_url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)

            # Construct the output filename
            video_id = info['id']
            filename = f"{video_id}.m4a"
            filepath = os.path.join(output_dir, filename)

            # Get video title
            title = info.get('title', 'Unknown Title')

            print(f"Audio downloaded successfully: {filepath}")
            print(f"Video title: {title}")

            return filepath, title

    except Exception as e:
        print(f"Download error: {e}")
        return None, None
