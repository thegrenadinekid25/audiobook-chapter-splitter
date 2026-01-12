#!/usr/bin/env python3
"""
Audiobook Chapter Splitter

Automatically detect and split audiobook MP3 files by chapters using Whisper
speech-to-text for chapter detection.

Requirements:
- ffmpeg (brew install ffmpeg)
- whisper-cpp (brew install whisper-cpp)
- Whisper model file (downloaded automatically)

Usage:
    python audiobook_splitter.py /path/to/audiobook/folder
    python audiobook_splitter.py /path/to/audiobook/folder --segment-minutes 10
    python audiobook_splitter.py /path/to/audiobook/folder --analyze-only
    python audiobook_splitter.py /path/to/audiobook/folder --album "Book Title" --author "Author Name" --narrator "Narrator Name"
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Chapter:
    number: int
    title: str
    part: int  # Which source file
    timestamp: float  # Seconds from start of that file
    source_file: str


@dataclass
class AudioFile:
    path: Path
    duration: float
    order: int


MODEL_DIR = Path.home() / ".whisper-models"
MODEL_FILE = MODEL_DIR / "ggml-base.en.bin"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"


def run_command(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run a command and return (return_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_dependencies():
    """Check that required tools are installed."""
    missing = []

    # Check ffmpeg
    code, _, _ = run_command(["ffmpeg", "-version"])
    if code != 0:
        missing.append("ffmpeg (brew install ffmpeg)")

    # Check whisper-cli
    code, _, _ = run_command(["whisper-cli", "--help"])
    if code != 0:
        missing.append("whisper-cpp (brew install whisper-cpp)")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        sys.exit(1)


def ensure_model():
    """Download Whisper model if not present."""
    if MODEL_FILE.exists():
        return

    print(f"Downloading Whisper model to {MODEL_FILE}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    code, _, err = run_command([
        "curl", "-L", "-o", str(MODEL_FILE), MODEL_URL
    ], timeout=600)

    if code != 0:
        print(f"Failed to download model: {err}")
        sys.exit(1)

    print("Model downloaded successfully.")


def get_audio_duration(file_path: Path) -> float:
    """Get duration of audio file in seconds."""
    code, out, _ = run_command([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(file_path)
    ])
    if code == 0 and out.strip():
        return float(out.strip())
    return 0.0


def get_mp3_files(directory: Path) -> list[Path]:
    """Get all MP3 files in directory."""
    return sorted(directory.glob("*.mp3"))


def transcribe_file(mp3_path: Path, output_dir: Path, part_num: int) -> Optional[Path]:
    """Transcribe a single MP3 file to SRT format."""
    wav_path = output_dir / f"part{part_num}.wav"
    srt_path = output_dir / f"part{part_num}.srt"

    print(f"  [{part_num}] Converting to WAV...")

    # Convert to WAV (Whisper format)
    code, _, err = run_command([
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(wav_path)
    ], timeout=600)

    if code != 0:
        print(f"  [{part_num}] Conversion failed: {err}")
        return None

    print(f"  [{part_num}] Transcribing with Whisper...")

    # Transcribe
    code, _, err = run_command([
        "whisper-cli",
        "-m", str(MODEL_FILE),
        "-f", str(wav_path),
        "-osrt",
        "-of", str(output_dir / f"part{part_num}")
    ], timeout=7200)  # 2 hours max per file

    # Clean up WAV
    wav_path.unlink(missing_ok=True)

    if code != 0:
        print(f"  [{part_num}] Transcription failed: {err}")
        return None

    print(f"  [{part_num}] Done!")
    return srt_path


def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp to seconds."""
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', time_str)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    return 0.0


def find_chapters_in_srt(srt_path: Path, part_num: int, source_file: str) -> list[Chapter]:
    """Extract chapter markers from SRT file."""
    chapters = []

    with open(srt_path, 'r') as f:
        content = f.read()

    blocks = content.split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        timestamp_line = lines[1]
        text = ' '.join(lines[2:]).strip()

        # Match patterns like "1. Title" or "1." or "One. Title"
        # Numeric pattern
        match = re.match(r'^\s*"?(\d+)[\.,]\s*(.*)$', text)
        if match:
            chapter_num = int(match.group(1))
            title = match.group(2).strip().strip('"').strip()

            if 1 <= chapter_num <= 50:
                ts_match = re.match(r'(\d+:\d+:\d+,\d+)', timestamp_line)
                if ts_match:
                    timestamp = parse_srt_time(ts_match.group(1))
                    chapters.append(Chapter(
                        number=chapter_num,
                        title=title[:50] if title else f"Chapter {chapter_num}",
                        part=part_num,
                        timestamp=timestamp,
                        source_file=source_file
                    ))

        # Word number pattern (One, Two, etc.)
        word_nums = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20
        }

        text_lower = text.lower()
        for word, num in word_nums.items():
            if text_lower.startswith(word + '.') or text_lower.startswith(word + ','):
                ts_match = re.match(r'(\d+:\d+:\d+,\d+)', timestamp_line)
                if ts_match:
                    timestamp = parse_srt_time(ts_match.group(1))
                    title_part = text[len(word)+1:].strip()
                    chapters.append(Chapter(
                        number=num,
                        title=title_part[:50] if title_part else f"Chapter {num}",
                        part=part_num,
                        timestamp=timestamp,
                        source_file=source_file
                    ))

    # Remove duplicates
    seen = set()
    unique = []
    for ch in chapters:
        key = (ch.number, ch.part, round(ch.timestamp, 0))
        if key not in seen:
            seen.add(key)
            unique.append(ch)

    return sorted(unique, key=lambda x: x.timestamp)


def determine_file_order(mp3_files: list[Path], transcripts_dir: Path) -> list[AudioFile]:
    """
    Determine correct order of MP3 files by analyzing chapter numbers.
    Returns files sorted by the chapter number they start with.
    """
    file_order = []

    for i, mp3 in enumerate(mp3_files, 1):
        srt_path = transcripts_dir / f"part{i}.srt"
        if not srt_path.exists():
            continue

        chapters = find_chapters_in_srt(srt_path, i, mp3.name)
        starting_chapter = min((ch.number for ch in chapters), default=999)
        duration = get_audio_duration(mp3)

        file_order.append(AudioFile(
            path=mp3,
            duration=duration,
            order=starting_chapter
        ))

    # Sort by starting chapter number
    file_order.sort(key=lambda x: x.order)
    return file_order


@dataclass
class Metadata:
    album: Optional[str] = None
    author: Optional[str] = None
    narrator: Optional[str] = None
    year: Optional[str] = None


def add_metadata(file_path: Path, title: str, track_num: int, total_tracks: int,
                 metadata: Metadata) -> bool:
    """Add ID3 metadata to an MP3 file."""
    if not any([metadata.album, metadata.author, metadata.narrator]):
        return True  # No metadata to add

    temp_file = file_path.with_suffix('.tmp.mp3')

    cmd = [
        "ffmpeg", "-y", "-i", str(file_path),
        "-c", "copy",
        "-metadata", f"title={title}",
        "-metadata", f"track={track_num}/{total_tracks}",
        "-metadata", "genre=Audiobook",
    ]

    if metadata.album:
        cmd.extend(["-metadata", f"album={metadata.album}"])
    if metadata.author:
        cmd.extend(["-metadata", f"album_artist={metadata.author}"])
        cmd.extend(["-metadata", f"composer={metadata.author}"])
    if metadata.narrator:
        cmd.extend(["-metadata", f"artist={metadata.narrator}"])
    if metadata.year:
        cmd.extend(["-metadata", f"date={metadata.year}"])

    cmd.append(str(temp_file))

    code, _, _ = run_command(cmd, timeout=60)
    if code == 0:
        temp_file.replace(file_path)
        return True
    else:
        temp_file.unlink(missing_ok=True)
        return False


def split_by_chapters(chapters: list[Chapter], mp3_files: list[Path],
                     output_dir: Path, file_durations: dict[str, float],
                     metadata: Optional[Metadata] = None):
    """Split audio files by chapter boundaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group chapters by part and sort
    chapters_by_part: dict[int, list[Chapter]] = {}
    for ch in chapters:
        if ch.part not in chapters_by_part:
            chapters_by_part[ch.part] = []
        chapters_by_part[ch.part].append(ch)

    for part, chs in chapters_by_part.items():
        chs.sort(key=lambda x: x.timestamp)

    # Split each chapter
    for i, ch in enumerate(sorted(chapters, key=lambda x: x.number)):
        # Find end time (next chapter in same part or end of file)
        same_part_chapters = chapters_by_part.get(ch.part, [])
        next_ch = None
        for other in same_part_chapters:
            if other.timestamp > ch.timestamp:
                if next_ch is None or other.timestamp < next_ch.timestamp:
                    next_ch = other

        start_time = ch.timestamp
        end_time = next_ch.timestamp if next_ch else None

        # Clean title for filename
        clean_title = re.sub(r'[^a-z0-9]', '_', ch.title.lower())
        clean_title = re.sub(r'_+', '_', clean_title).strip('_')
        if not clean_title:
            clean_title = f"chapter_{ch.number}"

        out_file = output_dir / f"ch{ch.number:02d}_{clean_title}.mp3"

        print(f"  Creating: ch{ch.number:02d} - {ch.title}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(Path(mp3_files[0]).parent / ch.source_file),
            "-ss", str(start_time),
        ]
        if end_time:
            cmd.extend(["-to", str(end_time)])
        cmd.extend(["-acodec", "copy", str(out_file)])

        run_command(cmd, timeout=300)

        # Add metadata if provided
        if metadata and out_file.exists():
            add_metadata(out_file, ch.title, ch.number, len(chapters), metadata)


def split_by_duration(mp3_files: list[Path], output_dir: Path, segment_minutes: int):
    """Split audio files into fixed-duration segments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_seconds = segment_minutes * 60
    segment_num = 1

    for mp3 in mp3_files:
        duration = get_audio_duration(mp3)
        num_segments = int((duration + segment_seconds - 1) // segment_seconds)

        print(f"  Splitting {mp3.name} into {num_segments} segments...")

        for seg in range(num_segments):
            start = seg * segment_seconds
            out_file = output_dir / f"segment_{segment_num:03d}.mp3"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(mp3),
                "-ss", str(start),
                "-t", str(segment_seconds),
                "-acodec", "copy",
                str(out_file)
            ]
            run_command(cmd, timeout=120)
            segment_num += 1


def main():
    parser = argparse.ArgumentParser(
        description="Split audiobook MP3 files by chapters or fixed duration"
    )
    parser.add_argument("directory", help="Directory containing MP3 files")
    parser.add_argument("--segment-minutes", type=int,
                       help="Split into fixed segments of N minutes (skips chapter detection)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze and show detected chapters, don't split")
    parser.add_argument("--parallel", type=int, default=4,
                       help="Number of parallel transcription jobs (default: 4)")
    parser.add_argument("--output", help="Output directory (default: <directory>/chapters)")
    parser.add_argument("--album", help="Album/book title for metadata")
    parser.add_argument("--author", help="Author name for metadata")
    parser.add_argument("--narrator", help="Narrator name for metadata")
    parser.add_argument("--year", help="Publication year for metadata")

    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    # Check dependencies
    check_dependencies()
    ensure_model()

    # Get MP3 files
    mp3_files = get_mp3_files(directory)
    if not mp3_files:
        print(f"No MP3 files found in {directory}")
        sys.exit(1)

    print(f"Found {len(mp3_files)} MP3 files")

    # Output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    elif args.segment_minutes:
        output_dir = directory / "segments"
    else:
        output_dir = directory / "chapters"

    # Fixed-duration split mode
    if args.segment_minutes:
        print(f"\nSplitting into {args.segment_minutes}-minute segments...")
        split_by_duration(mp3_files, output_dir, args.segment_minutes)
        print(f"\nDone! Output in: {output_dir}")
        return

    # Chapter-based split mode
    transcripts_dir = directory / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    # Transcribe files
    print("\nTranscribing audio files (this may take a while)...")

    for i, mp3 in enumerate(mp3_files, 1):
        srt_path = transcripts_dir / f"part{i}.srt"
        if srt_path.exists():
            print(f"  [{i}] Already transcribed, skipping...")
            continue
        transcribe_file(mp3, transcripts_dir, i)

    # Find chapters
    print("\nAnalyzing chapters...")
    all_chapters: list[Chapter] = []
    file_durations: dict[str, float] = {}

    for i, mp3 in enumerate(mp3_files, 1):
        srt_path = transcripts_dir / f"part{i}.srt"
        if srt_path.exists():
            chapters = find_chapters_in_srt(srt_path, i, mp3.name)
            all_chapters.extend(chapters)
            file_durations[mp3.name] = get_audio_duration(mp3)

    # Sort by chapter number
    all_chapters.sort(key=lambda x: (x.number, x.timestamp))

    # Remove duplicate chapter numbers (keep first occurrence)
    seen_chapters = set()
    unique_chapters = []
    for ch in all_chapters:
        if ch.number not in seen_chapters:
            seen_chapters.add(ch.number)
            unique_chapters.append(ch)

    # Display chapters
    print("\nDetected chapters:")
    print("-" * 60)
    for ch in unique_chapters:
        mins = int(ch.timestamp // 60)
        secs = int(ch.timestamp % 60)
        print(f"  Ch {ch.number:2d}: {ch.title[:40]:40} (Part {ch.part} @ {mins:02d}:{secs:02d})")

    if args.analyze_only:
        # Save analysis to JSON
        analysis_file = directory / "chapter_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump([{
                'number': ch.number,
                'title': ch.title,
                'part': ch.part,
                'timestamp': ch.timestamp,
                'source_file': ch.source_file
            } for ch in unique_chapters], f, indent=2)
        print(f"\nAnalysis saved to: {analysis_file}")
        return

    # Create metadata object if any metadata args provided
    metadata = None
    if any([args.album, args.author, args.narrator, args.year]):
        metadata = Metadata(
            album=args.album,
            author=args.author,
            narrator=args.narrator,
            year=args.year
        )
        print(f"\nMetadata: album='{args.album}', author='{args.author}', narrator='{args.narrator}'")

    # Split by chapters
    print(f"\nSplitting into chapters...")
    split_by_chapters(unique_chapters, mp3_files, output_dir, file_durations, metadata)

    print(f"\nDone! {len(unique_chapters)} chapters created in: {output_dir}")


if __name__ == "__main__":
    main()
