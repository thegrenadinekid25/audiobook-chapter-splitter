#!/usr/bin/env python3 -u
"""
Libby Audiobook Downloader & Chapter Splitter

End-to-end pipeline: Libby URL -> individual chapter MP3s with full metadata.

Requirements:
    - ffmpeg (brew install ffmpeg)
    - playwright (pip install playwright && playwright install chromium)
    - whisper-cpp (brew install whisper-cpp) -- only needed if book lacks chapter data

Usage:
    python libby_audiobook.py "https://libbyapp.com/open/loan/12345/67890"
    python libby_audiobook.py "https://libbyapp.com/open/loan/..." --output ~/Audiobooks
    python libby_audiobook.py "https://libbyapp.com/open/loan/..." --cover cover.jpg --year 2023
    python libby_audiobook.py  # opens Libby for you to browse
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LibbyBook:
    title: str
    creator: list[dict] = field(default_factory=list)
    spine: list[dict] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    odread_params: list[str] = field(default_factory=list)
    origin: str = ""
    cover_url: Optional[str] = None


@dataclass
class BookMetadata:
    album: str
    author: str
    narrator: str
    year: Optional[str] = None
    cover_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Whisper fallback constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path.home() / ".whisper-models"
MODEL_FILE = MODEL_DIR / "ggml-base.en.bin"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run a command and return (return_code, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


# ---------------------------------------------------------------------------
# Phase 1: Browser automation -- capture Libby data
# ---------------------------------------------------------------------------

def capture_and_download(start_url: Optional[str] = None,
                         book_dir: Optional[Path] = None) -> Optional[tuple[LibbyBook, Path]]:
    """Open browser, capture BIF data, download segments via browser fetch, merge."""

    print("=" * 60)
    print("Libby Audiobook Downloader")
    print("=" * 60)
    print()
    print("Instructions:")
    print("  1. A browser window will open")
    print("  2. Log in to Libby with your library card (if needed)")
    print("  3. Navigate to the audiobook and click 'Listen'")
    print("  4. The script will automatically detect and proceed")
    print()

    # JavaScript injected into ALL frames before they load.
    CAPTURE_SCRIPT = """
    (() => {
        if (window.__libbyCapture) return;
        window.__libbyCapture = {
            authTokens: null,
            bifData: null,
            ready: false
        };

        const originalParse = JSON.parse;
        JSON.parse = function(...args) {
            const result = originalParse.apply(this, args);
            try {
                if (result && typeof result === 'object' && result.b && result.b['-odread-cmpt-params']) {
                    window.__libbyCapture.authTokens = Array.from(result.b['-odread-cmpt-params']);
                }
            } catch(e) {}
            return result;
        };

        let checkCount = 0;
        const bifCheck = setInterval(() => {
            checkCount++;
            try {
                if (typeof BIF !== 'undefined' && BIF.objects && BIF.objects.spool) {
                    const spineToIndex = BIF.map.spine.map(x => x["-odread-original-path"]);
                    let coverUrl = null;
                    if (BIF.map.cover) {
                        let coverPath = BIF.map.cover;
                        if (typeof coverPath === 'object') {
                            coverPath = coverPath.href || coverPath.src || coverPath.url || coverPath.path || '';
                        }
                        if (typeof coverPath === 'string' && coverPath) {
                            coverUrl = coverPath.startsWith('http') ? coverPath : location.origin + '/' + coverPath;
                        }
                    }

                    window.__libbyCapture.bifData = {
                        title: BIF.map.title.main,
                        creator: BIF.map.creator,
                        spine: BIF.map.spine.map(x => ({
                            duration: x["audio-duration"],
                            type: x["media-type"],
                            bitrate: x["audio-bitrate"],
                            path: x["-odread-original-path"]
                        })),
                        chapters: BIF.map.nav && BIF.map.nav.toc ? BIF.map.nav.toc.map(ch => ({
                            title: ch.title,
                            spine: spineToIndex.indexOf(ch.path.split("#")[0]),
                            offset: parseFloat(ch.path.split("#")[1]) || 0
                        })) : [],
                        components: BIF.objects.spool.components.map(c => ({
                            path: c.meta.path,
                            position: c.spinePosition,
                            duration: c.meta["audio-duration"],
                            size: c.meta["-odread-file-bytes"],
                            type: c.meta["media-type"]
                        })),
                        coverUrl: coverUrl,
                        origin: location.origin
                    };
                    window.__libbyCapture.ready = true;
                    clearInterval(bifCheck);
                }
            } catch(e) {}
            if (checkCount > 120) clearInterval(bifCheck);
        }, 500);
    })();
    """

    with sync_playwright() as p:
        libby_profile = Path.home() / ".libby-audiobook-profile"
        libby_profile.mkdir(parents=True, exist_ok=True)

        print("Using dedicated browser profile (login is cached between runs)")
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(libby_profile),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context.add_init_script(CAPTURE_SCRIPT)
        page = context.new_page()

        initial_url = start_url or "https://libbyapp.com"
        page.goto(initial_url)
        print(f"Browser opened: {initial_url}")
        print("Waiting for audiobook player...")

        # Poll all frames for captured data
        max_wait = 600
        start_time = time.time()
        capture_result = None
        bif_frame = None

        while time.time() - start_time < max_wait:
            for frame in page.frames:
                try:
                    data = frame.evaluate("window.__libbyCapture")
                    if data and data.get("ready") and data.get("authTokens") and data.get("bifData"):
                        capture_result = data
                        bif_frame = frame
                        break
                except Exception:
                    pass

            if capture_result:
                print("\nAudiobook data captured!")
                time.sleep(1)
                break

            time.sleep(1)
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0 and elapsed > 0:
                print(f"  Still waiting... ({elapsed}s)")
        else:
            print("\nTimeout. Make sure the audiobook player loaded and audio is playing.")
            context.close()
            return None

        bif = capture_result["bifData"]
        auth_tokens = capture_result["authTokens"]

        book = LibbyBook(
            title=bif["title"],
            creator=bif.get("creator", []),
            spine=bif.get("spine", []),
            chapters=bif.get("chapters", []),
            components=bif.get("components", []),
            odread_params=auth_tokens,
            origin=bif.get("origin", ""),
            cover_url=bif.get("coverUrl"),
        )

        # Display info
        authors = [c["name"] for c in book.creator if c.get("role") == "author"]
        narrators = [c["name"] for c in book.creator if c.get("role") == "narrator"]
        total_duration = sum(s.get("duration") or 0 for s in book.spine)

        print()
        print("=" * 60)
        print(f"  Title:    {book.title}")
        if authors:
            print(f"  Author:   {', '.join(authors)}")
        if narrators:
            print(f"  Narrator: {', '.join(narrators)}")
        print(f"  Duration: {format_duration(total_duration)} ({total_duration/3600:.1f} hours)")
        print(f"  Segments: {len(book.components)}")
        print(f"  Chapters: {len(book.chapters)}")
        print("=" * 60)

        # --- Download segments via browser fetch (keeps auth context) ---
        book_name = sanitize_filename(book.title)
        if not book_dir:
            book_dir = Path.cwd() / book_name
        book_dir.mkdir(parents=True, exist_ok=True)

        segments_dir = book_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # Build download list
        downloads = []
        for comp in book.components:
            idx = comp["position"]
            if idx < len(auth_tokens):
                url = f"{book.origin}/{comp['path']}?{auth_tokens[idx]}"
                downloads.append((idx, url))

        print(f"\nDownloading {len(downloads)} segments via browser...")

        # Download each segment using fetch() inside the browser iframe
        success_count = 0
        for i, (idx, url) in enumerate(sorted(downloads)):
            seg_path = segments_dir / f"segment_{idx:03d}.mp3"
            try:
                # Use Playwright's request API (shares browser cookies/context)
                response = context.request.get(url)
                if response.ok:
                    seg_path.write_bytes(response.body())
                    if seg_path.stat().st_size > 1000:
                        success_count += 1
                        print(f"  [{i+1}/{len(downloads)}] Segment {idx + 1} downloaded ({seg_path.stat().st_size // 1024}KB)")
                    else:
                        print(f"  [{i+1}/{len(downloads)}] Segment {idx + 1} too small ({seg_path.stat().st_size}B)")
                else:
                    print(f"  [{i+1}/{len(downloads)}] Segment {idx + 1} HTTP {response.status}")
            except Exception as e:
                print(f"  [{i+1}/{len(downloads)}] Segment {idx + 1} FAILED: {e}")

        context.close()

    print(f"\nDownloaded {success_count}/{len(downloads)} segments")

    if success_count == 0:
        print("All downloads failed.")
        return None

    # Merge segments with ffmpeg
    valid_paths = []
    for idx in range(max(d[0] for d in downloads) + 1):
        p = segments_dir / f"segment_{idx:03d}.mp3"
        if p.exists() and p.stat().st_size > 1000:
            valid_paths.append(p)

    merged_path = book_dir / f"{book_name}.mp3"
    list_file = segments_dir / "concat.txt"
    with open(list_file, "w") as f:
        for seg_p in valid_paths:
            escaped = str(seg_p).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    print(f"\nMerging {len(valid_paths)} segments...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(merged_path)],
        capture_output=True, text=True
    )

    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        stderr_lines = result.stderr.strip().split('\n')
        error_lines = [l for l in stderr_lines if not l.startswith(('  ', 'ffmpeg version', '  built', '  configuration', '  lib'))]
        print(f"Merge failed: {chr(10).join(error_lines[-5:])}")
        print(f"  Segments preserved in: {segments_dir}")
        return None

    # Cleanup segments
    for seg_p in valid_paths:
        seg_p.unlink(missing_ok=True)
    try:
        segments_dir.rmdir()
    except OSError:
        pass

    print(f"  Merged: {merged_path.name}")
    return (book, merged_path)


# ---------------------------------------------------------------------------
# Phase 3a: Chapter splitting using BIF data
# ---------------------------------------------------------------------------

def calculate_chapter_timestamps(book: LibbyBook) -> list[dict]:
    """Convert BIF chapter references to absolute timestamps."""

    # Build cumulative spine offsets
    spine_offsets = []
    cumulative = 0.0
    for s in book.spine:
        spine_offsets.append(cumulative)
        cumulative += s["duration"]
    total_duration = cumulative

    # Convert chapters to absolute timestamps
    seen = set()
    chapters = []
    for ch in book.chapters:
        spine_idx = ch.get("spine", 0)
        offset = ch.get("offset", 0)

        # Skip invalid spine references
        if spine_idx < 0 or spine_idx >= len(spine_offsets):
            continue

        start = spine_offsets[spine_idx] + offset
        key = (ch["title"], round(start, 1))
        if key in seen:
            continue
        seen.add(key)

        chapters.append({"title": ch["title"], "start": start})

    # Sort by start time
    chapters.sort(key=lambda c: c["start"])

    # Filter very short chapters (< 1s) and assign numbers + end times
    result = []
    for i, ch in enumerate(chapters):
        end = chapters[i + 1]["start"] if i + 1 < len(chapters) else total_duration
        if end - ch["start"] < 1.0:
            continue
        ch["end"] = end
        ch["number"] = len(result) + 1
        result.append(ch)

    return result


def split_at_chapters(merged_mp3: Path, chapters: list[dict], output_dir: Path,
                      metadata: BookMetadata) -> list[Path]:
    """Split merged MP3 at chapter boundaries and apply metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    total_tracks = len(chapters)
    created = []

    for ch in chapters:
        # Build filename
        clean_title = re.sub(r'[^a-z0-9]', '_', ch["title"].lower())
        clean_title = re.sub(r'_+', '_', clean_title).strip('_')
        if not clean_title:
            clean_title = f"chapter_{ch['number']}"

        out_file = output_dir / f"ch{ch['number']:02d}_{clean_title}.mp3"
        print(f"  ch{ch['number']:02d} - {ch['title']}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(merged_mp3),
            "-ss", str(ch["start"]),
            "-to", str(ch["end"]),
            "-c", "copy",
            str(out_file)
        ]
        run_command(cmd, timeout=300)

        if out_file.exists():
            add_metadata(out_file, ch["title"], ch["number"], total_tracks, metadata)
            if metadata.cover_path and metadata.cover_path.exists():
                embed_cover(out_file, metadata.cover_path)
            created.append(out_file)

    return created


# ---------------------------------------------------------------------------
# Phase 3b: Whisper fallback for chapter detection
# ---------------------------------------------------------------------------

def check_whisper_dependencies():
    """Check that Whisper dependencies are installed."""
    missing = []

    code, _, _ = run_command(["ffmpeg", "-version"])
    if code != 0:
        missing.append("ffmpeg (brew install ffmpeg)")

    code, _, _ = run_command(["whisper-cli", "--help"])
    if code != 0:
        missing.append("whisper-cpp (brew install whisper-cpp)")

    if missing:
        print("Missing dependencies for Whisper fallback:")
        for dep in missing:
            print(f"  - {dep}")
        return False
    return True


def ensure_whisper_model():
    """Download Whisper model if not present."""
    if MODEL_FILE.exists():
        return True

    print(f"Downloading Whisper model to {MODEL_FILE}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    code, _, err = run_command([
        "curl", "-L", "-o", str(MODEL_FILE), MODEL_URL
    ], timeout=600)

    if code != 0:
        print(f"Failed to download model: {err}")
        return False

    print("Model downloaded.")
    return True


def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp to seconds."""
    match = re.match(r'(\d+):(\d+):(\d+),(\d+)', time_str)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    return 0.0


def find_chapters_in_srt(srt_path: Path) -> list[dict]:
    """Extract chapter markers from SRT file."""
    with open(srt_path, 'r') as f:
        content = f.read()

    chapters = []
    blocks = content.split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        timestamp_line = lines[1]
        text = ' '.join(lines[2:]).strip()

        # Numeric pattern: "1. Title" or "1,"
        match = re.match(r'^\s*"?(\d+)[\.,]\s*(.*)$', text)
        if match:
            chapter_num = int(match.group(1))
            title = match.group(2).strip().strip('"').strip()

            if 1 <= chapter_num <= 50:
                ts_match = re.match(r'(\d+:\d+:\d+,\d+)', timestamp_line)
                if ts_match:
                    timestamp = parse_srt_time(ts_match.group(1))
                    chapters.append({
                        "title": title[:50] if title else f"Chapter {chapter_num}",
                        "start": timestamp,
                        "_num": chapter_num,
                    })

        # Word number pattern: "One. Title", "Two,"
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
                    chapters.append({
                        "title": title_part[:50] if title_part else f"Chapter {num}",
                        "start": timestamp,
                        "_num": num,
                    })

    # Deduplicate
    seen = set()
    unique = []
    for ch in chapters:
        key = (ch["_num"], round(ch["start"], 0))
        if key not in seen:
            seen.add(key)
            unique.append(ch)

    unique.sort(key=lambda x: x["start"])

    # Remove internal _num, keep first occurrence per number
    seen_nums = set()
    result = []
    for ch in unique:
        if ch["_num"] not in seen_nums:
            seen_nums.add(ch["_num"])
            del ch["_num"]
            result.append(ch)

    return result


def whisper_chapter_detection(merged_mp3: Path) -> list[dict]:
    """Detect chapters via Whisper transcription (fallback)."""

    print("\nNo chapter data from Libby. Falling back to Whisper transcription...")

    if not check_whisper_dependencies():
        return []
    if not ensure_whisper_model():
        return []

    # Transcribe
    transcripts_dir = merged_mp3.parent / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    srt_path = transcripts_dir / "part1.srt"
    if not srt_path.exists():
        wav_path = transcripts_dir / "part1.wav"

        print("  Converting to WAV...")
        code, _, err = run_command([
            "ffmpeg", "-y", "-i", str(merged_mp3),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path)
        ], timeout=600)

        if code != 0:
            print(f"  WAV conversion failed: {err}")
            return []

        print("  Transcribing with Whisper (this may take a while)...")
        code, _, err = run_command([
            "whisper-cli",
            "-m", str(MODEL_FILE),
            "-f", str(wav_path),
            "-osrt",
            "-of", str(transcripts_dir / "part1")
        ], timeout=7200)

        wav_path.unlink(missing_ok=True)

        if code != 0:
            print(f"  Transcription failed: {err}")
            return []

        print("  Transcription complete!")

    # Parse chapters from SRT
    chapters = find_chapters_in_srt(srt_path)

    if not chapters:
        print("  No chapters detected in transcription.")
        return []

    # Add end times
    total_duration = get_audio_duration(merged_mp3)
    for i, ch in enumerate(chapters):
        ch["end"] = chapters[i + 1]["start"] if i + 1 < len(chapters) else total_duration
        ch["number"] = i + 1

    return chapters


# ---------------------------------------------------------------------------
# Cover art and metadata
# ---------------------------------------------------------------------------

def fetch_cover_from_isbn(isbn: str, output_path: Path) -> bool:
    """Try to fetch book cover from Google Books API."""
    import urllib.request
    import urllib.error

    api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    try:
        with urllib.request.urlopen(api_url, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get("totalItems", 0) == 0:
            return False

        volume = data["items"][0]["volumeInfo"]
        image_links = volume.get("imageLinks", {})

        image_url = (image_links.get("extraLarge") or image_links.get("large") or
                     image_links.get("medium") or image_links.get("thumbnail"))

        if not image_url:
            return False

        image_url = re.sub(r'&zoom=\d', '&zoom=0', image_url)
        image_url = image_url.replace('http://', 'https://')

        urllib.request.urlretrieve(image_url, str(output_path))
        return output_path.exists() and output_path.stat().st_size > 1000

    except (Exception,):
        return False


def download_cover_image(url_or_path: str, output_dir: Path) -> Optional[Path]:
    """Download cover image from URL or return path if local file."""
    import urllib.request
    import urllib.error

    local_path = Path(url_or_path)
    if local_path.exists():
        return local_path

    output_path = output_dir / "cover.jpg"
    try:
        print(f"  Downloading cover: {url_or_path[:60]}...")
        urllib.request.urlretrieve(url_or_path, str(output_path))
        if output_path.exists() and output_path.stat().st_size > 1000:
            return output_path
    except Exception as e:
        print(f"  Failed to download cover: {e}")

    return None


def embed_cover(mp3_path: Path, cover_path: Path) -> bool:
    """Embed cover art into MP3 file."""
    temp_file = mp3_path.with_suffix('.tmp.mp3')

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-i", str(cover_path),
        "-map", "0:a",
        "-map", "1:0",
        "-c:a", "copy",
        "-c:v", "mjpeg",
        "-metadata:s:v", "title=Album cover",
        "-metadata:s:v", "comment=Cover (front)",
        "-id3v2_version", "3",
        str(temp_file)
    ]

    code, _, _ = run_command(cmd, timeout=60)
    if code == 0:
        temp_file.replace(mp3_path)
        return True
    else:
        temp_file.unlink(missing_ok=True)
        return False


def add_metadata(file_path: Path, title: str, track_num: int, total_tracks: int,
                 metadata: BookMetadata) -> bool:
    """Add ID3 metadata to an MP3 file."""
    temp_file = file_path.with_suffix('.tmp.mp3')

    cmd = [
        "ffmpeg", "-y", "-i", str(file_path),
        "-c", "copy",
        "-metadata", f"title={title}",
        "-metadata", f"track={track_num}/{total_tracks}",
        "-metadata", "genre=Audiobook",
        "-metadata", f"album={metadata.album}",
    ]

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


def resolve_cover(book: LibbyBook, book_dir: Path,
                  cover_arg: Optional[str] = None,
                  isbn_arg: Optional[str] = None) -> Optional[Path]:
    """Resolve cover art from various sources. Priority: CLI > BIF > ISBN."""
    if cover_arg:
        path = download_cover_image(cover_arg, book_dir)
        if path:
            return path

    if book.cover_url:
        path = download_cover_image(book.cover_url, book_dir)
        if path:
            return path

    if isbn_arg:
        cover_file = book_dir / "cover.jpg"
        print(f"  Fetching cover for ISBN {isbn_arg}...")
        if fetch_cover_from_isbn(isbn_arg, cover_file):
            return cover_file

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(url: Optional[str], output_dir: Path, cover_arg: Optional[str] = None,
                 isbn_arg: Optional[str] = None, year: Optional[str] = None,
                 no_split: bool = False, force_whisper: bool = False,
                 keep_merged: bool = False, max_workers: int = 4) -> bool:
    """End-to-end: capture -> download -> split -> tag."""

    # Phase 1+2: Capture and download (browser stays open for auth)
    book_name_hint = None
    result = capture_and_download(url)
    if not result:
        return False

    book, merged_mp3 = result
    book_dir = merged_mp3.parent

    # Save metadata JSON
    metadata_file = book_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "title": book.title,
            "creator": book.creator,
            "spine": book.spine,
            "chapters": book.chapters,
            "components_count": len(book.components),
            "total_duration": sum(s.get("duration") or 0 for s in book.spine),
        }, f, indent=2)

    if no_split:
        print(f"\nDone! Audiobook saved to: {merged_mp3}")
        return True

    # Phase 3: Determine chapters
    chapters = []
    if book.chapters and not force_whisper:
        print(f"\nUsing {len(book.chapters)} chapters from Libby metadata")
        chapters = calculate_chapter_timestamps(book)
    else:
        chapters = whisper_chapter_detection(merged_mp3)

    if not chapters:
        print("\nNo chapters detected. Keeping merged file.")
        print(f"  {merged_mp3}")
        return True

    # Display chapters
    print(f"\nChapters ({len(chapters)}):")
    print("-" * 60)
    for ch in chapters:
        start_min = int(ch["start"] // 60)
        start_sec = int(ch["start"] % 60)
        print(f"  {ch['number']:2d}. {ch['title'][:45]:45} @ {start_min:02d}:{start_sec:02d}")

    # Resolve cover art
    cover_path = resolve_cover(book, book_dir, cover_arg, isbn_arg)
    if cover_path:
        print(f"\nCover art: {cover_path}")

    # Build metadata
    authors = ", ".join(c["name"] for c in book.creator if c.get("role") == "author")
    narrators = ", ".join(c["name"] for c in book.creator if c.get("role") == "narrator")

    metadata = BookMetadata(
        album=book.title,
        author=authors,
        narrator=narrators,
        year=year,
        cover_path=cover_path,
    )

    # Phase 4: Split
    chapters_dir = book_dir / "chapters"
    print(f"\nSplitting into chapters...")
    created = split_at_chapters(merged_mp3, chapters, chapters_dir, metadata)

    # Cleanup merged file
    if not keep_merged and created:
        merged_mp3.unlink(missing_ok=True)

    print(f"\nDone! {len(created)} chapters saved to:")
    print(f"  {chapters_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and split Libby audiobooks into chapters"
    )
    parser.add_argument("url", nargs="?",
                        help="Libby audiobook URL (opens libbyapp.com if omitted)")
    parser.add_argument("--output", "-o", default=".",
                        help="Output directory (default: current)")
    parser.add_argument("--cover",
                        help="Cover image URL or local file path")
    parser.add_argument("--isbn",
                        help="ISBN to auto-fetch cover from Google Books")
    parser.add_argument("--year",
                        help="Publication year for metadata")
    parser.add_argument("--no-split", action="store_true",
                        help="Download only, don't split into chapters")
    parser.add_argument("--force-whisper", action="store_true",
                        help="Force Whisper transcription even if Libby has chapters")
    parser.add_argument("--keep-merged", action="store_true",
                        help="Keep the merged MP3 after splitting")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")

    args = parser.parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    success = run_pipeline(
        url=args.url,
        output_dir=output_dir,
        cover_arg=args.cover,
        isbn_arg=args.isbn,
        year=args.year,
        no_split=args.no_split,
        force_whisper=args.force_whisper,
        keep_merged=args.keep_merged,
        max_workers=args.workers,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
