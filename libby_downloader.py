#!/usr/bin/env python3
"""
Libby Audiobook Downloader

Downloads audiobooks from Libby using browser automation to capture
authentication tokens and audio segment URLs.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python libby_downloader.py [output_dir]

    1. Opens Libby in a browser
    2. You log in and navigate to an audiobook
    3. Click "Start listening" to begin playback
    4. Press Enter in the terminal to start download
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def download_segment(url: str, output_path: Path, index: int, total: int) -> tuple[int, bool, str]:
    """Download a single audio segment using curl."""
    try:
        result = subprocess.run(
            ["curl", "-sS", "-L", "-o", str(output_path), url],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return (index, True, "")
        else:
            return (index, False, result.stderr or "Empty file")
    except Exception as e:
        return (index, False, str(e))


def merge_audio_files(segment_paths: list[Path], output_path: Path, metadata: dict) -> bool:
    """Merge audio segments using ffmpeg with chapter markers."""

    # Create file list for ffmpeg concat
    list_file = output_path.parent / "segments.txt"
    with open(list_file, "w") as f:
        for path in segment_paths:
            # Escape single quotes in path
            escaped = str(path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    # Create chapter metadata file if chapters exist
    chapter_file = None
    if metadata.get("chapters"):
        chapter_file = output_path.parent / "chapters.txt"

        # Calculate spine offsets
        spine_offsets = []
        offset = 0
        for spine in metadata.get("spine", []):
            spine_offsets.append(offset)
            offset += spine.get("duration", 0)

        total_duration = offset

        # Build chapter metadata
        with open(chapter_file, "w") as f:
            f.write(";FFMETADATA1\n\n")

            last_title = None
            chapters = []
            for ch in metadata["chapters"]:
                if ch["title"] != last_title:
                    last_title = ch["title"]
                    spine_idx = ch.get("spine", 0)
                    ch_offset = ch.get("offset", 0)
                    start_ns = int((spine_offsets[spine_idx] + ch_offset) * 1_000_000_000)
                    chapters.append({
                        "title": ch["title"],
                        "start": start_ns
                    })

            # Add end times
            for i, ch in enumerate(chapters):
                if i + 1 < len(chapters):
                    ch["end"] = chapters[i + 1]["start"]
                else:
                    ch["end"] = int(total_duration * 1_000_000_000)

            for ch in chapters:
                title = ch["title"].replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")
                f.write("[CHAPTER]\n")
                f.write(f"START={ch['start']}\n")
                f.write(f"END={ch['end']}\n")
                f.write(f"title={title}\n\n")

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file)
    ]

    if chapter_file:
        cmd.extend(["-i", str(chapter_file), "-map_metadata", "1"])

    # Add metadata tags
    title = metadata.get("title", "Unknown")
    authors = ", ".join(c["name"] for c in metadata.get("creator", []) if c.get("role") == "author")
    narrators = ", ".join(c["name"] for c in metadata.get("creator", []) if c.get("role") == "narrator")

    cmd.extend([
        "-c", "copy",
        "-map", "0:a",
        "-metadata", f"title={title}",
        "-metadata", f"album={title}",
        "-metadata", f"artist={narrators or authors}",
        "-metadata", f"album_artist={authors}",
        "-metadata", "genre=Audiobook",
        str(output_path)
    ])

    print(f"\nMerging {len(segment_paths)} segments...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Cleanup
    list_file.unlink(missing_ok=True)
    if chapter_file:
        chapter_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False

    return True


def run_downloader(output_dir: Path, start_url: str = None):
    """Main downloader function using Playwright."""

    print("=" * 60)
    print("Libby Audiobook Downloader")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. A browser window will open")
    print("2. Log in to Libby with your library card")
    print("3. Navigate to the audiobook you want to download")
    print("4. Click 'Open in Libby' or 'Listen' to start the audio player")
    print("5. Come back here and press Enter to capture & download")
    print()

    captured_data = {
        "bif": None,
        "odread_params": None,
        "origin": None
    }

    with sync_playwright() as p:
        # Use existing Chrome profile to preserve login session
        chrome_user_data = Path.home() / "Library/Application Support/Google/Chrome"

        if chrome_user_data.exists():
            print("Using your existing Chrome profile (you'll be logged in)")
            # Launch with persistent context to use existing profile
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(chrome_user_data),
                channel="chrome",  # Use installed Chrome
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]
            )
            page = context.new_page()
            browser = None  # No separate browser object with persistent context
        else:
            print("Chrome profile not found, using fresh browser")
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()

        # Intercept responses to capture odread-cmpt-params
        def handle_response(response):
            try:
                if response.status == 200:
                    content_type = response.headers.get("content-type", "")
                    if "json" in content_type:
                        try:
                            data = response.json()
                            if isinstance(data, dict) and "b" in data and "-odread-cmpt-params" in data.get("b", {}):
                                captured_data["odread_params"] = data["b"]["-odread-cmpt-params"]
                                print(f"[Captured auth params: {len(captured_data['odread_params'])} segments]")
                        except:
                            pass
            except:
                pass

        page.on("response", handle_response)

        # Navigate to Libby
        initial_url = start_url or "https://libbyapp.com"
        page.goto(initial_url)
        print(f"\nBrowser opened to: {initial_url}")
        if not start_url:
            print("Navigate to your audiobook and start playback.")
        else:
            print("Click 'Listen' or 'Open in Libby' to start the audio player.")
        print("\nThe script will automatically detect when the audiobook is ready...")

        # Wait for the audio player page and BIF object
        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                current_url = page.url
                # Check if we're on the listen page
                if "listen.libbyapp.com" in current_url or "listen.overdrive.com" in current_url:
                    # Check if BIF is available
                    has_bif = page.evaluate("typeof BIF !== 'undefined' && BIF.objects && BIF.objects.spool")
                    if has_bif and captured_data["odread_params"]:
                        print("\nAudiobook detected! Starting capture...")
                        time.sleep(2)  # Brief pause to ensure everything is loaded
                        break
            except:
                pass
            time.sleep(1)
        else:
            print("\nTimeout waiting for audiobook. Make sure you navigated to the listen page.")
            context.close()
            if browser:
                browser.close()
            return False

        # Capture the BIF object and current URL
        captured_data["origin"] = page.evaluate("location.origin")

        try:
            bif_data = page.evaluate("""() => {
                if (typeof BIF === 'undefined') return null;

                // Extract relevant data from BIF
                const spineToIndex = BIF.map.spine.map(x => x["-odread-original-path"]);

                return {
                    title: BIF.map.title.main,
                    description: BIF.map.description,
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
                    }))
                };
            }""")
            captured_data["bif"] = bif_data
        except Exception as e:
            print(f"Error capturing BIF: {e}")

        context.close()
        if browser:
            browser.close()

    # Validate captured data
    if not captured_data["bif"]:
        print("\nError: Could not capture audiobook data.")
        print("Make sure you're on the audio player page (listen.libbyapp.com)")
        return False

    if not captured_data["odread_params"]:
        print("\nError: Could not capture authentication parameters.")
        print("Try refreshing the page and starting playback again.")
        return False

    bif = captured_data["bif"]
    params = captured_data["odread_params"]
    origin = captured_data["origin"]

    # Display book info
    print()
    print("=" * 60)
    print(f"Title: {bif['title']}")
    authors = [c["name"] for c in bif.get("creator", []) if c.get("role") == "author"]
    narrators = [c["name"] for c in bif.get("creator", []) if c.get("role") == "narrator"]
    if authors:
        print(f"Author: {', '.join(authors)}")
    if narrators:
        print(f"Narrator: {', '.join(narrators)}")

    total_duration = sum(c["duration"] for c in bif["components"])
    print(f"Duration: {format_duration(total_duration)} ({total_duration/3600:.1f} hours)")
    print(f"Segments: {len(bif['components'])}")
    print(f"Chapters: {len(bif.get('chapters', []))}")
    print("=" * 60)

    # Auto-proceed with download
    print("\nStarting download...")

    # Create output directory
    book_name = sanitize_filename(bif["title"])
    book_dir = output_dir / book_name
    book_dir.mkdir(parents=True, exist_ok=True)

    segments_dir = book_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    # Save metadata
    with open(book_dir / "metadata.json", "w") as f:
        json.dump(bif, f, indent=2)

    # Build download URLs
    components = bif["components"]
    download_urls = []
    for comp in components:
        idx = comp["position"]
        if idx < len(params):
            url = f"{origin}/{comp['path']}?{params[idx]}"
            download_urls.append((idx, url, comp["duration"]))
        else:
            print(f"Warning: No auth param for segment {idx}")

    # Download segments in parallel
    print(f"\nDownloading {len(download_urls)} segments...")
    segment_paths = [None] * len(download_urls)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for idx, url, duration in download_urls:
            output_path = segments_dir / f"segment_{idx:03d}.mp3"
            segment_paths[idx] = output_path
            future = executor.submit(download_segment, url, output_path, idx, len(download_urls))
            futures[future] = idx

        completed = 0
        for future in as_completed(futures):
            idx, success, error = future.result()
            completed += 1
            if success:
                print(f"  [{completed}/{len(download_urls)}] Segment {idx+1} downloaded")
            else:
                print(f"  [{completed}/{len(download_urls)}] Segment {idx+1} FAILED: {error}")

    # Check all segments downloaded
    missing = [i for i, p in enumerate(segment_paths) if p is None or not p.exists()]
    if missing:
        print(f"\nWarning: {len(missing)} segments failed to download: {missing}")

    # Filter to existing segments
    valid_paths = [p for p in segment_paths if p and p.exists()]

    if not valid_paths:
        print("\nError: No segments downloaded successfully.")
        return False

    # Merge segments
    output_file = book_dir / f"{book_name}.mp3"
    if merge_audio_files(valid_paths, output_file, bif):
        print(f"\nSuccess! Audiobook saved to:")
        print(f"  {output_file}")

        # Cleanup segments automatically
        print("\nCleaning up segment files...")
        for p in valid_paths:
            p.unlink(missing_ok=True)
        try:
            segments_dir.rmdir()
        except:
            pass
        print("Segments cleaned up.")

        return True
    else:
        print("\nMerge failed. Individual segments preserved in:")
        print(f"  {segments_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download audiobooks from Libby")
    parser.add_argument("output_dir", nargs="?", default=".", help="Output directory (default: current)")
    parser.add_argument("--url", "-u", help="Libby audiobook URL to open directly")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    success = run_downloader(output_dir, args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
