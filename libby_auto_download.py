#!/usr/bin/env python3
"""
Libby Audiobook Auto-Downloader v2

Fully automated download using Playwright.
You'll need to log in to Libby once in the browser that opens,
then the script handles everything else.

Usage:
    python libby_auto_download.py "https://libbyapp.com/open/loan/XXXXX/XXXXX" output_folder
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


# JavaScript to inject into the iframe to capture data
CAPTURE_SCRIPT = """
(() => {
    if (window.__libbyCapture) return window.__libbyCapture;

    window.__libbyCapture = {
        authTokens: null,
        bifData: null,
        ready: false
    };

    // Hook JSON.parse to capture auth tokens
    const originalParse = JSON.parse;
    JSON.parse = function(...args) {
        const result = originalParse.apply(this, args);
        if (result && typeof result === 'object' && result.b && result.b['-odread-cmpt-params']) {
            window.__libbyCapture.authTokens = Array.from(result.b['-odread-cmpt-params']);
            console.log('[CAPTURE] Got auth tokens:', window.__libbyCapture.authTokens.length);
        }
        return result;
    };

    // Poll for BIF
    let checkCount = 0;
    let bifCheck = setInterval(() => {
        checkCount++;
        if (window.BIF && window.BIF.objects && window.BIF.objects.spool) {
            window.__libbyCapture.bifData = {
                title: BIF.map.title.main,
                creator: BIF.map.creator,
                chapters: BIF.map.nav?.toc || [],
                components: BIF.objects.spool.components.map(c => ({
                    path: c.meta.path,
                    position: c.spinePosition,
                    duration: c.meta["audio-duration"]
                }))
            };
            window.__libbyCapture.ready = true;
            console.log('[CAPTURE] Got BIF:', window.__libbyCapture.bifData.components.length, 'segments');
            clearInterval(bifCheck);
        }
        if (checkCount > 60) clearInterval(bifCheck);
    }, 500);

    return window.__libbyCapture;
})();
"""


def download_segment(url: str, output_path: Path, idx: int, total: int) -> tuple:
    """Download a single segment using curl."""
    try:
        result = subprocess.run(
            ["curl", "-sS", "-L", "-o", str(output_path), url],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            return (idx, True, output_path.stat().st_size)
        return (idx, False, result.stderr or "Small/empty file")
    except Exception as e:
        return (idx, False, str(e))


def merge_segments(segment_dir: Path, output_file: Path, metadata: dict):
    """Merge all segments using ffmpeg."""
    segments = sorted(segment_dir.glob("segment_*.mp3"))
    if not segments:
        print("No segments to merge")
        return False

    concat_file = segment_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in segments:
            f.write(f"file '{seg.name}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        "-metadata", f"title={metadata.get('title', 'Audiobook')}",
        "-metadata", f"artist={metadata.get('narrator', 'Unknown')}",
        "-metadata", f"album={metadata.get('title', 'Audiobook')}",
        str(output_file)
    ]

    print(f"\nMerging {len(segments)} segments...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_file.unlink()

    return result.returncode == 0


def run_download(libby_url: str, output_dir: Path):
    """Main download function."""

    print("=" * 60)
    print("Libby Auto-Downloader")
    print("=" * 60)
    print("\nThis will open a browser. If you're not logged into Libby,")
    print("you'll need to log in with your library card.\n")

    with sync_playwright() as p:
        # Use Playwright's built-in Chromium (avoids Chrome profile issues)
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context()

        # Add init script to ALL frames (including iframes) BEFORE they load
        context.add_init_script(CAPTURE_SCRIPT)

        page = context.new_page()

        print(f"Navigating to: {libby_url}")
        page.goto(libby_url)

        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        print("1. If prompted, log in with your library card")
        print("2. Click 'Listen' or 'Open' to start the audiobook player")
        print("3. Wait for the audio to start playing")
        print("4. The script will automatically detect and download")
        print("=" * 60 + "\n")

        # Wait for the listen.libbyapp.com iframe to appear
        print("Waiting for audiobook player to load...")

        max_wait = 300  # 5 minutes for login + navigation
        start = time.time()
        capture_data = None
        iframe_found = False
        injection_done = False

        while time.time() - start < max_wait:
            try:
                frames = page.frames

                for frame in frames:
                    if "listen.libbyapp.com" in frame.url:
                        if not iframe_found:
                            print("  Found audiobook player iframe!")
                            iframe_found = True

                        # Inject capture script if not done
                        if not injection_done:
                            try:
                                frame.evaluate(CAPTURE_SCRIPT)
                                injection_done = True
                                print("  Injected capture script. Waiting for data...")
                            except:
                                pass

                        # Check for captured data
                        if injection_done:
                            try:
                                data = frame.evaluate("window.__libbyCapture")
                                if data and data.get("ready") and data.get("authTokens"):
                                    capture_data = data
                                    print("  Data captured!")
                                    break
                            except:
                                pass

                if capture_data:
                    break

            except Exception as e:
                pass

            time.sleep(1)
            elapsed = int(time.time() - start)
            if elapsed % 15 == 0 and elapsed > 0:
                if not iframe_found:
                    print(f"  Waiting for player... ({elapsed}s) - Click 'Listen' if needed")
                elif not capture_data:
                    print(f"  Waiting for auth tokens... ({elapsed}s) - Try refreshing if stuck")

        if not capture_data:
            print("\n❌ Failed to capture audiobook data.")
            print("Make sure:")
            print("  - You're logged into Libby")
            print("  - The audiobook player is open and playing")
            print("  - Try refreshing the page after the player loads")
            browser.close()
            return False

        # Extract data
        bif = capture_data["bifData"]
        auth_tokens = capture_data["authTokens"]

        # Get origin from iframe
        origin = None
        for frame in page.frames:
            if "listen.libbyapp.com" in frame.url:
                origin = frame.evaluate("location.origin")
                break

        print("\n" + "=" * 60)
        print(f"Title: {bif['title']}")
        print(f"Segments: {len(bif['components'])}")
        print(f"Auth tokens: {len(auth_tokens)}")

        total_duration = sum(c["duration"] for c in bif["components"])
        hours = int(total_duration // 3600)
        mins = int((total_duration % 3600) // 60)
        print(f"Duration: {hours}h {mins}m")
        print("=" * 60)

        if len(auth_tokens) < len(bif["components"]):
            print(f"\n⚠️  Only {len(auth_tokens)} auth tokens for {len(bif['components'])} segments")
            print("Some downloads may fail.")

        browser.close()
        print("\nBrowser closed. Starting downloads...")

        # Create output directory
        book_name = "".join(c for c in bif["title"] if c.isalnum() or c in " -_").strip()
        book_dir = output_dir / book_name
        book_dir.mkdir(parents=True, exist_ok=True)

        segments_dir = book_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = {
            "title": bif["title"],
            "authors": [c["name"] for c in bif["creator"] if c.get("role") == "author"],
            "narrators": [c["name"] for c in bif["creator"] if c.get("role") == "narrator"],
            "chapters": [{"title": ch.get("title", ""), "path": ch.get("path", "")} for ch in bif["chapters"]],
            "duration": total_duration,
            "segments": len(bif["components"])
        }

        with open(book_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Build download URLs
        downloads = []
        for comp in sorted(bif["components"], key=lambda x: x["position"]):
            idx = comp["position"]
            if idx < len(auth_tokens):
                url = f"{origin}/{comp['path']}?{auth_tokens[idx]}"
                output_path = segments_dir / f"segment_{idx:03d}.mp3"
                downloads.append((url, output_path, idx))

        # Download in parallel
        print(f"\nDownloading {len(downloads)} segments...")
        success_count = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(download_segment, url, path, idx, len(downloads)): idx
                for url, path, idx in downloads
            }

            for future in as_completed(futures):
                idx, success, info = future.result()
                if success:
                    success_count += 1
                    print(f"  ✓ Segment {idx + 1}/{len(downloads)}")
                else:
                    print(f"  ✗ Segment {idx + 1}/{len(downloads)} failed: {info}")

        print(f"\nDownloaded {success_count}/{len(downloads)} segments")

        if success_count == 0:
            print("❌ All downloads failed")
            return False

        # Merge segments
        output_file = book_dir / f"{book_name}.mp3"
        if merge_segments(segments_dir, output_file, metadata):
            print(f"\n✅ Success! Audiobook saved to:")
            print(f"   {output_file}")

            # Cleanup segments
            for seg in segments_dir.glob("segment_*.mp3"):
                seg.unlink()
            try:
                segments_dir.rmdir()
            except:
                pass

            return True
        else:
            print(f"\n⚠️  Merge failed. Segments preserved in: {segments_dir}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Download audiobooks from Libby")
    parser.add_argument("url", help="Libby audiobook URL")
    parser.add_argument("output", nargs="?", default=".", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    success = run_download(args.url, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
