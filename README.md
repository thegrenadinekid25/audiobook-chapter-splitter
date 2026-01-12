# Audiobook Chapter Splitter

Automatically detect and split audiobook MP3 files by chapters using Whisper speech-to-text.

## Features

- Automatic chapter detection using AI transcription
- Handles audiobooks with files in random order
- Detects duplicate files
- Splits by chapters or fixed time segments
- Parallel transcription for speed
- Caches transcriptions for re-runs

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.10+
- ffmpeg
- whisper-cpp

## Installation

```bash
# Install dependencies
brew install ffmpeg whisper-cpp

# Clone the repo
git clone https://github.com/YOUR_USERNAME/audiobook-chapter-splitter.git
cd audiobook-chapter-splitter

# Make executable
chmod +x audiobook_splitter.py
```

The Whisper model (~140MB) is downloaded automatically on first run.

## Usage

### Split by Chapters (Automatic Detection)

```bash
python audiobook_splitter.py /path/to/audiobook/folder
```

This will:
1. Transcribe all MP3 files using Whisper
2. Detect chapter markers from the transcription
3. Split the audio at chapter boundaries
4. Output to `chapters/` subfolder

### Split by Fixed Duration

```bash
python audiobook_splitter.py /path/to/audiobook/folder --segment-minutes 10
```

Creates 10-minute segments without chapter detection.

### Analyze Only (No Splitting)

```bash
python audiobook_splitter.py /path/to/audiobook/folder --analyze-only
```

Shows detected chapters and saves analysis to `chapter_analysis.json`.

### Custom Output Directory

```bash
python audiobook_splitter.py /path/to/audiobook/folder --output /path/to/output
```

## Options

| Option | Description |
|--------|-------------|
| `--segment-minutes N` | Split into N-minute segments instead of chapters |
| `--analyze-only` | Only detect chapters, don't split |
| `--parallel N` | Number of parallel transcription jobs (default: 4) |
| `--output DIR` | Custom output directory |

## How It Works

1. **Transcription**: Each MP3 file is converted to WAV and transcribed using whisper-cpp
2. **Chapter Detection**: The transcription is scanned for chapter markers like "1. Chapter Title" or "Chapter One"
3. **Timestamp Extraction**: Chapter start times are extracted from the SRT subtitle format
4. **Splitting**: ffmpeg splits the audio at detected chapter boundaries

## Limitations

- Works best with audiobooks that have clearly announced chapter titles
- Transcription quality depends on audio clarity
- Very long audiobooks may take significant time to transcribe

## Performance

On Apple M2:
- Transcription: ~3-4x realtime (1 hour of audio in ~15-20 minutes)
- Parallel processing: 8 one-hour files in ~20-25 minutes

## Example Output

```
Found 9 MP3 files

Transcribing audio files...
  [1] Converting to WAV...
  [1] Transcribing with Whisper...
  [1] Done!
  ...

Detected chapters:
------------------------------------------------------------
  Ch  1: Funeral Plans                    (Part 1 @ 00:07)
  Ch  2: Two Weeks Earlier                (Part 1 @ 12:51)
  Ch  3: Chapter One                      (Part 1 @ 46:53)
  ...

Splitting into chapters...
  Creating: ch01 - Funeral Plans
  Creating: ch02 - Two Weeks Earlier
  ...

Done! 24 chapters created in: /path/to/audiobook/chapters
```

## License

MIT
