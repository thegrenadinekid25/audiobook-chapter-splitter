# Audiobook Chapter Splitter

Download audiobooks from Libby and split them into individual chapter MP3s with full metadata.

## Features

- **End-to-end Libby pipeline** -- give a URL, get chapter files
- **Automatic chapter detection** from Libby metadata (instant, no transcription needed)
- **Whisper fallback** -- AI transcription for books without chapter data
- **ID3 metadata tagging** -- title, album, author, narrator, track numbers, genre
- **Cover art embedding** -- from Libby, URL, local file, or auto-fetch by ISBN
- **Parallel downloads** for speed
- Splits by chapters or fixed time segments (standalone mode)

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)
- playwright (`pip install playwright && playwright install chromium`)
- whisper-cpp (`brew install whisper-cpp`) -- only needed if Libby book lacks chapter data

## Installation

```bash
brew install ffmpeg
pip install playwright
playwright install chromium

# Clone and set up
git clone https://github.com/YOUR_USERNAME/audiobook-chapter-splitter.git
cd audiobook-chapter-splitter
chmod +x libby_audiobook.py
```

## Libby Pipeline (Recommended)

### Basic Usage

```bash
python libby_audiobook.py "https://libbyapp.com/open/loan/12345/67890"
```

This will:
1. Open a browser (uses your Chrome login session)
2. Capture the audiobook data when you click "Listen"
3. Download all audio segments in parallel
4. Split into individual chapter MP3s using Libby's chapter data
5. Tag each file with title, author, narrator, track number, and genre

### Browse and Pick

```bash
python libby_audiobook.py
```

Opens Libby so you can browse your library and pick a book.

### With Cover Art

```bash
# Auto-fetch from ISBN
python libby_audiobook.py "https://libbyapp.com/open/loan/..." \
    --isbn "9780062676788"

# From a URL or local file
python libby_audiobook.py "https://libbyapp.com/open/loan/..." \
    --cover /path/to/cover.jpg
```

### Custom Output Directory

```bash
python libby_audiobook.py "https://libbyapp.com/open/loan/..." \
    --output ~/Audiobooks
```

### Download Only (No Splitting)

```bash
python libby_audiobook.py "https://libbyapp.com/open/loan/..." --no-split
```

### Libby Pipeline Options

| Option | Description |
|--------|-------------|
| `--output/-o DIR` | Output directory (default: current) |
| `--cover URL/PATH` | Cover image URL or local file path |
| `--isbn ISBN` | Auto-fetch cover from Google Books by ISBN |
| `--year YEAR` | Publication year for metadata |
| `--no-split` | Download only, keep as single merged MP3 |
| `--force-whisper` | Use Whisper transcription even if Libby has chapters |
| `--keep-merged` | Keep the merged MP3 after splitting into chapters |
| `--workers N` | Parallel download workers (default: 4) |

### Output Structure

```
./Book Title/
  chapters/
    ch01_prologue.mp3
    ch02_chapter_one.mp3
    ch03_the_discovery.mp3
    ...
  metadata.json
  cover.jpg
```

Each chapter MP3 includes:
- Title (chapter name)
- Album (book title)
- Artist (narrator)
- Album Artist / Composer (author)
- Track number
- Genre (Audiobook)
- Cover art (embedded)

## Standalone Splitter

For audiobook MP3s you already have (not from Libby), the standalone splitter uses Whisper transcription to detect chapters:

```bash
python audiobook_splitter.py /path/to/audiobook/folder
python audiobook_splitter.py /path/to/audiobook/folder --segment-minutes 10
python audiobook_splitter.py /path/to/audiobook/folder --analyze-only
python audiobook_splitter.py /path/to/audiobook/folder \
    --album "Book Title" --author "Author" --narrator "Narrator"
```

### Standalone Options

| Option | Description |
|--------|-------------|
| `--segment-minutes N` | Split into N-minute segments instead of chapters |
| `--analyze-only` | Only detect chapters, don't split |
| `--parallel N` | Number of parallel transcription jobs (default: 4) |
| `--output DIR` | Custom output directory |
| `--album TITLE` | Book title for ID3 metadata |
| `--author NAME` | Author name |
| `--narrator NAME` | Narrator name |
| `--year YEAR` | Publication year |
| `--cover URL/PATH` | Cover image |
| `--isbn ISBN` | Auto-fetch cover by ISBN |

## How It Works

### Libby Pipeline
1. **Browser automation** (Playwright) opens Libby with your Chrome session
2. **BIF extraction** captures the book's metadata, chapter data, and auth tokens
3. **Parallel download** fetches all audio segments using curl
4. **Merge** concatenates segments into a single MP3 with ffmpeg
5. **Chapter splitting** uses publisher-provided chapter timestamps to split precisely
6. **Metadata tagging** adds ID3 tags and cover art to each chapter file

### Standalone Splitter (Whisper)
1. **Transcription**: MP3 -> WAV -> whisper-cpp -> SRT subtitles
2. **Chapter detection**: Regex scan for patterns like "1. Title" or "Chapter One"
3. **Splitting**: ffmpeg splits at detected chapter boundaries
4. **Tagging**: ID3 metadata + cover art applied to each file

## License

MIT
