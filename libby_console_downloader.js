// Libby Audiobook Downloader
//
// INSTRUCTIONS:
// 1. Go to your audiobook on libbyapp.com
// 2. Open DevTools (Cmd+Option+J)
// 3. Paste this ENTIRE script and press Enter
// 4. Click "Listen" or refresh the page if already on the player
// 5. Once loaded, type: download() in the console
//
// The script hooks into Libby to capture the authentication tokens,
// then downloads all audio segments.

(function() {
    console.log('='.repeat(50));
    console.log('Libby Downloader Installed');
    console.log('='.repeat(50));

    // Storage for auth params
    window._libbyAuth = null;

    // Hook JSON.parse to capture auth tokens
    const originalParse = JSON.parse;
    JSON.parse = function(...args) {
        const result = originalParse.apply(this, args);

        // Capture odread-cmpt-params when they appear
        if (result && typeof result === 'object' && result.b && result.b['-odread-cmpt-params']) {
            window._libbyAuth = Array.from(result.b['-odread-cmpt-params']);
            console.log(`[Libby Downloader] Captured auth for ${window._libbyAuth.length} segments`);
        }

        return result;
    };

    // Main download function
    window.download = async function() {
        if (!window.BIF) {
            console.error('BIF not found. Make sure you are on listen.libbyapp.com with audio playing.');
            return;
        }

        if (!window._libbyAuth) {
            console.error('Auth tokens not captured. Refresh the page and try again.');
            return;
        }

        const title = BIF.map.title.main;
        const components = BIF.objects.spool.components;
        const auth = window._libbyAuth;

        const totalDuration = components.reduce((sum, c) => sum + c.meta["audio-duration"], 0);
        const hours = Math.floor(totalDuration / 3600);
        const mins = Math.floor((totalDuration % 3600) / 60);

        console.log('\n' + '='.repeat(50));
        console.log(`Title: ${title}`);
        console.log(`Segments: ${components.length}`);
        console.log(`Duration: ${hours}h ${mins}m`);
        console.log(`Auth tokens: ${auth.length}`);
        console.log('='.repeat(50) + '\n');

        if (auth.length < components.length) {
            console.warn('Warning: Fewer auth tokens than segments. Some downloads may fail.');
        }

        // Build URLs
        const downloads = components
            .sort((a, b) => a.spinePosition - b.spinePosition)
            .map(c => ({
                index: c.spinePosition,
                url: location.origin + '/' + c.meta.path + '?' + (auth[c.spinePosition] || ''),
                duration: c.meta["audio-duration"],
                filename: `segment_${String(c.spinePosition).padStart(3, '0')}.mp3`
            }));

        // Save metadata
        const metadata = {
            title: title,
            authors: BIF.map.creator.filter(c => c.role === 'author').map(c => c.name),
            narrators: BIF.map.creator.filter(c => c.role === 'narrator').map(c => c.name),
            chapters: BIF.map.nav?.toc?.map(ch => ({ title: ch.title, path: ch.path })) || [],
            totalDuration: totalDuration,
            segments: downloads.map(d => ({ index: d.index, duration: d.duration, filename: d.filename }))
        };

        // Download metadata.json
        downloadFile(JSON.stringify(metadata, null, 2), 'metadata.json', 'application/json');

        // Download files.txt for ffmpeg
        const filesList = downloads.map(d => `file '${d.filename}'`).join('\n');
        downloadFile(filesList, 'files.txt', 'text/plain');

        console.log('Downloading segments... (allow multiple downloads if prompted)\n');

        // Download each segment
        let success = 0, failed = 0;
        for (const d of downloads) {
            try {
                process.stdout?.write?.(`\r[${d.index + 1}/${downloads.length}] Downloading...`) ||
                    console.log(`[${d.index + 1}/${downloads.length}] Downloading segment ${d.index}...`);

                const response = await fetch(d.url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const blob = await response.blob();
                downloadBlob(blob, d.filename);
                success++;

                // Delay between downloads
                await sleep(400);

            } catch (e) {
                console.error(`  Segment ${d.index} failed: ${e.message}`);
                failed++;
            }
        }

        console.log('\n' + '='.repeat(50));
        console.log(`Complete! ${success} downloaded, ${failed} failed`);
        console.log('='.repeat(50));
        console.log('\nNext steps:');
        console.log('1. Move all downloaded files to a folder');
        console.log('2. Run: ffmpeg -f concat -safe 0 -i files.txt -c copy "audiobook.mp3"');
        console.log('\nOr use audiobook_splitter.py for chapter detection.');
    };

    // Helper functions
    function downloadFile(content, filename, type) {
        const blob = new Blob([content], { type });
        downloadBlob(blob, filename);
    }

    function downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function sleep(ms) {
        return new Promise(r => setTimeout(r, ms));
    }

    console.log('\nINSTRUCTIONS:');
    console.log('1. Navigate to your audiobook (listen.libbyapp.com)');
    console.log('2. Wait for it to load and start playing');
    console.log('3. Type: download()');
    console.log('\nIf you see "Auth tokens not captured", refresh the page and try again.');
    console.log('='.repeat(50));
})();
