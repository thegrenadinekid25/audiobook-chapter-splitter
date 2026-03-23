// Libby Audiobook Extractor
// Run this in the browser console while on listen.libbyapp.com
// Then save the output and use libby_download_from_json.py to download

(async function() {
    if (typeof BIF === 'undefined') {
        alert('BIF not found. Make sure you are on the listen.libbyapp.com page with an audiobook playing.');
        return;
    }

    // Get the odread params from the page
    let odreadParams = null;
    const originalParse = JSON.parse;

    // They're already loaded, we need to find them in the BIF
    const components = BIF.objects.spool.components;

    // Build URLs by examining network requests or extracting from components
    const spineToIndex = BIF.map.spine.map(x => x["-odread-original-path"]);

    const metadata = {
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
        components: components.map(c => ({
            path: c.meta.path,
            position: c.spinePosition,
            duration: c.meta["audio-duration"],
            size: c.meta["-odread-file-bytes"],
            type: c.meta["media-type"]
        })),
        origin: location.origin
    };

    // Now we need to get the actual URLs with auth params
    // These are constructed from odread-cmpt-params which we need to extract

    console.log('='.repeat(60));
    console.log('LIBBY AUDIOBOOK DATA');
    console.log('='.repeat(60));
    console.log(`Title: ${metadata.title}`);
    console.log(`Segments: ${metadata.components.length}`);
    console.log(`Chapters: ${metadata.chapters.length}`);

    // Calculate total duration
    const totalDuration = metadata.components.reduce((sum, c) => sum + c.duration, 0);
    const hours = Math.floor(totalDuration / 3600);
    const minutes = Math.floor((totalDuration % 3600) / 60);
    console.log(`Duration: ${hours}h ${minutes}m`);
    console.log('='.repeat(60));

    // Try to download each segment directly
    console.log('\nAttempting to get download URLs...');

    const urls = [];
    for (const comp of components) {
        // The URL is constructed from the component path + auth params
        // Auth params are in the component's internal state
        const url = location.origin + '/' + comp.meta.path;

        // Try to fetch with credentials to see if it works
        try {
            const testResp = await fetch(url, { method: 'HEAD', credentials: 'include' });
            if (testResp.ok) {
                urls.push({
                    index: comp.spinePosition,
                    url: url,
                    duration: comp.meta["audio-duration"]
                });
            }
        } catch (e) {
            console.log(`Segment ${comp.spinePosition} needs auth params`);
        }
    }

    if (urls.length === components.length) {
        console.log('\nAll URLs accessible! Starting download...');

        // Download all segments
        for (const item of urls) {
            console.log(`Downloading segment ${item.index + 1}/${urls.length}...`);
            try {
                const response = await fetch(item.url, { credentials: 'include' });
                const blob = await response.blob();

                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `segment_${String(item.index).padStart(3, '0')}.mp3`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);

                // Small delay between downloads
                await new Promise(r => setTimeout(r, 500));
            } catch (e) {
                console.error(`Failed to download segment ${item.index}:`, e);
            }
        }

        // Also save metadata
        const metaBlob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
        const metaA = document.createElement('a');
        metaA.href = URL.createObjectURL(metaBlob);
        metaA.download = 'metadata.json';
        document.body.appendChild(metaA);
        metaA.click();
        document.body.removeChild(metaA);

        console.log('\nDownload complete! Check your Downloads folder.');
    } else {
        console.log('\nURLs need authentication. Saving metadata for manual processing...');
        console.log('Copy the JSON below and save it to a file:');
        console.log(JSON.stringify(metadata, null, 2));
    }
})();
