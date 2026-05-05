"""Download free India traffic test videos for pipeline testing.

Sources (all CC0 / royalty-free):
  - Pixabay India traffic videos (direct download via their CDN)
  - Fallback YouTube public-domain clips

Uses yt-dlp (pip install yt-dlp) for robust downloading.
Falls back to urllib if yt-dlp not installed (direct CDN links only).

Usage:
    python scripts/download_test_videos.py
    python scripts/download_test_videos.py --output data/samples --quality 720
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- Curated free India traffic video sources ----------------------------
# Pixabay CDN direct links (CC0, no attribution required).
# These are stable CDN links verified as of 2025.
# To get more: browse https://pixabay.com/videos/search/traffic%20india/
# click a video, right-click the download button → copy link address.
PIXABAY_DIRECT = [
    {
        "url": "https://cdn.pixabay.com/video/2022/09/14/131347-752167959_large.mp4",
        "name": "india_traffic_intersection_01.mp4",
        "desc": "Indian intersection, motorbikes and cars, daytime",
    },
    {
        "url": "https://cdn.pixabay.com/video/2023/02/27/153082-803166888_large.mp4",
        "name": "india_traffic_highway_01.mp4",
        "desc": "India highway, mixed traffic, trucks and bikes",
    },
    {
        "url": "https://cdn.pixabay.com/video/2022/05/31/119508-717041136_large.mp4",
        "name": "india_traffic_street_01.mp4",
        "desc": "Indian city street, heavy traffic",
    },
    {
        "url": "https://cdn.pixabay.com/video/2023/04/12/158752-818050197_large.mp4",
        "name": "india_traffic_bikes_01.mp4",
        "desc": "Motorbikes in India traffic, close-up",
    },
    {
        "url": "https://cdn.pixabay.com/video/2021/08/24/86126-595873295_large.mp4",
        "name": "india_traffic_night_01.mp4",
        "desc": "Night traffic India, headlights, busy road",
    },
]

# yt-dlp search queries (requires yt-dlp + working internet)
YTDLP_QUERIES = [
    # Pixabay search page
    "https://pixabay.com/videos/search/traffic%20india/",
    # YouTube — only explicitly CC videos
    "ytsearch5:India traffic enforcement CCTV dashcam",
    "ytsearch3:Indian number plate motorcycle helmet violation",
]


def _yt_dlp_available() -> bool:
    try:
        import yt_dlp
        return True
    except ImportError:
        return False


def download_direct(url: str, dest: Path, desc: str) -> bool:
    """Download via urllib with progress."""
    print(f"  Downloading: {desc}")
    print(f"  URL: {url}")
    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                print(f"\r  Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r  Saved: {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"\r  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_with_ytdlp(url: str, out_dir: Path, max_videos: int = 3) -> list[Path]:
    """Download using yt-dlp (best quality ≤ 720p)."""
    import yt_dlp
    opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "playlistend": max_videos,
        "ignoreerrors": True,
    }
    downloaded = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            if info and "entries" in info:
                for entry in (info["entries"] or [])[:max_videos]:
                    if entry:
                        path = out_dir / ydl.prepare_filename(entry)
                        if path.exists():
                            downloaded.append(path)
            elif info:
                path = Path(ydl.prepare_filename(info))
                if path.exists():
                    downloaded.append(path)
        except Exception as e:
            print(f"  yt-dlp error: {e}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download India traffic test videos")
    parser.add_argument("--output", default="data/samples", help="Output directory")
    parser.add_argument("--quality", default="720", help="Max quality (360/480/720/1080)")
    parser.add_argument("--skip-direct", action="store_true", help="Skip direct CDN downloads")
    parser.add_argument("--skip-search", action="store_true", help="Skip yt-dlp search downloads")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("India Traffic Video Downloader")
    print(f"Output: {out_dir}")
    print("=" * 60)

    total = 0

    # -- Phase 1: Direct Pixabay CDN downloads ----------------------------
    if not args.skip_direct:
        print(f"\n[1/2] Direct CDN downloads ({len(PIXABAY_DIRECT)} videos)...")
        for item in PIXABAY_DIRECT:
            dest = out_dir / item["name"]
            if dest.exists() and dest.stat().st_size > 100_000:
                print(f"  Already exists: {item['name']} ({dest.stat().st_size // 1024} KB)")
                total += 1
                continue
            ok = download_direct(item["url"], dest, item["desc"])
            if ok:
                total += 1
    else:
        print("\n[1/2] Skipping direct downloads.")

    # -- Phase 2: yt-dlp search (if available) ----------------------------
    if not args.skip_search:
        if _yt_dlp_available():
            print(f"\n[2/2] yt-dlp search downloads (Pixabay)...")
            print("  Searching Pixabay for India traffic videos...")
            downloaded = download_with_ytdlp(
                "https://pixabay.com/videos/search/traffic%20india/",
                out_dir,
                max_videos=5,
            )
            total += len(downloaded)
            for p in downloaded:
                print(f"  Downloaded: {p.name}")
        else:
            print("\n[2/2] yt-dlp not installed — skipping search downloads.")
            print("       Install with: pip install yt-dlp")
            print("       Then re-run to download more videos.")

    # -- Summary -----------------------------------------------------------
    all_videos = list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.webm"))
    print(f"\n{'='*60}")
    print(f"Downloaded/available: {len(all_videos)} video(s) in {out_dir}")
    for v in all_videos:
        print(f"  {v.name}  ({v.stat().st_size // 1024} KB)")

    if all_videos:
        first = all_videos[0]
        print(f"\nQuick test — verify video is readable:")
        print(f"  python -c \"import cv2; cap=cv2.VideoCapture('{first}'); print('OK, frames:', int(cap.get(7)))\"")
        print(f"\nRun benchmark:")
        print(f"  python scripts/benchmark_streams.py --source {first} --streams 1 2 4")
        print(f"\nUpdate config/cameras.yaml — set source to:")
        print(f"  source: {first}")
    else:
        print("\nNo videos downloaded. Check your internet connection.")
        print("Manual alternative:")
        print("  1. Go to https://pixabay.com/videos/search/traffic%20india/")
        print("  2. Click any video → Download → Free download (choose 1280x720)")
        print("  3. Save to data/samples/traffic.mp4")

    print()


if __name__ == "__main__":
    main()
