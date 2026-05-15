"""
Model Weight Download Script
Downloads all required pretrained weights for the restoration pipeline.
"""
import os
import sys
import subprocess
import hashlib
from pathlib import Path

WEIGHTS_DIR = Path("./weights")

MODELS = {
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "subdir": "realesrgan",
        "sha256": None,  # Skip hash check for speed
    },
    "realesrgan_anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "subdir": "realesrgan",
        "sha256": None,
    },
    "deoldify_video": {
        "url": "https://data.deepai.org/deoldify/ColorizeVideo_gen.pth",
        "filename": "ColorizeVideo_gen.pth",
        "subdir": "deoldify",
        "sha256": None,
    },
    "deoldify_artistic": {
        "url": "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        "filename": "ColorizeArtistic_gen.pth",
        "subdir": "deoldify",
        "sha256": None,
    },
    "face_detection_deploy": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "filename": "deploy.prototxt",
        "subdir": "face_detector",
        "sha256": None,
    },
    "face_detection_weights": {
        "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "filename": "res10_300x300_ssd_iter_140000.caffemodel",
        "subdir": "face_detector",
        "sha256": None,
    },
}


def download_file(url: str, dest: Path):
    """Download a file using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  ✓ Already exists: {dest}")
        return

    print(f"  ↓ Downloading: {url}")
    print(f"    → {dest}")

    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
    except FileNotFoundError:
        # wget not available, try curl
        subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            check=True,
        )

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✓ Downloaded: {size_mb:.1f} MB")


def main():
    print("=" * 60)
    print("  AI Cinematic Restoration — Model Weight Downloader")
    print("=" * 60)
    print()

    # Allow selective download
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    for name, info in MODELS.items():
        if target != "all" and target not in name:
            continue

        dest = WEIGHTS_DIR / info["subdir"] / info["filename"]
        print(f"[{name}]")
        download_file(info["url"], dest)
        print()

    print("=" * 60)
    print("  All weights downloaded successfully!")
    print(f"  Location: {WEIGHTS_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
