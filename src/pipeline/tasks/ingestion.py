"""
Ingestion Task — Video intake, validation, and metadata extraction.
Uses ffmpeg-python for deep probing and integrity checks.
"""
import os
import hashlib
import shutil
from pathlib import Path

import ffmpeg
from prefect import task

from src.core.config import settings
from src.core.logging import logger


def _compute_checksum(filepath: str, algorithm: str = "sha256") -> str:
    """Compute file hash for integrity verification."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _extract_metadata(filepath: str) -> dict:
    """Extract comprehensive video metadata using FFmpeg probe."""
    probe = ffmpeg.probe(filepath)

    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )

    if video_stream is None:
        raise ValueError(f"No video stream found in {filepath}")

    # Parse frame rate safely
    fps_str = video_stream.get("r_frame_rate", "24/1")
    num, den = map(int, fps_str.split("/"))
    fps = round(num / den, 2) if den != 0 else 24.0

    metadata = {
        "filename": os.path.basename(filepath),
        "filepath": os.path.abspath(filepath),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration_sec": float(probe["format"].get("duration", 0)),
        "total_frames": int(float(probe["format"].get("duration", 0)) * fps),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "pixel_format": video_stream.get("pix_fmt", "unknown"),
        "bitrate_kbps": int(probe["format"].get("bit_rate", 0)) // 1000,
        "file_size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "audio_sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
    }

    return metadata


def _validate_video(filepath: str) -> bool:
    """Run a quick decode test to ensure the file isn't corrupted."""
    try:
        # Decode first 10 seconds to check for corruption
        ffmpeg.input(filepath, t=10).output("pipe:", format="null").run(
            capture_stdout=True, capture_stderr=True
        )
        return True
    except ffmpeg.Error:
        return False


@task(name="Video Ingestion", retries=2, retry_delay_seconds=30)
def ingest_video(input_path: str) -> dict:
    """
    Full ingestion pipeline:
    1. Validate file exists and is readable
    2. Compute checksum for deduplication
    3. Extract deep metadata (resolution, FPS, codecs, duration)
    4. Run integrity check (decode test)
    5. Copy to managed storage
    """
    logger.info(f"Starting ingestion for: {input_path}")

    # --- Step 1: File Validation ---
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_ext = Path(input_path).suffix.lower()
    supported_formats = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".mpeg", ".mpg"}
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported format: {file_ext}. Supported: {supported_formats}")

    # --- Step 2: Checksum ---
    logger.info("Computing file checksum...")
    checksum = _compute_checksum(input_path)
    logger.info(f"SHA256: {checksum[:16]}...")

    # --- Step 3: Metadata Extraction ---
    logger.info("Extracting video metadata...")
    metadata = _extract_metadata(input_path)
    metadata["checksum_sha256"] = checksum

    logger.info(
        f"Video: {metadata['width']}x{metadata['height']} | "
        f"{metadata['fps']} FPS | {metadata['duration_sec']:.0f}s | "
        f"{metadata['video_codec']} | {metadata['file_size_mb']} MB"
    )

    # --- Step 4: Integrity Check ---
    logger.info("Running integrity check (decode test)...")
    is_valid = _validate_video(input_path)
    if not is_valid:
        raise RuntimeError(f"Video integrity check failed: {input_path}")
    metadata["integrity_check"] = "passed"
    logger.info("Integrity check: PASSED")

    # --- Step 5: Copy to Managed Storage ---
    raw_dir = Path(settings.RAW_STORAGE_PATH)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest_path = raw_dir / metadata["filename"]
    if not dest_path.exists():
        logger.info(f"Copying to managed storage: {dest_path}")
        shutil.copy2(input_path, dest_path)
    else:
        logger.info("File already in managed storage, skipping copy.")

    metadata["managed_path"] = str(dest_path)

    logger.info("=== Ingestion Complete ===")
    return metadata
