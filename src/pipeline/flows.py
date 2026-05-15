"""
Production Pipeline Flow — Ties together all 4 core modules.
Ingestion → Restoration → Upscaling → Shorts Generation
"""
import sys
from pathlib import Path

from prefect import flow

from src.core.config import settings
from src.core.logging import logger
from src.pipeline.tasks.ingestion import ingest_video
from src.pipeline.tasks.restoration import restore_video
from src.pipeline.tasks.shorts_generation import generate_shorts
from src.pipeline.tasks.upscaling import upscale_video


@flow(name="AI Cinematic Restoration & Viral Shorts", log_prints=True)
def production_pipeline(
    input_path: str,
    colorize: bool = True,
    scale: int = None,
    num_shorts: int = None,
):
    """
    Main entry point for the full restoration pipeline.

    Args:
        input_path: Path to the source video file.
        colorize: Whether to apply DeOldify colorization.
        scale: Upscale factor (default from config: 4x).
        num_shorts: Number of shorts to generate (default from config: 5).
    """
    scale = scale or settings.ESRGAN_SCALE
    num_shorts = num_shorts or settings.SHORTS_NUM_CLIPS

    logger.info("=" * 60)
    logger.info(f"  AI Cinematic Restoration Pipeline")
    logger.info(f"  Input:      {input_path}")
    logger.info(f"  Colorize:   {colorize}")
    logger.info(f"  Upscale:    {scale}x")
    logger.info(f"  Shorts:     {num_shorts}")
    logger.info(f"  GPU VRAM:   {settings.VRAM_LIMIT_GB}GB allocated")
    logger.info("=" * 60)

    # ── Stage 1: Ingestion ──────────────────────────────────
    metadata = ingest_video(input_path)
    logger.info(
        f"Ingested: {metadata['width']}x{metadata['height']} | "
        f"{metadata['duration_sec']:.0f}s | {metadata['file_size_mb']}MB"
    )

    # ── Stage 2: Restoration & Colorization ─────────────────
    restored_path = restore_video(
        metadata["managed_path"],
        use_colorization=colorize,
    )

    # ── Stage 3: Super-Resolution Upscaling ─────────────────
    upscaled_path = upscale_video(restored_path, scale_factor=scale)

    # ── Stage 4: AI Shorts Generation ───────────────────────
    shorts = generate_shorts(upscaled_path, num_shorts=num_shorts)

    # ── Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Pipeline Complete!")
    logger.info(f"  Restored Video:  {restored_path}")
    logger.info(f"  Upscaled Video:  {upscaled_path}")
    logger.info(f"  Shorts Created:  {len(shorts)}")
    for i, s in enumerate(shorts):
        logger.info(f"    Short #{i}: {s}")
    logger.info("=" * 60)

    return {
        "metadata": metadata,
        "restored": restored_path,
        "upscaled": upscaled_path,
        "shorts": shorts,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.flows <video_path>")
        print("Example: python -m src.pipeline.flows data/raw/night_of_the_living_dead.mp4")
        sys.exit(1)

    input_file = sys.argv[1]
    production_pipeline(input_path=input_file)
