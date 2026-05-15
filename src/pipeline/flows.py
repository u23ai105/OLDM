from prefect import flow
from src.pipeline.tasks.ingestion import ingest_video
from src.pipeline.tasks.restoration import restore_video
from src.pipeline.tasks.upscaling import upscale_video
from src.pipeline.tasks.shorts_generation import generate_shorts
from src.core.logging import logger

@flow(name="AI Cinematic Restoration & Viral Shorts")
def production_pipeline(input_path: str, colorize: bool = True, scale: int = 4):
    """
    Main entry point for the restoration and shorts generation pipeline.
    """
    logger.info(f"--- Starting Production Pipeline for {input_path} ---")
    
    # 1. Ingestion & Metadata
    metadata = ingest_video(input_path)
    
    # 2. Restoration (DeOldify / Denoising)
    restored_path = restore_video(input_path, use_colorization=colorize)
    
    # 3. Upscaling (Real-ESRGAN)
    upscaled_path = upscale_video(restored_path, scale_factor=scale)
    
    # 4. Viral Shorts Generation (AI Reframe)
    shorts = generate_shorts(upscaled_path)
    
    logger.info(f"--- Pipeline Completed Successfully ---")
    logger.info(f"Main Video: {upscaled_path}")
    logger.info(f"Shorts Created: {len(shorts)}")
    
    return {
        "metadata": metadata,
        "video": upscaled_path,
        "shorts": shorts
    }

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_movie.mp4"
    production_pipeline(input_path=input_file)
