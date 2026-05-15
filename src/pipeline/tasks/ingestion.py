import ffmpeg
from prefect import task
from src.core.logging import logger
import os

@task(name="Advanced Ingestion", retries=2)
def ingest_video(input_path: str):
    """
    Extracts deep metadata and validates video integrity.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Probing video: {input_path}")
    
    try:
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        metadata = {
            "width": int(video_stream['width']),
            "height": int(video_stream['height']),
            "duration": float(probe['format']['duration']),
            "fps": eval(video_stream['r_frame_rate']),
            "codec": video_stream['codec_name'],
            "bitrate": int(probe['format'].get('bit_rate', 0)),
            "filename": os.path.basename(input_path)
        }
        
        logger.info(f"Ingestion successful: {metadata['width']}x{metadata['height']} @ {metadata['fps']} FPS")
        return metadata
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg probe failed: {e.stderr.decode()}")
        raise
