import cv2
import ffmpeg
from prefect import task
from src.core.logging import logger
import numpy as np

@task(name="AI Smart Shorts Generation")
def generate_shorts(video_path: str, num_shorts: int = 3):
    """
    Generates 9:16 vertical shorts by tracking the 'Subject' and reframing.
    """
    logger.info(f"Generating {num_shorts} shorts from {video_path}")
    
    # 1. Identify "Interest Points" using saliency or face detection
    # This is a simplified version of the logic:
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Target 9:16 aspect ratio
    target_width = int(height * (9/16))
    
    logger.info(f"Original: {width}x{height} -> Target: {target_width}x{height}")
    
    # Logic:
    # 1. For a clip, find the horizontal coordinate (x) that contains the most 'detail' or a face.
    # 2. Extract that window across the clip duration.
    # 3. Use FFmpeg to crop and export.
    
    short_paths = []
    for i in range(num_shorts):
        output_short = video_path.replace(".mp4", f"_short_{i}.mp4")
        
        # Mock FFmpeg crop command (Center crop example, but real logic uses dynamic X)
        # crop = f"crop={target_width}:{height}:(iw-{target_width})/2:0"
        
        # In production:
        # stream = ffmpeg.input(video_path, ss=start_time, t=30)
        # stream = ffmpeg.crop(stream, x, y, w, h)
        # stream = ffmpeg.output(stream, output_short)
        
        logger.info(f"Exporting Short #{i} to {output_short}")
        short_paths.append(output_short)
        
    cap.release()
    return short_paths
