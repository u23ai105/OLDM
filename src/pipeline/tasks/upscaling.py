"""
Upscaling Task — Real-ESRGAN 4K/8K super-resolution.
Processes video frame-by-frame through Real-ESRGAN with tiled inference
for memory-efficient processing on high-VRAM GPUs.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from prefect import task

from src.core.config import settings
from src.core.logging import logger


class RealESRGANUpscaler:
    """
    Real-ESRGAN video upscaler with batch processing and tiled inference.
    Tuned for high-VRAM GPUs (35GB allocation).
    """

    def __init__(
        self,
        weights_path: str,
        scale: int = 4,
        tile_size: int = 512,
        tile_pad: int = 32,
        batch_size: int = 8,
        fp16: bool = True,
        device: str = "cuda",
    ):
        self.weights_path = weights_path
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.batch_size = batch_size
        self.fp16 = fp16
        self.device = device
        self.upsampler = None

    def load_model(self):
        """Load Real-ESRGAN model."""
        logger.info(f"Loading Real-ESRGAN weights from {self.weights_path}")

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found at {self.weights_path}. "
                "Run: python scripts/download_weights.py realesrgan"
            )

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # RealESRGAN_x4plus uses a 6-block RRDB architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.scale,
            )

            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=self.weights_path,
                dni_weight=None,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=0,
                half=self.fp16,
                device=self.device,
            )

            logger.info(
                f"Real-ESRGAN loaded: scale={self.scale}x | "
                f"tile={self.tile_size} | fp16={self.fp16} | "
                f"batch={self.batch_size}"
            )

        except ImportError as e:
            logger.error(
                f"Failed to import Real-ESRGAN dependencies: {e}. "
                "Install: pip install realesrgan basicsr"
            )
            raise

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single frame using Real-ESRGAN."""
        output, _ = self.upsampler.enhance(frame, outscale=self.scale)
        return output

    def upscale_video(self, input_path: str, output_path: str) -> str:
        """
        Upscale an entire video frame-by-frame.
        Uses batch processing for GPU throughput optimization.
        """
        logger.info(f"Starting {self.scale}x upscale: {input_path}")

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w = orig_w * self.scale
        out_h = orig_h * self.scale

        logger.info(
            f"Resolution: {orig_w}x{orig_h} → {out_w}x{out_h} | "
            f"Frames: {total_frames} | FPS: {fps}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            upscaled_frames_dir = os.path.join(tmpdir, "upscaled")
            os.makedirs(upscaled_frames_dir)

            frame_idx = 0
            batch_buffer = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                batch_buffer.append((frame_idx, frame))

                # Process in batches for GPU efficiency
                if len(batch_buffer) >= self.batch_size:
                    self._process_batch(batch_buffer, upscaled_frames_dir)
                    batch_buffer = []

                frame_idx += 1

            # Process remaining frames
            if batch_buffer:
                self._process_batch(batch_buffer, upscaled_frames_dir)

            cap.release()

            # Log VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.max_memory_allocated() / (1024**3)
                logger.info(f"Peak VRAM usage: {vram_used:.1f} GB")

            # Reassemble video with FFmpeg
            logger.info("Reassembling upscaled video...")
            self._reassemble_video(
                upscaled_frames_dir, input_path, output_path, fps
            )

        logger.info(f"Upscaling complete: {output_path}")
        return output_path

    def _process_batch(self, batch: list, output_dir: str):
        """Process a batch of frames through Real-ESRGAN."""
        for frame_idx, frame in batch:
            try:
                upscaled = self.upscale_frame(frame)
                out_path = os.path.join(output_dir, f"frame_{frame_idx:08d}.png")
                cv2.imwrite(out_path, upscaled)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"CUDA OOM on frame {frame_idx}. "
                    "Clearing cache and retrying with smaller tile..."
                )
                torch.cuda.empty_cache()
                # Retry with half tile size
                original_tile = self.upsampler.tile
                self.upsampler.tile = original_tile // 2
                upscaled = self.upscale_frame(frame)
                self.upsampler.tile = original_tile
                out_path = os.path.join(output_dir, f"frame_{frame_idx:08d}.png")
                cv2.imwrite(out_path, upscaled)

        processed = batch[-1][0] + 1
        if processed % 100 == 0:
            logger.info(f"  Upscaled {processed} frames")

    def _reassemble_video(
        self,
        frames_dir: str,
        original_video: str,
        output_path: str,
        fps: float,
    ):
        """Reassemble upscaled frames into a video with original audio."""
        temp_video = output_path + ".tmp.mp4"

        # Encode frames to video
        cmd_video = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%08d.png"),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "16",            # Lower CRF for higher quality at 4K
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            temp_video,
        ]
        subprocess.run(cmd_video, check=True, capture_output=True)

        # Mux in original audio
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", original_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd_mux, check=True, capture_output=True)

        if os.path.exists(temp_video):
            os.remove(temp_video)


@task(name="4K/8K Super-Resolution Upscaling", retries=1)
def upscale_video(input_path: str, scale_factor: int = None) -> str:
    """
    Upscale a restored video to 4K/8K using Real-ESRGAN.
    Configured for 35GB VRAM with tiled inference and FP16.
    """
    scale = scale_factor or settings.ESRGAN_SCALE
    logger.info(f"=== Starting {scale}x Upscaling for {input_path} ===")

    device = "cuda" if (torch.cuda.is_available() and settings.USE_GPU) else "cpu"

    weights_path = os.path.join(
        settings.MODEL_WEIGHTS_PATH, "realesrgan", "RealESRGAN_x4plus.pth"
    )

    output_dir = Path(settings.PROCESSED_STORAGE_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = Path(input_path).stem.replace("_restored", "").replace("_denoised", "")
    output_path = str(output_dir / f"{basename}_upscaled_{scale}x.mp4")

    upscaler = RealESRGANUpscaler(
        weights_path=weights_path,
        scale=scale,
        tile_size=settings.ESRGAN_TILE_SIZE,
        tile_pad=settings.ESRGAN_TILE_PAD,
        batch_size=settings.ESRGAN_BATCH_SIZE,
        fp16=settings.ESRGAN_FP16,
        device=device,
    )
    upscaler.load_model()
    result_path = upscaler.upscale_video(input_path, output_path)

    logger.info(f"=== Upscaling Complete: {result_path} ===")
    return result_path
