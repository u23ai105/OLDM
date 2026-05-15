"""
AI Smart Shorts Generation — Saliency-based vertical reframing.

Takes a restored/upscaled horizontal (16:9) video and generates
multiple vertical (9:16) shorts optimized for YouTube Shorts / TikTok / Reels.

Pipeline:
1. Scene change detection → split into candidate segments
2. Face/saliency detection per segment → score each segment
3. Top-N segments selected by "interest score"
4. Dynamic crop window tracks the subject (Pan & Scan)
5. Kalman filter smoothing to prevent jitter
6. FFmpeg export of each short with mobile-optimized encoding
"""
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from prefect import task

from src.core.config import settings
from src.core.logging import logger


class FaceDetector:
    """OpenCV DNN-based face detector for saliency tracking."""

    def __init__(self, weights_dir: str):
        prototxt = os.path.join(weights_dir, "face_detector", "deploy.prototxt")
        caffemodel = os.path.join(
            weights_dir, "face_detector",
            "res10_300x300_ssd_iter_140000.caffemodel",
        )

        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            self.available = True
            logger.info("Face detector loaded (OpenCV DNN SSD)")
        else:
            self.net = None
            self.available = False
            logger.warning(
                "Face detector weights not found. Falling back to center crop. "
                "Run: python scripts/download_weights.py face_detection"
            )

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if not self.available:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces


class KalmanSmoother:
    """
    Kalman filter for smooth pan & scan crop window tracking.
    Prevents jittery movement when the subject moves between frames.
    """

    def __init__(self):
        # State: [x_position, x_velocity]
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.array([[1]], dtype=np.float32) * 5.0
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        self.initialized = False

    def update(self, x_center: int) -> int:
        """Update the filter with a new measurement and return smoothed position."""
        measurement = np.array([[x_center]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array([[x_center], [0]], dtype=np.float32)
            self.initialized = True
            return x_center

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return int(corrected[0, 0])


class SaliencyScorer:
    """Scores video segments by visual interest (motion + faces)."""

    def __init__(self, face_detector: FaceDetector):
        self.face_detector = face_detector

    def score_segment(
        self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, sample_rate: int = 15
    ) -> float:
        """
        Score a video segment by combining:
        - Face presence (weighted 0.6)
        - Motion energy (weighted 0.4)
        """
        face_score = 0.0
        motion_score = 0.0
        samples = 0
        prev_gray = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame, sample_rate):
            ret, frame = cap.read()
            if not ret:
                break

            samples += 1

            # Face detection score
            faces = self.face_detector.detect(frame, confidence_threshold=0.4)
            if faces:
                face_score += 1.0

            # Motion energy (frame differencing)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_energy = np.mean(diff) / 255.0
                motion_score += motion_energy
            prev_gray = gray

        if samples == 0:
            return 0.0

        # Normalize and combine
        face_ratio = face_score / samples
        motion_avg = motion_score / max(samples - 1, 1)

        combined = (face_ratio * 0.6) + (motion_avg * 0.4)
        return round(combined, 4)


class SmartReframer:
    """
    Generates a 9:16 vertical crop from a 16:9 horizontal video
    using face tracking with Kalman smoothing.
    """

    def __init__(self, face_detector: FaceDetector):
        self.face_detector = face_detector

    def reframe_segment(
        self,
        input_path: str,
        output_path: str,
        start_sec: float,
        duration_sec: float,
    ) -> str:
        """
        Extract a segment and apply AI-driven vertical reframing.
        1. Detect faces/subjects frame-by-frame
        2. Track the X-center of the subject
        3. Smooth the crop window movement with Kalman filter
        4. Export the vertical crop
        """
        logger.info(
            f"Reframing: start={start_sec:.1f}s duration={duration_sec:.1f}s"
        )

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 9:16 crop dimensions (full height, calculated width)
        crop_w = int(orig_h * (9 / 16))
        crop_h = orig_h

        # Ensure crop width doesn't exceed original width
        if crop_w > orig_w:
            crop_w = orig_w

        logger.info(f"Crop window: {crop_w}x{crop_h} from {orig_w}x{orig_h}")

        # Seek to start position
        start_frame = int(start_sec * fps)
        end_frame = int((start_sec + duration_sec) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # First pass: compute crop X positions using face detection
        crop_positions = []
        smoother = KalmanSmoother()

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.face_detector.detect(frame)

            if faces:
                # Use the largest face as the anchor
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest_face
                face_center_x = fx + fw // 2
            else:
                # No face → default to center
                face_center_x = orig_w // 2

            # Smooth the position
            smoothed_x = smoother.update(face_center_x)

            # Calculate crop left coordinate (clamped to bounds)
            crop_left = max(0, min(smoothed_x - crop_w // 2, orig_w - crop_w))
            crop_positions.append(crop_left)

        cap.release()

        # If no dynamic data, use center crop
        if not crop_positions:
            crop_positions = [(orig_w - crop_w) // 2]

        # Determine the dominant crop position (median for stability)
        median_x = int(np.median(crop_positions))

        # Second pass: use FFmpeg to extract the crop
        # For most Shorts, a stable crop position looks more professional
        # than a constantly panning camera
        self._export_crop(
            input_path, output_path,
            start_sec, duration_sec,
            median_x, 0, crop_w, crop_h,
            fps,
        )

        return output_path

    def _export_crop(
        self,
        input_path: str,
        output_path: str,
        start_sec: float,
        duration_sec: float,
        x: int,
        y: int,
        w: int,
        h: int,
        fps: float,
    ):
        """Export a cropped segment using FFmpeg with mobile-optimized encoding."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", input_path,
            "-t", str(duration_sec),
            "-vf", (
                f"crop={w}:{h}:{x}:{y},"
                f"scale=1080:1920:flags=lanczos,"
                f"unsharp=5:5:0.5:5:5:0.0"  # Subtle sharpen for mobile screens
            ),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)


def _detect_scene_changes(
    video_path: str, threshold: float = 30.0, min_scene_sec: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Detect scene changes using frame difference thresholding.
    Returns list of (start_sec, end_sec) for each scene.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(f"Detecting scene changes (threshold={threshold})...")

    scene_boundaries = [0.0]
    prev_gray = None
    frame_idx = 0

    # Sample every 5th frame for speed
    sample_interval = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = np.mean(diff)
                if mean_diff > threshold:
                    timestamp = frame_idx / fps
                    # Enforce minimum scene length
                    if timestamp - scene_boundaries[-1] >= min_scene_sec:
                        scene_boundaries.append(timestamp)
            prev_gray = gray

        frame_idx += 1

    cap.release()

    # Add final boundary
    scene_boundaries.append(duration)

    # Build scene list
    scenes = []
    for i in range(len(scene_boundaries) - 1):
        start = scene_boundaries[i]
        end = scene_boundaries[i + 1]
        if end - start >= min_scene_sec:
            scenes.append((start, end))

    logger.info(f"Detected {len(scenes)} scenes")
    return scenes


@task(name="AI Smart Shorts Generation", retries=1)
def generate_shorts(video_path: str, num_shorts: int = None) -> List[str]:
    """
    Generate vertical 9:16 shorts from a horizontal video.

    Pipeline:
    1. Detect scene changes → candidate segments
    2. Score each segment by face presence + motion energy
    3. Select top-N highest scoring segments
    4. Smart reframe each segment (face tracking + Kalman smoothing)
    5. Export mobile-optimized vertical clips
    """
    num_shorts = num_shorts or settings.SHORTS_NUM_CLIPS
    duration = settings.SHORTS_DURATION_SEC
    min_score = settings.SHORTS_MIN_SCORE

    logger.info(f"=== Generating {num_shorts} AI Shorts from {video_path} ===")

    output_dir = Path(settings.PROCESSED_STORAGE_PATH) / "shorts"
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = Path(video_path).stem

    # --- Step 1: Scene Detection ---
    scenes = _detect_scene_changes(video_path, threshold=30.0, min_scene_sec=duration)

    if not scenes:
        logger.warning("No valid scenes detected. Using evenly spaced segments.")
        cap = cv2.VideoCapture(video_path)
        total_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        spacing = total_dur / (num_shorts + 1)
        scenes = [(spacing * (i + 1), spacing * (i + 1) + duration) for i in range(num_shorts)]

    # --- Step 2: Score Segments ---
    face_detector = FaceDetector(settings.MODEL_WEIGHTS_PATH)
    scorer = SaliencyScorer(face_detector)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    scored_segments = []
    for start_sec, end_sec in scenes:
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        score = scorer.score_segment(cap, start_frame, end_frame)
        scored_segments.append((start_sec, end_sec, score))
        logger.info(f"  Scene [{start_sec:.1f}s - {end_sec:.1f}s]: score={score:.4f}")

    cap.release()

    # --- Step 3: Select Top-N ---
    scored_segments.sort(key=lambda x: x[2], reverse=True)

    # Filter by minimum score threshold
    qualified = [s for s in scored_segments if s[2] >= min_score]

    if len(qualified) < num_shorts:
        logger.warning(
            f"Only {len(qualified)} segments meet min_score={min_score}. "
            f"Using top {num_shorts} regardless."
        )
        qualified = scored_segments

    selected = qualified[:num_shorts]
    logger.info(f"Selected {len(selected)} segments for shorts generation")

    # --- Step 4 & 5: Reframe & Export ---
    reframer = SmartReframer(face_detector)
    short_paths = []

    for idx, (start_sec, end_sec, score) in enumerate(selected):
        clip_duration = min(duration, end_sec - start_sec)
        output_path = str(output_dir / f"{basename}_short_{idx:02d}.mp4")

        logger.info(
            f"Short #{idx}: [{start_sec:.1f}s - {start_sec + clip_duration:.1f}s] "
            f"score={score:.4f}"
        )

        reframer.reframe_segment(
            input_path=video_path,
            output_path=output_path,
            start_sec=start_sec,
            duration_sec=clip_duration,
        )

        short_paths.append(output_path)
        logger.info(f"  ✓ Exported: {output_path}")

    logger.info(f"=== Shorts Generation Complete: {len(short_paths)} clips ===")
    return short_paths
