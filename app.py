import os
import random
import threading

# Limit thread concurrency so Render free tier does not crash from OOM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

import io
import json
import numpy as np
import re
import shutil
import subprocess
import time
import uuid
import urllib.request
import xml.etree.ElementTree as ET
import html
from typing import List, Tuple, Dict, Optional, Callable
from pydantic import BaseModel, Field
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from google import genai
from google.genai import types

MAX_ALLOWED_HOURS = 5
MAX_ALLOWED_SECONDS = MAX_ALLOWED_HOURS * 3600

MODELS_DIR = "models"
JOBS_ROOT = "jobs"

FREE_TIER_RETENTION_HOURS = 12
JOB_METADATA_FILENAME = "job_meta.json"
TEMP_FILE_PREFIXES = ("temp_subs", "temp_whisper", "temp_raw_", "temp_paced_", "temp_visual", "preview_frame_")


def _write_job_metadata(job_dir: str, job_id: str, premium: bool = False):
    meta_path = os.path.join(job_dir, JOB_METADATA_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    meta.setdefault("job_id", job_id)
    meta.setdefault("created_at", time.time())
    meta["premium"] = premium

    with open(meta_path, "w") as f:
        json.dump(meta, f)


def _cleanup_job_temp_files(job_dir: str):
    if not os.path.isdir(job_dir):
        return
    for fname in os.listdir(job_dir):
        if fname.startswith(TEMP_FILE_PREFIXES):
            fpath = os.path.join(job_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except OSError:
                pass


def sweep_expired_jobs(retention_hours: float = FREE_TIER_RETENTION_HOURS):
    if not os.path.isdir(JOBS_ROOT):
        return

    now = time.time()
    cutoff_seconds = retention_hours * 3600
    swept = []

    for job_id in os.listdir(JOBS_ROOT):
        job_dir = os.path.join(JOBS_ROOT, job_id)
        if not os.path.isdir(job_dir):
            continue

        meta_path = os.path.join(job_dir, JOB_METADATA_FILENAME)
        created_at = None
        premium = False

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                created_at = meta.get("created_at")
                premium = meta.get("premium", False)
            except Exception:
                pass

        if created_at is None:
            created_at = os.path.getmtime(job_dir)

        if premium:
            continue

        if now - created_at > cutoff_seconds:
            shutil.rmtree(job_dir, ignore_errors=True)
            swept.append(job_id)

    if swept:
        print(f"[Retention Sweep] Deleted {len(swept)} expired job(s): {swept}")


class ViralClip(BaseModel):
    title: str = Field(description="Catchy, high CTR short title for the clip")
    hook: str = Field(description="The exact hook statement that opens the clip")
    start_seconds: int = Field(description="Start time in integer seconds")
    end_seconds: int = Field(description="End time in integer seconds")
    virality_score: int = Field(description="Predicted virality score from 1 to 100")
    reasoning: str = Field(description="Why this clip will perform well on Shorts/TikTok")


class HighlightResponse(BaseModel):
    clips: List[ViralClip]


def get_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:live\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract a valid 11 character YouTube video ID.")


def find_viral_moments(transcript_text: str) -> HighlightResponse:
    print("Analyzing highlights with Gemini using viral short form criteria...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
    You are an expert short form video editor who cuts the clips that actually blow up.
    Select the TOP 3 moments with the highest potential to go viral as standalone clips.

    Guidelines:
    Clip duration MUST be between 30 and 65 seconds.
    Start timestamp must be right before the hook setup begins.
    End timestamp must land right after the punchline, reaction, or reveal.
    Provide a virality score (1 to 100).
    Provide concise reasoning.

    Transcript:
    {transcript_text}
    """

    # Up-to-date Gemini 3 and 2.5 series priority stack
    priority_models = [
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-2.5-flash",
    ]

    try:
        available_models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if hasattr(m, "supported_actions") and "generateContent" in m.supported_actions
        ]
        excluded_keywords = ["tts", "gemma", "image", "audio", "embedding", "vision", "live", "robotics", "veo"]
        available_models = [
            m for m in available_models
            if not any(kw in m.lower() for kw in excluded_keywords)
        ]
        candidate_models = [m for m in priority_models if m in available_models] + [
            m for m in available_models if m not in priority_models
        ]
    except Exception:
        candidate_models = priority_models

    for model_name in candidate_models:
        try:
            print(f"Analyzing via: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=HighlightResponse,
                    temperature=0.2,
                ),
            )
            return HighlightResponse.model_validate_json(response.text)
        except Exception as e:
            print(f"Model {model_name} unavailable ({e}). Falling back immediately...")
            continue

    raise RuntimeError("Failed to generate highlights with available models.")

def remove_silence(
    input_path: str,
    output_path: str,
    min_silence_len: float = 1.2,
    noise_floor_db: str = "-35dB",
    pad: float = 0.15,
) -> str:
    print(f"[Pacing Optimizer] Scanning for dead pauses longer than {min_silence_len}s...")
    detect_cmd = [
        "ffmpeg", "-i", input_path,
        "-af", f"silencedetect=noise={noise_floor_db}:d={min_silence_len}",
        "-f", "null", "-"
    ]
    result = subprocess.run(detect_cmd, capture_output=True, text=True)

    silence_starts = [float(x) for x in re.findall(r"silence_start: ([0-9\.]+)", result.stderr)]
    silence_ends = [float(x) for x in re.findall(r"silence_end: ([0-9\.]+)", result.stderr)]

    if not silence_starts or not silence_ends:
        print("[Pacing Optimizer] Pacing is already tight. Skipping silence cut.")
        return input_path

    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", input_path]
    duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip() or 0)

    padded_silences = []
    for s_start, s_end in zip(silence_starts, silence_ends):
        padded_start = s_start + pad
        padded_end = s_end - pad
        if padded_end > padded_start:
            padded_silences.append((padded_start, padded_end))

    if not padded_silences:
        print("[Pacing Optimizer] No safe cuts after padding. Skipping silence cut.")
        return input_path

    keep_ranges = []
    cur_pos = 0.0
    for s_start, s_end in padded_silences:
        if s_start > cur_pos + 0.2:
            keep_ranges.append((cur_pos, s_start))
        cur_pos = s_end
    if cur_pos < duration - 0.2:
        keep_ranges.append((cur_pos, duration))

    if not keep_ranges:
        return input_path

    filter_complex = ""
    for i, (start, end) in enumerate(keep_ranges):
        filter_complex += f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
        filter_complex += f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"

    concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(len(keep_ranges))])
    filter_complex += f"{concat_inputs}concat=n={len(keep_ranges)}:v=1:a=1[outv][outa]"

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print("[Pacing Optimizer] Trimmed dead pauses successfully.")
    return output_path


def get_face_landmarker_model_path() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Face Landmarker task bundle...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path


def _render_solo_crop(frame, face_centers, smooth_solo_x_ref, width, height, crop_w_cut, out_w, out_h, alpha):
    smooth_solo_x = smooth_solo_x_ref[0]
    centers_only = [c[0] for c in face_centers]

    if centers_only:
        target_x = min(centers_only, key=lambda x: abs(x - smooth_solo_x))
    else:
        target_x = smooth_solo_x

    if abs(target_x - smooth_solo_x) > 300:
        smooth_solo_x = target_x
    else:
        smooth_solo_x += alpha * (target_x - smooth_solo_x)

    smooth_solo_x_ref[0] = smooth_solo_x

    crop_sx = int(max(0, min(smooth_solo_x - (crop_w_cut // 2), width - crop_w_cut)))
    solo_panel = frame[0:height, crop_sx:crop_sx + crop_w_cut]
    return cv2.resize(solo_panel, (out_w, out_h))


def extract_preview_frames(video_path: str, job_dir: str, num_frames: int = 3) -> List[Tuple[float, str]]:
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path]
    duration_str = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()
    duration = float(duration_str) if duration_str else 30.0

    timestamps = [round(duration * (i / (num_frames)), 2) for i in range(num_frames)]
    timestamps[0] = min(0.5, duration / 4)

    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(job_dir, f"preview_frame_{i}.png")
        cmd = [
            "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frame_paths.append((ts, out_path))

    return frame_paths


def _parse_box_fractions(raw: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 'x,y,w,h' but got: {raw!r}")
    x, y, w, h = (float(p) for p in parts)
    for name, val in (("x", x), ("y", y), ("w", w), ("h", h)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name}={val} is out of range 0.0-1.0")
    return x, y, w, h


def _prompt_for_box_segments(
    video_path: str, job_dir: str, box_label: str,
    prompt_fn: Optional[Callable[[str], str]] = None,
) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    ask = prompt_fn or (lambda q: input(q).strip())
    frames = extract_preview_frames(video_path, job_dir, num_frames=3)

    print(f"\n[{box_label} Picker] Preview frames saved:")
    for ts, path in frames:
        print(f"  t={ts}s -> {path}")

    segments: List[Tuple[float, Tuple[float, float, float, float]]] = []
    first_box_input = ask(f"{box_label} box for the start of the clip (x,y,w,h): ")
    segments.append((0.0, _parse_box_fractions(first_box_input)))

    while True:
        moved = ask(
            f"Does the {box_label.lower()} move again later in this clip? "
            "Enter a timestamp in seconds to add another position, or leave blank if it stays put: "
        )
        if not moved:
            break
        try:
            move_ts = float(moved)
        except ValueError:
            print("Not a valid number, skipping.")
            continue
        new_box_input = ask(f"{box_label} box starting at {move_ts}s (x,y,w,h): ")
        segments.append((move_ts, _parse_box_fractions(new_box_input)))

    segments.sort(key=lambda s: s[0])
    return segments


def prompt_for_facecam_segments(
    video_path: str, job_dir: str, prompt_fn: Optional[Callable[[str], str]] = None,
) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    return _prompt_for_box_segments(video_path, job_dir, box_label="Facecam", prompt_fn=prompt_fn)


def prompt_for_reaction_start_box(
    video_path: str, job_dir: str, prompt_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[float, float, float, float]:
    extract_preview_frames(video_path, job_dir, num_frames=3)
    ask = prompt_fn or (lambda q: input(q).strip())
    raw = ask("Reaction PIP box for the start of the clip (x,y,w,h): ")
    return _parse_box_fractions(raw)


def _frac_box_to_px(box_frac: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x_frac, y_frac, w_frac, h_frac = box_frac
    bx = int(x_frac * width)
    by = int(y_frac * height)
    bw = max(1, int(w_frac * width))
    bh = max(1, int(h_frac * height))
    bx = max(0, min(bx, width - bw))
    by = max(0, min(by, height - bh))
    return bx, by, bw, bh


def _create_pip_tracker():
    candidates = []
    legacy = getattr(cv2, "legacy", None)
    for attr in ("TrackerCSRT_create", "TrackerKCF_create"):
        ctor = getattr(cv2, attr, None)
        if ctor is not None:
            candidates.append(ctor)
        if legacy is not None:
            legacy_ctor = getattr(legacy, attr, None)
            if legacy_ctor is not None:
                candidates.append(legacy_ctor)

    for ctor in candidates:
        try:
            return ctor()
        except Exception:
            continue

    raise RuntimeError(
        "No OpenCV object tracker available. Install opencv-contrib-python to enable reaction auto-tracking."
    )


def _box_for_time(segments: List[Tuple[float, Tuple[float, float, float, float]]], t: float) -> Tuple[float, float, float, float]:
    active = segments[0][1]
    for start_t, box in segments:
        if start_t <= t:
            active = box
        else:
            break
    return active


OUTPUT_DIMENSIONS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


def render_gimbal_tracked_video(
    input_path: str,
    output_path: str,
    job_dir: str,
    mode: str = "split",
    facecam_segments: Optional[List[Tuple[float, Tuple[float, float, float, float]]]] = None,
    reaction_start_box: Optional[Tuple[float, float, float, float]] = None,
    aspect_ratio: str = "9:16",
    prompt_fn: Optional[Callable[[str], str]] = None,
):
    ask = prompt_fn or (lambda q: input(q).strip())
    task_path = get_face_landmarker_model_path()

    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=4,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = OUTPUT_DIMENSIONS.get(aspect_ratio, OUTPUT_DIMENSIONS["9:16"])
    panel_h = (out_h - 4) // 2
    crop_w_cut = int(height * (out_w / out_h))

    SPLIT_ZOOM = 0.55
    crop_h_split = int(height * SPLIT_ZOOM)
    crop_w_split = int(crop_h_split * (out_w / panel_h))

    temp_processed_video = os.path.join(job_dir, "temp_visual.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_processed_video, fourcc, fps, (out_w, out_h))

    default_center_x = float(width // 2)
    default_left_x = float(width * 0.28)
    default_right_x = float(width * 0.72)
    default_left_y = float(height * 0.38)
    default_right_y = float(height * 0.38)

    smooth_solo_x = default_center_x
    smooth_left_x = default_left_x
    smooth_right_x = default_right_x
    smooth_left_y = default_left_y
    smooth_right_y = default_right_y
    alpha = 0.10

    facecam_panel_h = None
    game_panel_h = None
    if mode == "gaming":
        if not facecam_segments:
            raise ValueError("'gaming' mode requires facecam_segments.")
        facecam_panel_h = int(out_h * 0.35)
        game_panel_h = out_h - facecam_panel_h - 4

    reaction_panel_h = None
    main_panel_h = None
    if mode == "reaction":
        if not reaction_start_box:
            raise ValueError("'reaction' mode requires reaction_start_box.")
        reaction_panel_h = int(out_h * 0.32)
        main_panel_h = out_h - reaction_panel_h - 4

    reaction_tracker = None
    reaction_box_px: Optional[Tuple[int, int, int, int]] = None
    reaction_needs_reinit = (mode == "reaction")

    frames_since_last_detection = 0
    MAX_LOST_FRAMES = int(fps * 3.0)

    HOLD_SECONDS = 0.6
    hold_frames_required = max(1, int(fps * HOLD_SECONDS))
    active_is_two_shot = False
    pending_is_two_shot = False
    pending_streak = 0

    MOUTH_ALPHA = 0.35
    SWITCH_MARGIN = 0.015
    SPEAKER_HOLD_SECONDS = 0.9
    speaker_hold_frames_required = max(1, int(fps * SPEAKER_HOLD_SECONDS))
    smooth_mouth_left = 0.0
    smooth_mouth_right = 0.0
    active_speaker = "left"
    pending_speaker = "left"
    speaker_streak = 0

    prev_gray_small = None
    CUT_DIFF_THRESHOLD = 35
    MIN_SPLIT_GAP = 90

    def _reset_tracking_state():
        nonlocal smooth_solo_x, smooth_left_x, smooth_right_x, smooth_left_y, smooth_right_y
        nonlocal active_is_two_shot, pending_is_two_shot, pending_streak
        nonlocal active_speaker, pending_speaker, speaker_streak
        nonlocal frames_since_last_detection, smooth_mouth_left, smooth_mouth_right
        nonlocal reaction_needs_reinit
        if mode == "reaction":
            reaction_needs_reinit = True
        smooth_solo_x = default_center_x
        smooth_left_x = default_left_x
        smooth_right_x = default_right_x
        smooth_left_y = default_left_y
        smooth_right_y = default_right_y
        active_is_two_shot = False
        pending_is_two_shot = False
        pending_streak = 0
        active_speaker = "left"
        pending_speaker = "left"
        speaker_streak = 0
        frames_since_last_detection = 0
        smooth_mouth_left = 0.0
        smooth_mouth_right = 0.0

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_time = frame_idx / fps
            frame_idx += 1

            gray_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18))
            if prev_gray_small is not None:
                cut_diff = float(np.mean(np.abs(gray_small.astype(np.int16) - prev_gray_small.astype(np.int16))))
                if cut_diff > CUT_DIFF_THRESHOLD:
                    _reset_tracking_state()
            prev_gray_small = gray_small

            face_points = []
            if mode not in ("gaming", "reaction"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)

                if res.face_landmarks:
                    for landmarks in res.face_landmarks:
                        cx = int(landmarks[1].x * width)
                        cy = int(landmarks[1].y * height)
                        mouth_gap = abs(landmarks[14].y - landmarks[13].y)
                        eye_span = abs(landmarks[263].x - landmarks[33].x) + 1e-6
                        mouth_ratio = mouth_gap / eye_span
                        face_points.append((cx, cy, mouth_ratio))

            midpoint = width // 2
            left_faces = [p for p in face_points if p[0] < midpoint - MIN_SPLIT_GAP]
            right_faces = [p for p in face_points if p[0] > midpoint + MIN_SPLIT_GAP]

            if face_points:
                frames_since_last_detection = 0
            else:
                frames_since_last_detection += 1

            face_centers = [(p[0], p[1]) for p in face_points]
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            if mode == "split":
                raw_is_two_shot = len(left_faces) > 0 and len(right_faces) > 0

                if raw_is_two_shot == pending_is_two_shot:
                    pending_streak += 1
                else:
                    pending_is_two_shot = raw_is_two_shot
                    pending_streak = 1

                if pending_streak >= hold_frames_required and active_is_two_shot != pending_is_two_shot:
                    active_is_two_shot = pending_is_two_shot

                if active_is_two_shot:
                    target_left = min(left_faces, key=lambda p: abs(p[0] - smooth_left_x)) if left_faces else (smooth_left_x, smooth_left_y, 0)
                    target_right = min(right_faces, key=lambda p: abs(p[0] - smooth_right_x)) if right_faces else (smooth_right_x, smooth_right_y, 0)

                    smooth_left_x += alpha * (target_left[0] - smooth_left_x)
                    smooth_left_y += alpha * (target_left[1] - smooth_left_y)
                    smooth_right_x += alpha * (target_right[0] - smooth_right_x)
                    smooth_right_y += alpha * (target_right[1] - smooth_right_y)

                    crop_lx = int(max(0, min(smooth_left_x - (crop_w_split // 2), width - crop_w_split)))
                    crop_ly = int(max(0, min(smooth_left_y - (crop_h_split // 2), height - crop_h_split)))
                    top_panel = frame[crop_ly:crop_ly + crop_h_split, crop_lx:crop_lx + crop_w_split]
                    top_resized = cv2.resize(top_panel, (out_w, panel_h))

                    crop_rx = int(max(0, min(smooth_right_x - (crop_w_split // 2), width - crop_w_split)))
                    crop_ry = int(max(0, min(smooth_right_y - (crop_h_split // 2), height - crop_h_split)))
                    bot_panel = frame[crop_ry:crop_ry + crop_h_split, crop_rx:crop_rx + crop_w_split]
                    bot_resized = cv2.resize(bot_panel, (out_w, panel_h))

                    canvas[0:panel_h, :] = top_resized
                    canvas[panel_h:panel_h + 4, :] = (20, 20, 20)
                    canvas[panel_h + 4:panel_h + 4 + panel_h, :] = bot_resized
                else:
                    if not face_centers and frames_since_last_detection > MAX_LOST_FRAMES:
                        smooth_solo_x += 0.02 * (default_center_x - smooth_solo_x)

                    canvas = _render_solo_crop(frame, face_centers, smooth_solo_x_ref := [smooth_solo_x], width, height, crop_w_cut, out_w, out_h, alpha)
                    smooth_solo_x = smooth_solo_x_ref[0]

            elif mode == "speaker_switch":
                left_mouth = max((p[2] for p in left_faces), default=0.0)
                right_mouth = max((p[2] for p in right_faces), default=0.0)
                smooth_mouth_left += MOUTH_ALPHA * (left_mouth - smooth_mouth_left)
                smooth_mouth_right += MOUTH_ALPHA * (right_mouth - smooth_mouth_right)

                if left_faces:
                    target_left = min(left_faces, key=lambda p: abs(p[0] - smooth_left_x))
                    smooth_left_x += alpha * (target_left[0] - smooth_left_x)
                    smooth_left_y += alpha * (target_left[1] - smooth_left_y)
                if right_faces:
                    target_right = min(right_faces, key=lambda p: abs(p[0] - smooth_right_x))
                    smooth_right_x += alpha * (target_right[0] - smooth_right_x)
                    smooth_right_y += alpha * (target_right[1] - smooth_right_y)

                if active_speaker == "left":
                    desired = "right" if (smooth_mouth_right - smooth_mouth_left) > SWITCH_MARGIN else "left"
                else:
                    desired = "left" if (smooth_mouth_left - smooth_mouth_right) > SWITCH_MARGIN else "right"

                if not left_faces and right_faces:
                    desired = "right"
                elif left_faces and not right_faces:
                    desired = "left"

                if desired == pending_speaker:
                    speaker_streak += 1
                else:
                    pending_speaker = desired
                    speaker_streak = 1

                if speaker_streak >= speaker_hold_frames_required and active_speaker != pending_speaker:
                    active_speaker = pending_speaker

                target_x = smooth_left_x if active_speaker == "left" else smooth_right_x
                crop_sx = int(max(0, min(target_x - (crop_w_cut // 2), width - crop_w_cut)))
                solo_panel = frame[0:height, crop_sx:crop_sx + crop_w_cut]
                canvas = cv2.resize(solo_panel, (out_w, out_h))

            elif mode == "gaming":
                x_frac, y_frac, w_frac, h_frac = _box_for_time(facecam_segments, current_time)
                fx = int(x_frac * width)
                fy = int(y_frac * height)
                fw = max(1, int(w_frac * width))
                fh = max(1, int(h_frac * height))
                fx = max(0, min(fx, width - fw))
                fy = max(0, min(fy, height - fh))

                facecam_panel = frame[fy:fy + fh, fx:fx + fw]
                facecam_resized = cv2.resize(facecam_panel, (out_w, facecam_panel_h))

                crop_w_game = min(width, int(height * (out_w / game_panel_h)))
                crop_gx = int(max(0, min((width // 2) - (crop_w_game // 2), width - crop_w_game)))
                game_panel = frame[0:height, crop_gx:crop_gx + crop_w_game]
                game_resized = cv2.resize(game_panel, (out_w, game_panel_h))

                canvas[0:facecam_panel_h, :] = facecam_resized
                canvas[facecam_panel_h:facecam_panel_h + 4, :] = (20, 20, 20)
                canvas[facecam_panel_h + 4:facecam_panel_h + 4 + game_panel_h, :] = game_resized

            elif mode == "reaction":
                if reaction_needs_reinit:
                    if reaction_box_px is not None:
                        cut_preview_path = os.path.join(job_dir, f"preview_frame_cut_{frame_idx}.png")
                        cv2.imwrite(cut_preview_path, frame)
                        raw = ask(
                            "If the PIP moved/resized, enter its new box (x,y,w,h). "
                            "Otherwise leave blank to keep tracking from the same spot: "
                        )
                        box_frac = _parse_box_fractions(raw) if raw else None
                    else:
                        box_frac = reaction_start_box

                    if box_frac is not None:
                        reaction_box_px = _frac_box_to_px(box_frac, width, height)

                    reaction_tracker = _create_pip_tracker()
                    reaction_tracker.init(frame, tuple(reaction_box_px))
                    reaction_needs_reinit = False
                else:
                    ok, tracked_box = reaction_tracker.update(frame)
                    if ok:
                        tx, ty, tw, th = (int(v) for v in tracked_box)
                        tw = max(1, tw)
                        th = max(1, th)
                        tx = max(0, min(tx, width - tw))
                        ty = max(0, min(ty, height - th))
                        reaction_box_px = (tx, ty, tw, th)

                rx, ry, rw, rh = reaction_box_px
                reaction_panel = frame[ry:ry + rh, rx:rx + rw]
                interp = cv2.INTER_CUBIC if (out_w > rw or reaction_panel_h > rh) else cv2.INTER_AREA
                reaction_resized = cv2.resize(reaction_panel, (out_w, reaction_panel_h), interpolation=interp)

                crop_w_main = min(width, int(height * (out_w / main_panel_h)))
                crop_mx = int(max(0, min((width // 2) - (crop_w_main // 2), width - crop_w_main)))
                main_panel = frame[0:height, crop_mx:crop_mx + crop_w_main]
                main_resized = cv2.resize(main_panel, (out_w, main_panel_h))

                canvas[0:reaction_panel_h, :] = reaction_resized
                canvas[reaction_panel_h:reaction_panel_h + 4, :] = (20, 20, 20)
                canvas[reaction_panel_h + 4:reaction_panel_h + 4 + main_panel_h, :] = main_resized

            elif mode == "gaming_split_noface":
                top_h = int(height * 0.6)
                top_w = int(top_h * (out_w / panel_h))
                top_x = max(0, (width - top_w) // 2)
                top_panel = frame[0:top_h, top_x:top_x + top_w]
                top_resized = cv2.resize(top_panel, (out_w, panel_h))

                bot_h = int(height * 0.55)
                bot_w = int(bot_h * (out_w / panel_h))
                bot_x = max(0, (width - bot_w) // 2)
                bot_y = max(0, height - bot_h)
                bot_panel = frame[bot_y:height, bot_x:bot_x + bot_w]
                bot_resized = cv2.resize(bot_panel, (out_w, panel_h))

                canvas[0:panel_h, :] = top_resized
                canvas[panel_h:panel_h + 4, :] = (20, 20, 20)
                canvas[panel_h + 4:panel_h + 4 + panel_h, :] = bot_resized

            else:
                centers_only = [p[0] for p in face_centers]
                if centers_only:
                    target_x = min(centers_only, key=lambda x: abs(x - smooth_solo_x))
                    if abs(target_x - smooth_solo_x) > 300:
                        smooth_solo_x = target_x
                    else:
                        smooth_solo_x += alpha * (target_x - smooth_solo_x)
                elif frames_since_last_detection > MAX_LOST_FRAMES:
                    smooth_solo_x += 0.02 * (default_center_x - smooth_solo_x)

                crop_sx = int(max(0, min(smooth_solo_x - (crop_w_cut // 2), width - crop_w_cut)))
                solo_panel = frame[0:height, crop_sx:crop_sx + crop_w_cut]
                canvas = cv2.resize(solo_panel, (out_w, out_h))

            out_writer.write(canvas)
    finally:
        detector.close()
        cap.release()
        out_writer.release()

    try:
        merge_cmd = [
           "ffmpeg", "-y",
            "-i", temp_processed_video,
            "-i", input_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]
        subprocess.run(merge_cmd, check=True)
    finally:
        if os.path.exists(temp_processed_video):
            os.remove(temp_processed_video)


def process_clip_from_file(
    raw_video_path: str,
    clip: ViralClip,
    index: int,
    job_dir: str,
    mode: str = "split",
    aspect_ratio: str = "9:16",
    prompt_fn: Optional[Callable[[str], str]] = None
):
    clean_title = re.sub(r'[^a-zA-Z0-9]', '_', clip.title)[:25]
    clean_ratio = aspect_ratio.replace(':', '_')
    temp_paced = os.path.join(job_dir, f"temp_paced_{index}.mp4")
    final_output = os.path.join(job_dir, f"clip_{index}_{clean_title}_{clean_ratio}.mp4")

    print(f"\nProcessing Uploaded Clip {index}: {clip.title}")
    print(f"Hook: {clip.hook}")

    try:
        paced_file = remove_silence(raw_video_path, temp_paced, min_silence_len=0.6)

        if aspect_ratio == "16:9":
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", paced_file,
                "-vf", "scale=1920:1080,setsar=1",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac", final_output
            ]
            subprocess.run(ffmpeg_cmd, check=True)
        elif mode == "gaming":
            facecam_segments = prompt_for_facecam_segments(paced_file, job_dir, prompt_fn=prompt_fn)
            render_gimbal_tracked_video(
                paced_file, final_output, job_dir, mode=mode,
                facecam_segments=facecam_segments, aspect_ratio=aspect_ratio, prompt_fn=prompt_fn,
            )
        elif mode == "reaction":
            reaction_start_box = prompt_for_reaction_start_box(paced_file, job_dir, prompt_fn=prompt_fn)
            render_gimbal_tracked_video(
                paced_file, final_output, job_dir, mode=mode,
                reaction_start_box=reaction_start_box, aspect_ratio=aspect_ratio, prompt_fn=prompt_fn,
            )
        else:
            render_gimbal_tracked_video(paced_file, final_output, job_dir, mode=mode, aspect_ratio=aspect_ratio)

        print(f"SUCCESS: Exported {final_output}")
    finally:
        for tmp in [raw_video_path, temp_paced]:
            if os.path.exists(tmp):
                os.remove(tmp)