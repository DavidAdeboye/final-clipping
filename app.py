import os
import random
import threading

# Limit thread concurrency to avoid OOM on free tier
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

# Proxy Configuration from Environment
_RAW_PROXIES = os.environ.get("YTDLP_PROXIES", "").strip()

def _clean_proxy_url(raw: str) -> str:
    # Extracts the pure http/https url and strips any accidental markdown brackets
    match = re.search(r'https?://[^\s\)\]]+', raw)
    return match.group(0) if match else raw.strip()

_PROXY_LIST: List[str] = [
    _clean_proxy_url(p) for p in _RAW_PROXIES.split(",") if p.strip()
]
_proxy_stats: Dict[str, Dict[str, int]] = {}

YT_EXTRA_ARGS = ["--extractor-args", "youtube:player_client=android,ios,tv"]
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
_AUTO = object()


def _proxy_pool_shuffled() -> List[Optional[str]]:
    if not _PROXY_LIST:
        return [None]
    pool = list(_PROXY_LIST)
    random.shuffle(pool)
    return pool


def _record_proxy_result(proxy: Optional[str], success: bool):
    key = proxy or "direct"
    stats = _proxy_stats.setdefault(key, {"ok": 0, "fail": 0})
    if success:
        stats["ok"] += 1
    else:
        stats["fail"] += 1


def _proxy_args() -> list:
    if not _PROXY_LIST:
        return []
    proxy_url = random.choice(_PROXY_LIST)
    # If using a Cloudflare worker URL proxy, yt-dlp expects standard http/socks format
    if "workers.dev" in proxy_url:
        # Pass headers and prevent connect tunnel failure
        return []
    return ["--proxy", proxy_url]

def yt_client_args(proxy=_AUTO) -> list:
    if proxy is _AUTO:
        proxy_part = _proxy_args()
    elif proxy and "workers.dev" not in proxy:
        proxy_part = ["--proxy", proxy]
    else:
        proxy_part = []

    return [
        "--user-agent", _BROWSER_UA,
        "--rm-cache-dir",
        "--no-check-certificates",
        "--no-warnings",
        "--prefer-free-formats",
        "--geo-bypass",
        "--extractor-args", "youtube:player_client=android,ios",
    ] + proxy_part


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


def sweep_expired_jobs(retention_hours: float = FREE_TIER_RETENTION_HOURS):
    if not os.path.isdir(JOBS_ROOT):
        return
    now = time.time()
    cutoff_seconds = retention_hours * 3600
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


def get_video_duration(video_url: str) -> int:
    cmd = ["yt-dlp", "--print", "duration"] + yt_client_args() + [video_url]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = res.stdout.strip()
    return int(float(raw)) if raw else 0


def extract_captions(video_url: str, job_dir: str) -> str:
    print("Fetching video captions via yt-dlp...")
    sub_base = os.path.join(job_dir, "temp_subs")
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang", "en.*,en",
        "--sub-format", "ttml/srv3/srv2/srv1/vtt",
        "--skip-download",
    ] + yt_client_args() + [video_url, "-o", sub_base]

    subprocess.run(cmd, check=True, capture_output=True)

    vtt_file = None
    for f in os.listdir(job_dir):
        if f.startswith("temp_subs") and (f.endswith(".ttml") or f.endswith(".vtt") or f.endswith(".srv3")):
            vtt_file = os.path.join(job_dir, f)
            break

    if not vtt_file or not os.path.exists(vtt_file):
        raise RuntimeError("No subtitles found for this video.")

    with open(vtt_file, "r", encoding="utf-8") as file:
        content = file.read()

    lines = []
    for line in content.splitlines():
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean and not clean.startswith("WEBVTT") and "-->" not in clean and not clean.isdigit():
            lines.append(clean)

    return " ".join(lines)


def find_viral_moments(transcript_text: str) -> HighlightResponse:
    print("Analyzing highlights with Gemini using viral short form criteria...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
    You are an expert short form video editor who cuts clips that go viral.
    Select the TOP 3 moments with the highest potential to perform well as standalone clips.

    Guidelines:
    Clip duration MUST be between 30 and 65 seconds.
    Start timestamp must be right before the hook setup begins.
    End timestamp must land right after the punchline, reaction, or reveal.
    Provide a virality score (1 to 100).
    Provide concise reasoning.

    Transcript:
    {transcript_text[:12000]}
    """

    priority_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    for model_name in priority_models:
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
            print(f"Model {model_name} unavailable ({e}). Retrying fallback...")
            continue

    raise RuntimeError("Failed to generate highlights with Gemini models.")


def remove_silence(
    input_path: str,
    output_path: str,
    min_silence_len: float = 1.2,
    noise_floor_db: str = "-35dB",
    pad: float = 0.15,
) -> str:
    detect_cmd = [
        "ffmpeg", "-i", input_path,
        "-af", f"silencedetect=noise={noise_floor_db}:d={min_silence_len}",
        "-f", "null", "-"
    ]
    result = subprocess.run(detect_cmd, capture_output=True, text=True)

    silence_starts = [float(x) for x in re.findall(r"silence_start: ([0-9\.]+)", result.stderr)]
    silence_ends = [float(x) for x in re.findall(r"silence_end: ([0-9\.]+)", result.stderr)]

    if not silence_starts or not silence_ends:
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
    return output_path


def get_face_landmarker_model_path() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "face_landmarker.task")
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path


OUTPUT_DIMENSIONS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


def render_gimbal_tracked_video(
    input_path: str,
    output_path: str,
    job_dir: str,
    mode: str = "cut",
    aspect_ratio: str = "9:16",
):
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

    MIN_SPLIT_GAP = 90

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)

            face_points = []
            if res.face_landmarks:
                for landmarks in res.face_landmarks:
                    cx = int(landmarks[1].x * width)
                    cy = int(landmarks[1].y * height)
                    face_points.append((cx, cy))

            midpoint = width // 2
            left_faces = [p for p in face_points if p[0] < midpoint - MIN_SPLIT_GAP]
            right_faces = [p for p in face_points if p[0] > midpoint + MIN_SPLIT_GAP]

            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            if mode == "split" and left_faces and right_faces:
                target_left = min(left_faces, key=lambda p: abs(p[0] - smooth_left_x))
                target_right = min(right_faces, key=lambda p: abs(p[0] - smooth_right_x))

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
                if face_points:
                    target_x = min([p[0] for p in face_points], key=lambda x: abs(x - smooth_solo_x))
                    smooth_solo_x += alpha * (target_x - smooth_solo_x)

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


def process_clip(
    video_url: str,
    clip: ViralClip,
    index: int,
    job_dir: str,
    mode: str = "cut",
    aspect_ratio: str = "9:16"
):
    clean_title = re.sub(r'[^a-zA-Z0-9]', '_', clip.title)[:25]
    clean_ratio = aspect_ratio.replace(':', '_')
    temp_raw = os.path.join(job_dir, f"temp_raw_{index}.mp4")
    temp_paced = os.path.join(job_dir, f"temp_paced_{index}.mp4")
    final_output = os.path.join(job_dir, f"clip_{index}_{clean_title}_{clean_ratio}.mp4")

    print(f"\nProcessing Clip {index}: {clip.title}")
    slice_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--download-sections", f"*{clip.start_seconds}-{clip.end_seconds}",
        "--merge-output-format", "mp4",
        "--force-keyframes-at-cuts",
        "--retries", "5",
        "--socket-timeout", "25"
    ] + yt_client_args() + [video_url, "-o", temp_raw]

    subprocess.run(slice_cmd, check=True)

    try:
        paced_file = remove_silence(temp_raw, temp_paced, min_silence_len=0.6)
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
        else:
            render_gimbal_tracked_video(paced_file, final_output, job_dir, mode=mode, aspect_ratio=aspect_ratio)
        print(f"SUCCESS: Exported {final_output}")
    finally:
        for tmp in [temp_raw, temp_paced]:
            if os.path.exists(tmp):
                os.remove(tmp)


def run_pipeline(
    video_url: str,
    aspect_ratio: str = "9:16",
    mode: str = "cut",
    prompt_fn: Optional[Callable[[str], str]] = None,
    job_id: Optional[str] = None
) -> str:
    sweep_expired_jobs()

    if not job_id:
        job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    _write_job_metadata(job_dir, job_id)

    duration = get_video_duration(video_url)
    if duration > MAX_ALLOWED_SECONDS:
        raise ValueError(f"Video exceeds the {MAX_ALLOWED_HOURS}-hour limit.")

    transcript = extract_captions(video_url, job_dir)
    highlight_data = find_viral_moments(transcript)

    for idx, clip in enumerate(highlight_data.clips, start=1):
        process_clip(video_url, clip, idx, job_dir, mode=mode, aspect_ratio=aspect_ratio)

    return job_dir