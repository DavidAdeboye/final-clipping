import os
import io
import json
import random
import re
import shutil
import subprocess
import time
import uuid
import urllib.request
import urllib.parse
from typing import List, Tuple, Dict, Optional, Callable
import html
import xml.etree.ElementTree as ET
# Limit thread concurrency to avoid OOM on constrained server tiers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)
import numpy as np

from pydantic import BaseModel, Field
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

MAX_ALLOWED_HOURS = 5
MAX_ALLOWED_SECONDS = MAX_ALLOWED_HOURS * 3600

MODELS_DIR = "models"
JOBS_ROOT = "jobs"
FREE_TIER_RETENTION_HOURS = 12
JOB_METADATA_FILENAME = "job_meta.json"
TEMP_FILE_PREFIXES = ("temp_subs", "temp_whisper", "temp_raw_", "temp_paced_", "temp_visual", "preview_frame_")

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"


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


def get_video_duration(video_url: str) -> int:
    try:
        video_id = get_video_id(video_url)
        req = urllib.request.Request(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            headers={"User-Agent": _BROWSER_UA}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data.get("title"):
                return 600
    except Exception:
        pass
    return 600

def extract_captions(video_url: str, job_dir: str) -> str:
    print("Fetching video captions via YouTube InnerTube API...")
    video_id = get_video_id(video_url)

    # 1. Query YouTube InnerTube Android endpoint for playback metadata & captions
    innertube_url = "https://www.youtube.com/youtubei/v1/player"
    payload = json.dumps({
        "context": {
            "client": {
                "clientName": "ANDROID",
                "clientVersion": "19.09.37",
                "androidSdkVersion": 30,
                "hl": "en",
                "gl": "US"
            }
        },
        "videoId": video_id
    }).encode("utf-8")

    req = urllib.request.Request(
        innertube_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11; en_US)",
            "X-YouTube-Client-Name": "3",
            "X-YouTube-Client-Version": "19.09.37"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to query YouTube API: {e}")

    # 2. Locate caption tracks inside response
    caption_tracks = (
        data.get("captions", {})
        .get("playerCaptionsTracklistRenderer", {})
        .get("captionTracks", [])
    )

    if not caption_tracks:
        raise RuntimeError("No captions available for this video.")

    # Prioritize English tracks
    target_url = None
    for track in caption_tracks:
        lang = track.get("languageCode", "").lower()
        if lang.startswith("en"):
            target_url = track.get("baseUrl")
            break

    if not target_url:
        target_url = caption_tracks[0].get("baseUrl")

    # Request caption transcript in JSON3 format
    if "fmt=" not in target_url:
        target_url += "&fmt=json3"

    cap_req = urllib.request.Request(
        target_url,
        headers={"User-Agent": _BROWSER_UA}
    )

    with urllib.request.urlopen(cap_req, timeout=15) as c_resp:
        cap_data = c_resp.read().decode("utf-8", errors="ignore")

    lines = []
    try:
        json_subs = json.loads(cap_data)
        for event in json_subs.get("events", []):
            segs = event.get("segs", [])
            for seg in segs:
                utf8 = seg.get("utf8", "").strip()
                if utf8 and utf8 != "\n":
                    lines.append(utf8)
    except Exception:
        # Fallback XML parsing if response was returned as standard timedtext XML
        root = ET.fromstring(cap_data)
        for text_el in root.findall(".//text"):
            if text_el.text:
                clean_t = html.unescape(text_el.text).strip()
                if clean_t:
                    lines.append(clean_t)

    full_transcript = " ".join(lines)
    if not full_transcript.strip():
        raise RuntimeError("Retrieved transcript track was empty.")

    print(f"Successfully retrieved {len(full_transcript)} characters of transcript.")
    return full_transcript

def find_viral_moments_direct(video_url: str) -> HighlightResponse:
    print("Analyzing YouTube video directly with Gemini for transcript & highlights...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
    You are an expert short form video editor.
    Watch and analyze this YouTube video: {video_url}

    Extract the transcript internally, identify the most engaging parts, and select the TOP 3 moments with the highest potential to perform well as standalone vertical clips (Shorts/TikTok/Reels).

    Strict Rules:
    1. Clip duration MUST be between 30 and 65 seconds.
    2. Start timestamp (start_seconds) must begin right before the hook or question is asked.
    3. End timestamp (end_seconds) must end right after the punchline, reaction, or takeaway.
    4. Ensure timestamps match the actual timeline of the video precisely.
    5. Provide a virality score (1 to 100).
    6. Provide concise reasoning.
    """

    priority_models = [
        "gemini-3.6-flash",
        "gemini-3.7-flash",
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
            print(f"Model {model_name} error ({e}). Trying fallback...")
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


def resolve_direct_video_stream(video_url: str) -> Optional[str]:
    video_id = get_video_id(video_url)

    # 1. Fetch direct stream URLs via Cloudflare Worker using YouTube Android Player API
    worker_url = "https://yt-proxy.thatguyjude.workers.dev/?url=" + urllib.parse.quote(
        "https://www.youtube.com/youtubei/v1/player"
    )
    
    payload = json.dumps({
        "context": {
            "client": {
                "clientName": "ANDROID_TESTSUITE",
                "clientVersion": "1.9",
                "androidSdkVersion": 30,
                "hl": "en",
                "gl": "US"
            }
        },
        "videoId": video_id
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            worker_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11; en_US)"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            formats = data.get("streamingData", {}).get("formats", []) + data.get("streamingData", {}).get("adaptiveFormats", [])
            for f in formats:
                # Find direct MP4 stream with audio and video
                if "url" in f and "video/mp4" in f.get("mimeType", "") and f.get("height", 0) >= 360:
                    stream_url = f["url"]
                    print(f"Direct stream URL acquired via Cloudflare Worker bridge ({f.get('qualityLabel', '720p')})")
                    return stream_url
    except Exception as e:
        print(f"Worker stream bridge notice: {e}")

    # 2. Query public streaming instances
    cobalt_instances = [
        "https://cobalt-api.kwiatekmiki.pl/",
        "https://co.eepy.today/",
    ]
    cobalt_payload = json.dumps({
        "url": video_url,
        "videoQuality": "720",
        "youtubeVideoCodec": "h264",
        "downloadMode": "auto"
    }).encode("utf-8")

    for c_url in cobalt_instances:
        try:
            req = urllib.request.Request(
                c_url,
                data=cobalt_payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": _BROWSER_UA
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                stream_url = data.get("url")
                if stream_url:
                    return stream_url
        except Exception:
            continue

    return None


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
    print(f"Time: {clip.start_seconds}s to {clip.end_seconds}s")

    stream_url = resolve_direct_video_stream(video_url)
    duration = max(5, clip.end_seconds - clip.start_seconds)

    if stream_url:
        ffmpeg_dl = [
            "ffmpeg", "-y",
            "-ss", str(clip.start_seconds),
            "-i", stream_url,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac",
            temp_raw
        ]
        subprocess.run(ffmpeg_dl, check=True)
    else:
        raise RuntimeError("Unable to acquire direct video stream for slicing.")

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
        raise ValueError(f"Video exceeds the {MAX_ALLOWED_HOURS} hour limit.")

    # Single multimodal API call: Gemini fetches video transcript & extracts viral segments
    highlight_data = find_viral_moments_direct(video_url)

    for idx, clip in enumerate(highlight_data.clips, start=1):
        process_clip(video_url, clip, idx, job_dir, mode=mode, aspect_ratio=aspect_ratio)

    return job_dir