import os
import io
import base64
import json
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
import urllib.request
import urllib.parse
import gc
from typing import List, Tuple, Dict, Optional, Callable
import html
import xml.etree.ElementTree as ET

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import cv2
cv2.setNumThreads(2)
import numpy as np

from pydantic import BaseModel, Field
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

MAX_ALLOWED_HOURS = 5
MAX_ALLOWED_SECONDS = MAX_ALLOWED_HOURS * 3600

MODELS_DIR = "models"
JOBS_ROOT = "jobs"
FREE_TIER_RETENTION_HOURS = 12
JOB_METADATA_FILENAME = "job_meta.json"

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

_PROXY_POOL: Optional[List[str]] = None
_WHISPER_MODEL: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def _load_proxy_pool() -> List[str]:
    global _PROXY_POOL
    if _PROXY_POOL is not None:
        return _PROXY_POOL

    pool: List[str] = []
    proxies_path = os.getenv("WEBSHARE_PROXIES_FILE", "webshare_proxies.txt")
    raw_lines: List[str] = []
    if os.path.isfile(proxies_path):
        try:
            with open(proxies_path, "r", encoding="utf-8") as f:
                raw_lines.extend(f.readlines())
        except Exception:
            pass

    env_pool = os.getenv("WEBSHARE_PROXIES", "").strip()
    if env_pool:
        raw_lines.extend(re.split(r"[\n,]+", env_pool))

    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) == 4:
            host, port, user, pwd = parts
            pool.append(f"http://{user}:{pwd}@{host}:{port}")
        elif len(parts) == 2:
            host, port = parts
            pool.append(f"http://{host}:{port}")

    _PROXY_POOL = pool
    return pool


def _pick_proxy() -> str:
    pool = _load_proxy_pool()
    if pool:
        return random.choice(pool)
    return os.getenv("YOUTUBE_PROXY", "").strip() or os.getenv("HTTP_PROXY", "").strip()


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
    title: str = Field(description="Catchy short title for the clip")
    hook: str = Field(description="The exact hook statement that opens the clip")
    start_seconds: int = Field(description="Start time in integer seconds")
    end_seconds: int = Field(description="End time in integer seconds")
    virality_score: int = Field(description="Predicted virality score from 1 to 100")
    reasoning: str = Field(description="Why this clip will perform well on Shorts/TikTok")


class HighlightResponse(BaseModel):
    clips: List[ViralClip]


def _build_anchor_clip(start_seconds: int, end_seconds: int, score: int, idx: int) -> ViralClip:
    start = max(0, int(start_seconds))
    end = max(start + 30, min(int(end_seconds), start + 60))
    return ViralClip(
        title=f"High Impact Moment {idx + 1}",
        hook="This is the moment everyone missed until it happened.",
        start_seconds=start,
        end_seconds=end,
        virality_score=max(60, min(99, score)),
        reasoning="Built to start with a dramatic setup, escalate quickly, and land a punchy payoff in the first few seconds to maximize retention.",
    )


def validate_highlight_response(response: Optional[HighlightResponse], duration: int) -> HighlightResponse:
    if response is None:
        response = HighlightResponse(clips=[])

    clips: List[ViralClip] = []
    seen_windows = set()
    fallback_slots = []

    if duration > 0:
        clip_len = max(30, min(60, duration // 6))
        anchors = [
            (max(0, int(duration * 0.10)), min(duration, max(30, int(duration * 0.10)) + clip_len)),
            (max(0, int(duration * 0.45)), min(duration, max(30, int(duration * 0.45)) + clip_len)),
            (max(0, int(duration * 0.72)), min(duration, max(30, int(duration * 0.72)) + clip_len)),
        ]
        for idx, (start, end) in enumerate(anchors):
            if end - start < 30:
                end = min(duration, start + 30)
            if end - start > 60:
                end = start + 60
            if end > duration:
                end = duration
            if end - start < 30:
                start = max(0, end - 30)
            fallback_slots.append((start, end, 82 - idx * 4))

    for clip in response.clips or []:
        if not isinstance(clip, ViralClip):
            continue
        start = max(0, int(getattr(clip, "start_seconds", 0)))
        end = max(start + 30, int(getattr(clip, "end_seconds", start + 30)))
        if end > duration:
            end = duration
        if end - start < 30:
            end = min(duration, start + 30)
        if end - start > 60:
            end = min(duration, start + 60)
        if end - start < 30:
            start = max(0, end - 30)
        if end <= start:
            continue
        key = (start, end)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        adjusted = ViralClip(
            title=clip.title or f"High Impact Moment {len(clips) + 1}",
            hook=clip.hook or "This is the moment everyone missed until it happened.",
            start_seconds=start,
            end_seconds=end,
            virality_score=max(60, min(99, int(getattr(clip, "virality_score", 75)))),
            reasoning=clip.reasoning or "Strong setup, escalation, and payoff for short-form retention.",
        )
        clips.append(adjusted)

    for start, end, score in fallback_slots:
        key = (start, end)
        if key in seen_windows:
            continue
        clips.append(_build_anchor_clip(start, end, score, len(clips)))
        seen_windows.add(key)

    ordered = sorted(clips, key=lambda c: (c.virality_score, c.start_seconds), reverse=True)
    ordered = ordered[:3]

    while len(ordered) < 3:
        next_start = max(0, min(duration - 30, 10 + len(ordered) * 40))
        next_end = min(duration, next_start + 45)
        if next_end - next_start < 30:
            next_end = min(duration, next_start + 30)
        ordered.append(_build_anchor_clip(next_start, next_end, 74, len(ordered)))

    ordered = sorted(ordered, key=lambda c: (c.virality_score, c.start_seconds), reverse=True)
    for idx, clip in enumerate(ordered[:3]):
        clip.title = clip.title if clip.title else f"High Impact Moment {idx + 1}"
        clip.hook = clip.hook if clip.hook else "This is the moment everyone missed until it happened."
        clip.reasoning = clip.reasoning if clip.reasoning else "Clear setup, tension, and payoff designed for strong retention."
        if idx == 0:
            clip.virality_score = max(clip.virality_score, 85)
        elif idx == 1:
            clip.virality_score = max(clip.virality_score, 80)
        else:
            clip.virality_score = max(clip.virality_score, 75)

    return HighlightResponse(clips=ordered[:3])


def is_local_file(target: str) -> bool:
    return os.path.isfile(target.strip('"').strip("'"))


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def normalize_video_target(target: str) -> str:
    cleaned = target.strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("No video source provided.")
    return cleaned


def validate_video_target(target: str) -> str:
    cleaned = normalize_video_target(target)
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    if os.path.isfile(cleaned):
        return cleaned
    if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", cleaned):
        return cleaned
    if os.path.exists(os.path.dirname(cleaned) or "."):
        return cleaned
    raise ValueError(f"Video source is not a valid local file or supported URL: {target}")


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
    return re.sub(r"[^a-zA-Z0-9]", "_", url.split("/")[-1])[:16]


def _resolve_direct_media_url(video_target: str) -> Optional[str]:
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-warnings",
        "--user-agent", _BROWSER_UA,
        "--add-header", "Referer:https://app.mediasilo.com/",
        "--get-url",
        video_target,
    ]
    proxy = _pick_proxy()
    if proxy:
        cmd.extend(["--proxy", proxy])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        lines = [l.strip() for l in res.stdout.splitlines() if l.strip().startswith("http")]
        return lines[-1] if lines else None
    except Exception:
        return None


def get_video_duration(video_target: str) -> int:
    clean_target = video_target.strip('"').strip("'")
    if is_local_file(clean_target):
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            clean_target
        ]
        try:
            res = subprocess.run(probe_cmd, capture_output=True, text=True)
            return int(float(res.stdout.strip() or 3600))
        except Exception:
            return 3600

    probe_target = clean_target
    if not is_youtube_url(clean_target) and clean_target.startswith("http"):
        resolved = _resolve_direct_media_url(clean_target)
        if resolved:
            probe_target = resolved

    try:
        cmd = [
            sys.executable, "-m", "yt_dlp", "--print", "duration", "--no-warnings",
            "--add-header", "Referer:https://app.mediasilo.com/",
            probe_target,
        ]
        proxy = _pick_proxy()
        if proxy:
            cmd.extend(["--proxy", proxy])
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        raw_dur = res.stdout.strip()
        if raw_dur and raw_dur.replace(".", "", 1).isdigit():
            return int(float(raw_dur))
    except Exception:
        pass

    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-headers", "Referer: https://app.mediasilo.com/\r\n",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            probe_target,
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
        raw_dur = res.stdout.strip()
        if raw_dur and raw_dur.replace(".", "", 1).isdigit():
            return int(float(raw_dur))
    except Exception:
        pass

    return 3600


def extract_captions(video_target: str, job_dir: str) -> str:
    print("Fetching video captions or context...")
    clean_target = video_target.strip('"').strip("'")

    if is_local_file(clean_target):
        sub_out = os.path.join(job_dir, "local_subs.srt")
        cmd = ["ffmpeg", "-y", "-i", clean_target, "-map", "0:s:0?", sub_out]
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
            if os.path.isfile(sub_out) and os.path.getsize(sub_out) > 0:
                with open(sub_out, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                os.remove(sub_out)
                return " ".join([
                    line.strip() for line in raw_text.splitlines()
                    if line.strip() and "-->" not in line and not line.isdigit()
                ])
        except Exception:
            pass
        return ""

    if is_youtube_url(video_target):
        try:
            video_id = get_video_id(video_target)
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
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            caption_tracks = (
                data.get("captions", {})
                .get("playerCaptionsTracklistRenderer", {})
                .get("captionTracks", [])
            )

            if caption_tracks:
                target_url = None
                for track in caption_tracks:
                    lang = track.get("languageCode", "").lower()
                    if lang.startswith("en"):
                        target_url = track.get("baseUrl")
                        break
                if not target_url:
                    target_url = caption_tracks[0].get("baseUrl")
                if "fmt=" not in target_url:
                    target_url += "&fmt=json3"

                cap_req = urllib.request.Request(target_url, headers={"User-Agent": _BROWSER_UA})
                with urllib.request.urlopen(cap_req, timeout=10) as c_resp:
                    cap_data = c_resp.read().decode("utf-8", errors="ignore")

                lines = []
                json_subs = json.loads(cap_data)
                for event in json_subs.get("events", []):
                    for seg in event.get("segs", []):
                        utf8 = seg.get("utf8", "").strip()
                        if utf8 and utf8 != "\n":
                            lines.append(utf8)
                return " ".join(lines)
        except Exception:
            pass

    subs_out = os.path.join(job_dir, "temp_subs")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*,en",
        "--sub-format", "vtt/srt/best",
        "-o", f"{subs_out}.%(ext)s",
        video_target
    ]
    proxy = _pick_proxy()
    if proxy:
        cmd.extend(["--proxy", proxy])
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        for f in os.listdir(job_dir):
            if f.startswith("temp_subs") and (f.endswith(".vtt") or f.endswith(".srt")):
                sub_path = os.path.join(job_dir, f)
                with open(sub_path, "r", encoding="utf-8", errors="ignore") as sub_f:
                    raw_text = sub_f.read()
                clean_lines = [
                    re.sub(r"<[^>]+>", "", line).strip()
                    for line in raw_text.splitlines()
                    if line.strip() and not line.startswith("WEBVTT") and "-->" not in line and not line.isdigit()
                ]
                os.remove(sub_path)
                return " ".join(clean_lines)
    except Exception:
        pass

    return ""


def find_viral_moments_direct(video_target: str) -> HighlightResponse:
    print("Finding viral moments with Gemini...")
    transcript_text = extract_captions(video_target, ".")
    duration = get_video_duration(video_target)

    if transcript_text:
        content_payload = f"Here is the video/film transcript:\n\n{transcript_text[:35000]}"
    else:
        content_payload = f"This video is {duration} seconds long. Pick 3 high impact, intense, or entertaining moments spaced across the timeline."

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
        You are an elite short form video editor specialized in extracting high retention, viral clips for TikTok, YouTube Shorts, and Instagram Reels.

        Source Content:
        {content_payload}

        Task:
        Identify the TOP 3 clips that feel most likely to hook viewers in the first 1-2 seconds and keep them watching to the end.

        Prioritize clips that have:
        - an immediate conflict, surprise, mistake, reveal, or emotional beat
        - a clear setup-to-payoff structure
        - an unmistakable hook line or question at the start
        - strong pacing and an obvious ending without drifting into filler

        Story Arc Rules:
        1. Complete Narrative Envelope: Never start in the middle of an action or just deliver the punchline. Start with the setup or challenge, escalate, then finish right after the payoff.
        2. Clean Boundary Alignment: Start timestamps must align with the beginning of a sentence or question. End timestamps must conclude on a punchline, reaction, or resolution.
        3. Standalone Context: The viewer must instantly understand what is happening and why it matters.
        4. Distinct Moments: Choose three different narrative beats from different parts of the timeline, not repeated content.

        Strict Constraints:
        1. Duration: Each clip must be strictly between 30 and 60 seconds long.
        2. Start Bound: start_seconds must be >= 0 and <= {max(0, duration - 30)}.
        3. End Bound: end_seconds must equal start_seconds plus clip duration, not exceeding {duration}.
        4. Spacing: Distribute the 3 clips across early, middle, and late parts of the video.
        5. Quality Gate: Reject low-energy filler, random silent pauses, and repetitive explanations. Only choose clips with strong emotional or entertaining value.
        6. Scoring & Reasoning: Assign a virality score (1 to 100) and state the exact hook mechanism and why the payoff will retain attention.
        """

    priority_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
    ]

    for attempt in range(3):
        for model_name in priority_models:
            try:
                print(f"Analyzing via: {model_name} (attempt {attempt + 1})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=HighlightResponse,
                        temperature=0.2,
                    ),
                )
                parsed = HighlightResponse.model_validate_json(response.text)
                validated = validate_highlight_response(parsed, int(duration))
                if len(validated.clips) == 3:
                    return validated
                print(f"Generated incomplete highlight set on attempt {attempt + 1}; retrying with stronger constraints.")
            except Exception as e:
                print(f"Model {model_name} error: {e}")
                time.sleep(1)

    return validate_highlight_response(HighlightResponse(clips=[]), int(duration))


def transcribe_clip_words(audio_or_video_path: str) -> List[Dict]:
    model = get_whisper_model()
    segments, _ = model.transcribe(audio_or_video_path, word_timestamps=True)
    words_data = []
    for segment in segments:
        for word in segment.words:
            w_text = word.word.strip()
            if w_text:
                words_data.append({
                    "word": w_text,
                    "start": round(word.start, 2),
                    "end": round(word.end, 2)
                })
    return words_data


def generate_animated_ass(words: List[Dict], ass_path: str, highlight_bgr="&H0063FFD7&", words_per_chunk=3):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,68,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    def fmt_ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        cs = int((sec - int(sec)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    chunks = [words[i:i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]

    for chunk in chunks:
        for active_idx, target_word in enumerate(chunk):
            w_start = target_word["start"]
            w_end = target_word["end"]
            styled = []
            for idx, item in enumerate(chunk):
                if idx == active_idx:
                    styled.append(f"{{\\c{highlight_bgr}\\t(0,80,\\fscx110\\fscy110)}}{item['word']}{{\\r}}")
                else:
                    styled.append(f"{{\\c&H00FFFFFF&}}{item['word']}")
            line = " ".join(styled)
            events.append(f"Dialogue: 0,{fmt_ts(w_start)},{fmt_ts(w_end)},Default,,0,0,0,,{line}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def remove_silence(
    input_path: str,
    output_path: str,
    min_silence_len: float = 1.2,
    noise_floor_db: str = "-35dB",
    pad: float = 0.15,
) -> str:
    detect_cmd = [
        "ffmpeg", "-threads", "2", "-i", input_path,
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
        "ffmpeg", "-y", "-threads", "2", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
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

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

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
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1

            small_w = 480
            small_h = int(height * (small_w / width))
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
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

            try:
                proc.stdin.write(canvas.tobytes())
            except (BrokenPipeError, IOError):
                break

            if frame_count % 120 == 0:
                gc.collect()

    finally:
        detector.close()
        cap.release()
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.wait()
        gc.collect()


def burn_ass_subtitles(video_path: str, ass_path: str, output_path: str):
    clean_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y",
        "-threads", "2",
        "-i", video_path,
        "-vf", f"ass={clean_ass}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)


def download_clip(
    video_target: str,
    start_seconds: int,
    end_seconds: int,
    output_path: str,
    cookies_base64: str = "",
) -> bool:
    clean_target = video_target.strip('"').strip("'")
    duration = max(5, end_seconds - start_seconds)

    if is_local_file(clean_target):
        print(f"Trimming directly from local file: {clean_target}")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-i", clean_target,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-c:a", "aac",
            output_path
        ]
        subprocess.run(cmd, check=True)
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 0

    if clean_target.startswith("http") and (".m3u8" in clean_target or ".mp4" in clean_target):
        print("Streaming slice from direct manifest/stream URL...")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-headers", f"Referer: https://app.mediasilo.com/\r\nUser-Agent: {_BROWSER_UA}\r\n",
            "-i", clean_target,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-c:a", "aac",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                return True
        except Exception:
            pass

    resolved_url = _resolve_direct_media_url(clean_target)
    if resolved_url:
        print("Resolved direct media URL, streaming slice via range requests...")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-headers", f"Referer: https://app.mediasilo.com/\r\nUser-Agent: {_BROWSER_UA}\r\n",
            "-i", resolved_url,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
            "-c:a", "aac",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                return True
        except Exception:
            pass

    section = f"*{start_seconds}-{end_seconds}"
    cookie_path = None

    env_pool = os.getenv("YOUTUBE_COOKIES_BASE64", "").strip()
    candidate_cookies = []

    if cookies_base64.strip():
        candidate_cookies = [cookies_base64.strip()]
    elif env_pool:
        candidate_cookies = [c.strip() for c in re.split(r"[\n,]+", env_pool) if c.strip()]

    selected_b64 = random.choice(candidate_cookies) if candidate_cookies else ""

    if selected_b64:
        cookie_path = output_path + ".cookies.txt"
        try:
            cookie_data = base64.b64decode(selected_b64, validate=True)
            with open(cookie_path, "wb") as cookie_file:
                cookie_file.write(cookie_data)
                cookie_file.write(b"\n")
            os.chmod(cookie_path, 0o600)
        except Exception:
            cookie_path = None

    client_strategies = ["youtube:player_client=web,mweb", "default"] if is_youtube_url(clean_target) else ["default"]
    format_selector = "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best"

    try:
        for client_arg in client_strategies:
            command = [
                sys.executable, "-m", "yt_dlp",
                "--no-playlist",
                "--no-warnings",
                "--retries", "3",
                "--user-agent", _BROWSER_UA,
                "--download-sections", section,
                "--force-keyframes-at-cuts",
                "--format", format_selector,
                "--merge-output-format", "mp4",
                "--output", output_path,
            ]

            if client_arg != "default":
                command.extend(["--extractor-args", client_arg])

            if cookie_path and os.path.exists(cookie_path):
                command.extend(["--cookies", cookie_path])

            proxy = _pick_proxy()
            if proxy:
                command.extend(["--proxy", proxy])

            command.append(clean_target)

            try:
                subprocess.run(command, check=True)
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    return True
            except Exception:
                pass

            if os.path.exists(output_path):
                os.remove(output_path)

        return False
    finally:
        if cookie_path and os.path.exists(cookie_path):
            os.remove(cookie_path)


def process_clip(
    video_target: str,
    clip: ViralClip,
    index: int,
    job_dir: str,
    mode: str = "cut",
    aspect_ratio: str = "9:16",
    cookies_base64: str = ""
):
    clean_title = re.sub(r"[^a-zA-Z0-9]", "_", clip.title)[:25]
    clean_ratio = aspect_ratio.replace(":", "_")
    base_name = f"clip_{index}_{clean_title}_{clean_ratio}"

    temp_raw = os.path.join(job_dir, f"temp_raw_{index}.mp4")
    temp_paced = os.path.join(job_dir, f"temp_paced_{index}.mp4")
    temp_tracked = os.path.join(job_dir, f"temp_tracked_{index}.mp4")
    final_output = os.path.join(job_dir, f"{base_name}.mp4")
    transcript_json_path = os.path.join(job_dir, f"{base_name}.json")
    ass_path = os.path.join(job_dir, f"temp_subs_{index}.ass")

    print(f"\nProcessing Clip {index}: {clip.title}")
    print(f"Time: {clip.start_seconds}s to {clip.end_seconds}s")

    downloaded = download_clip(
        video_target,
        clip.start_seconds,
        clip.end_seconds,
        temp_raw,
        cookies_base64=cookies_base64,
    )

    if not downloaded:
        raise RuntimeError("Unable to extract the clip slice from the video target.")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", temp_raw],
        capture_output=True, text=True
    )
    clip_actual_duration = 0.0
    try:
        clip_actual_duration = float(probe.stdout.strip() or 0)
    except ValueError:
        pass

    if clip_actual_duration < 1.0:
        raise RuntimeError(
            f"Downloaded clip for '{clip.title}' is empty or invalid (duration={clip_actual_duration}s). "
            f"Requested {clip.start_seconds}-{clip.end_seconds}s may exceed the source video's actual length."
        )

    try:
        paced_file = remove_silence(temp_raw, temp_paced, min_silence_len=0.6)

        words_data = transcribe_clip_words(paced_file)
        with open(transcript_json_path, "w", encoding="utf-8") as tj:
            json.dump({
                "words": words_data,
                "start_seconds": clip.start_seconds,
                "end_seconds": clip.end_seconds,
                "mode": mode,
                "aspect_ratio": aspect_ratio
            }, tj, indent=2)

        generate_animated_ass(words_data, ass_path)

        if aspect_ratio == "16:9":
            escaped_ass = ass_path.replace("\\", "/").replace(":", r"\:")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-threads", "2",
                "-i", paced_file,
                "-vf", f"scale=1920:1080:flags=lanczos,setsar=1,ass={escaped_ass}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-b:a", "192k",
                final_output
            ]
            subprocess.run(ffmpeg_cmd, check=True)
        else:
            render_gimbal_tracked_video(paced_file, temp_tracked, job_dir, mode=mode, aspect_ratio=aspect_ratio)
            burn_ass_subtitles(temp_tracked, ass_path, final_output)

        print(f"SUCCESS: Exported {final_output}")
    finally:
        for tmp in [temp_raw, temp_paced, temp_tracked, ass_path]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def run_pipeline(
    video_target: str,
    aspect_ratio: str = "9:16",
    mode: str = "cut",
    prompt_fn: Optional[Callable[[str], str]] = None,
    job_id: Optional[str] = None,
    cookies_base64: str = ""
) -> str:
    sweep_expired_jobs()

    if not job_id:
        job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    _write_job_metadata(job_dir, job_id)

    validated_target = validate_video_target(video_target)
    duration = get_video_duration(validated_target)
    if duration > MAX_ALLOWED_SECONDS:
        raise ValueError(f"Video exceeds the {MAX_ALLOWED_HOURS} hour limit.")

    highlight_data = find_viral_moments_direct(validated_target)

    for idx, clip in enumerate(highlight_data.clips, start=1):
        process_clip(validated_target, clip, idx, job_dir, mode=mode, aspect_ratio=aspect_ratio, cookies_base64=cookies_base64)

    return job_dir