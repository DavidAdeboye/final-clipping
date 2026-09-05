import os
import json
import re
import threading
import time
import traceback
import uuid
import shutil
import subprocess
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app as core

app = FastAPI(title="AutoClip Studio")
UPLOAD_ROOT = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)


def _job_meta_path(job_id: str) -> str:
    return os.path.join(core.JOBS_ROOT, job_id, core.JOB_METADATA_FILENAME)


def _persist_job_state(job_id: str, **fields):
    job_dir = os.path.join(core.JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)
    meta_path = _job_meta_path(job_id)
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta.update(fields)
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    except Exception:
        traceback.print_exc()


class Job:
    def __init__(self, job_id: str, status: str = "queued"):
        self.id = job_id
        self.status = status
        self.error: Optional[str] = None
        self.result_dir: Optional[str] = None
        self.clips: list = []
        self.question: Optional[str] = None
        self.frames: list = []
        self._answer_event = threading.Event()
        self._answer: Optional[str] = None
        self.thread: Optional[threading.Thread] = None

    def set_status(self, status: str, error: Optional[str] = None):
        self.status = status
        if error is not None:
            self.error = error
        _persist_job_state(self.id, status=status, error=self.error)


JOBS: Dict[str, Job] = {}


def _rehydrate_jobs_from_disk():
    if not os.path.isdir(core.JOBS_ROOT):
        return

    for job_id in os.listdir(core.JOBS_ROOT):
        job_dir = os.path.join(core.JOBS_ROOT, job_id)
        if not os.path.isdir(job_dir):
            continue

        meta_path = os.path.join(job_dir, core.JOB_METADATA_FILENAME)
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                pass

        clips = sorted(
            f for f in os.listdir(job_dir) if f.startswith("clip_") and f.endswith(".mp4")
        )
        prior_status = meta.get("status")

        job = Job(job_id)
        if clips:
            job.status = "done"
        elif prior_status in ("queued", "running", "awaiting_input"):
            job.status = "error"
            job.error = "Server restarted while this job was in progress. Please start a new clip job."
        elif prior_status == "error":
            job.status = "error"
            job.error = meta.get("error") or "Job failed."
        else:
            continue

        JOBS[job_id] = job
        _persist_job_state(job_id, status=job.status, error=job.error)


_rehydrate_jobs_from_disk()


class WebPrompt:
    def __init__(self, job: Job):
        self.job = job

    def ask(self, question: str) -> str:
        job = self.job
        job_dir = os.path.join(core.JOBS_ROOT, job.id)
        frame_files = sorted(
            f for f in os.listdir(job_dir)
            if f.startswith("preview_frame") and f.endswith(".png")
        ) if os.path.isdir(job_dir) else []

        job.question = question.strip()
        job.frames = frame_files
        job.set_status("awaiting_input")
        job._answer_event.clear()
        job._answer_event.wait()

        job.set_status("running")
        answer = job._answer or ""
        job._answer = None
        return answer

    def __call__(self, question: str) -> str:
        return self.ask(question)


def _run_job(job: Job, youtube_url: str, aspect_ratio: str, mode: str, cookies_base64: str = ""):
    prompt_handler = WebPrompt(job)
    try:
        job.set_status("running")
        job_dir = core.run_pipeline(
            youtube_url,
            aspect_ratio=aspect_ratio,
            mode=mode,
            prompt_fn=prompt_handler,
            job_id=job.id,
            cookies_base64=cookies_base64
        )
        job.result_dir = job_dir
        clips = sorted(
            f for f in os.listdir(job_dir) if f.startswith("clip_") and f.endswith(".mp4")
        ) if os.path.isdir(job_dir) else []
        job.clips = clips
        job.set_status("done")
    except Exception as e:
        job.set_status("error", error=str(e))
        traceback.print_exc()


class StartJobRequest(BaseModel):
    youtube_url: str
    aspect_ratio: str = "9:16"
    mode: str = "cut"
    client_id: Optional[str] = None
    cookies_base64: Optional[str] = None


class ManualEditRequest(BaseModel):
    filename: str
    words: List[Dict]
    aspect_ratio: str = "9:16"
    mode: str = "cut"


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#14171e"/><stop offset="1" stop-color="#07080a"/></linearGradient></defs>
      <rect width="512" height="512" rx="112" fill="url(#g)"/>
      <path d="M168 376 L232 144 C236 130 248 120 264 120 C280 120 292 130 296 144 L360 376 C364 390 354 404 340 404 C328 404 318 396 314 384 L294 312 L218 312 L198 384 C194 396 184 404 172 404 C158 404 148 390 152 376 Z" fill="#d7ff63"/>
      <path d="M256 196 L284 272 L228 272 Z" fill="#07080a"/>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file selected")

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(file.filename)) or "uploaded_video"
    target_path = os.path.join(UPLOAD_ROOT, f"{uuid.uuid4()}_{safe_name}")

    with open(target_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    return {"video_path": target_path}


@app.post("/jobs")
def create_job(req: StartJobRequest):
    job_id = str(uuid.uuid4())
    job = Job(job_id)
    JOBS[job_id] = job

    _persist_job_state(
        job_id,
        client_id=req.client_id,
        created_at=time.time(),
        status=job.status,
        error=None,
    )

    t = threading.Thread(
        target=_run_job,
        args=(job, req.youtube_url, req.aspect_ratio, req.mode, (req.cookies_base64 or "")[:2_000_000]),
        daemon=True,
    )
    job.thread = t
    t.start()

    return {"job_id": job_id}


@app.get("/projects")
def list_projects(client_id: Optional[str] = None):
    projects = []
    if not os.path.isdir(core.JOBS_ROOT):
        return projects

    for jid in os.listdir(core.JOBS_ROOT):
        jdir = os.path.join(core.JOBS_ROOT, jid)
        if not os.path.isdir(jdir):
            continue

        meta_path = os.path.join(jdir, core.JOB_METADATA_FILENAME)
        created_at = os.path.getmtime(jdir)
        job_client = None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    created_at = meta.get("created_at", created_at)
                    job_client = meta.get("client_id")
            except Exception:
                pass

        if client_id and job_client != client_id:
            continue

        clips = sorted(
            f for f in os.listdir(jdir) if f.startswith("clip_") and f.endswith(".mp4")
        )
        if clips:
            projects.append({
                "job_id": jid,
                "created_at": created_at,
                "clips_count": len(clips),
                "first_clip": f"/jobs/{jid}/clip/{clips[0]}",
                "clips": [f"/jobs/{jid}/clip/{c}" for c in clips]
            })

    projects.sort(key=lambda p: p["created_at"], reverse=True)
    return projects


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        job_dir = os.path.join(core.JOBS_ROOT, job_id)
        if not os.path.isdir(job_dir):
            raise HTTPException(404, "job not found")

        clips = sorted(
            f for f in os.listdir(job_dir) if f.startswith("clip_") and f.endswith(".mp4")
        )
        job = Job(job_id)
        job.status = "done" if clips else "error"
        JOBS[job_id] = job

    job_dir = os.path.join(core.JOBS_ROOT, job.id)
    live_clips = []
    if os.path.isdir(job_dir):
        raw_clips = sorted(
            f for f in os.listdir(job_dir) if f.startswith("clip_") and f.endswith(".mp4")
        )
        for c in raw_clips:
            cpath = os.path.join(job_dir, c)
            if os.path.getsize(cpath) > 10000 and (time.time() - os.path.getmtime(cpath) > 1.5):
                live_clips.append(c)

    return {
        "id": job.id,
        "status": job.status,
        "error": job.error,
        "question": job.question if job.status == "awaiting_input" else None,
        "frames": [f"/jobs/{job.id}/frame/{f}" for f in job.frames] if job.status == "awaiting_input" else [],
        "clips": [f"/jobs/{job.id}/clip/{c}" for c in live_clips],
    }


@app.get("/jobs/{job_id}/clip-data/{filename}")
def get_clip_metadata(job_id: str, filename: str):
    job_dir = os.path.join(core.JOBS_ROOT, job_id)
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(job_dir, f"{base_name}.json")

    words = []
    start_seconds = 0
    end_seconds = 60
    mode = "cut"
    aspect_ratio = "9:16"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            words = data.get("words", [])
            start_seconds = data.get("start_seconds", 0)
            end_seconds = data.get("end_seconds", 60)
            mode = data.get("mode", "cut")
            aspect_ratio = data.get("aspect_ratio", "9:16")

    return {
        "job_id": job_id,
        "filename": filename,
        "video_url": f"/jobs/{job_id}/clip/{filename}",
        "words": words,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "mode": mode,
        "aspect_ratio": aspect_ratio
    }


@app.post("/jobs/{job_id}/re-render")
def rerender_clip(job_id: str, req: ManualEditRequest):
    job_dir = os.path.join(core.JOBS_ROOT, job_id)
    base_name = os.path.splitext(req.filename)[0]
    json_path = os.path.join(job_dir, f"{base_name}.json")
    ass_path = os.path.join(job_dir, f"{base_name}_manual.ass")
    final_output = os.path.join(job_dir, req.filename)
    temp_retrack = os.path.join(job_dir, f"{base_name}_retrack.mp4")

    if not os.path.exists(final_output):
        raise HTTPException(404, "Original clip not found")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "words": req.words,
            "mode": req.mode,
            "aspect_ratio": req.aspect_ratio
        }, f, indent=2)

    core.generate_animated_ass(req.words, ass_path)

    try:
        if req.aspect_ratio == "16:9":
            escaped_ass = ass_path.replace("\\", "/").replace(":", r"\:")
            cmd = [
                "ffmpeg", "-y", "-i", final_output,
                "-vf", f"ass={escaped_ass}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "copy",
                temp_retrack
            ]
            subprocess.run(cmd, check=True)
        else:
            core.render_gimbal_tracked_video(final_output, temp_retrack, job_dir, mode=req.mode, aspect_ratio=req.aspect_ratio)
            core.burn_ass_subtitles(temp_retrack, ass_path, final_output)

        if os.path.exists(temp_retrack):
            shutil.move(temp_retrack, final_output)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)
        if os.path.exists(temp_retrack):
            os.remove(temp_retrack)

    return {"ok": True, "clip_url": f"/jobs/{job_id}/clip/{req.filename}"}


class RespondRequest(BaseModel):
    answer: str


@app.post("/jobs/{job_id}/respond")
def respond_to_job(job_id: str, req: RespondRequest):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "awaiting_input":
        raise HTTPException(409, "job is not currently waiting on an answer")
    job._answer = req.answer.strip()
    job.question = None
    job.frames = []
    job._answer_event.set()
    return {"ok": True}


def send_bytes_range_requests(file_path: str, start: int, end: int, chunk_size: int = 1024 * 1024):
    with open(file_path, "rb") as f:
        f.seek(start)
        while (pos := f.tell()) <= end:
            read_size = min(chunk_size, end + 1 - pos)
            data = f.read(read_size)
            if not data:
                break
            yield data


@app.get("/jobs/{job_id}/clip/{filename}")
def get_clip(job_id: str, filename: str, request: Request):
    path = os.path.join(core.JOBS_ROOT, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "clip not found")

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")

    if range_header:
        byte1, byte2 = range_header.replace("bytes=", "").split("-")
        start = int(byte1)
        end = int(byte2) if byte2 else file_size - 1
        length = (end - start) + 1

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(
            send_bytes_range_requests(path, start, end),
            status_code=206,
            headers=headers,
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(
        send_bytes_range_requests(path, 0, file_size - 1),
        status_code=200,
        headers=headers,
    )


@app.get("/jobs/{job_id}/frame/{filename}")
def get_frame(job_id: str, filename: str):
    path = os.path.join(core.JOBS_ROOT, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "frame not found")
    with open(path, "rb") as f:
        content = f.read()
    return Response(content=content, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index():
    picker_path = os.path.join(os.path.dirname(__file__), "static", "picker.html")
    with open(picker_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/editor", response_class=HTMLResponse)
def editor_page():
    editor_path = os.path.join(os.path.dirname(__file__), "static", "editor.html")
    with open(editor_path, "r", encoding="utf-8") as f:
        return f.read()


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health_check():
    return {"status": "ok"}