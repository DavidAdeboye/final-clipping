"""
Thin FastAPI wrapper around app.run_pipeline.

Why this exists: the picker functions in app.py (facecam box,
reaction PIP box, "did the PIP move at this cut") were written around
input(), which only works over a real terminal. This server runs the
pipeline in a background thread per job and swaps input() for a
WebPrompt that blocks the *job thread* (not the HTTP server) until the
person answers through the picker page.

Deploy target: a normal long-running host with a writable disk and
ffmpeg installed (Railway / Render / Fly.io / a small VPS) -- NOT
Vercel or other serverless-function platforms. This process needs to
stay alive for minutes per clip and read/write files across requests,
which serverless functions aren't built for (short execution caps, no
persistent disk between invocations).

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os
import threading
import time
import traceback
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app as core

app = FastAPI(title="Viral Clipper")

# ---------------------------------------------------------------------------
# In-memory job store. Fine for a single-process deploy; if you outgrow one
# instance, swap this dict for Redis and keep the same interface.
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.status = "queued"          # queued | running | awaiting_input | done | error
        self.error: Optional[str] = None
        self.result_dir: Optional[str] = None
        self.clips: list = []
        # the current outstanding question, if status == awaiting_input
        self.question: Optional[str] = None
        self.frames: list = []          # [(timestamp, filename), ...] for the picker to show
        self._answer_event = threading.Event()
        self._answer: Optional[str] = None
        self.thread: Optional[threading.Thread] = None


JOBS: Dict[str, Job] = {}


class WebPrompt:
    """
    Drop-in replacement for the CLI's input(). Called from inside the
    pipeline's background thread. Blocks that thread (never the HTTP
    server) until POST /jobs/{id}/respond delivers an answer.
    """
    def __init__(self, job: Job):
        self.job = job

    def ask(self, question: str) -> str:
        job = self.job
        # Preview frames were just written into job_dir by the picker
        # functions right before they call ask() -- pick up whatever's
        # newest so the picker page has something to show.
        job_dir = os.path.join(core.JOBS_ROOT, job.id)
        frame_files = sorted(
            f for f in os.listdir(job_dir)
            if f.startswith("preview_frame") and f.endswith(".png")
        ) if os.path.isdir(job_dir) else []

        job.question = question.strip()
        job.frames = frame_files
        job.status = "awaiting_input"
        job._answer_event.clear()

        job._answer_event.wait()  # blocks this job's thread only

        job.status = "running"
        answer = job._answer or ""
        job._answer = None
        return answer

    def __call__(self, question: str) -> str:
        return self.ask(question)


def _run_job(job: Job, youtube_url: str, aspect_ratio: str, mode: str):
    job.status = "running"
    try:
        job_dir = core.run_pipeline(
            youtube_url,
            aspect_ratio=aspect_ratio,
            mode=mode,
            prompt_fn=WebPrompt(job),
        )
        job.result_dir = job_dir
        job.clips = sorted(
            f for f in os.listdir(job_dir) if f.startswith("clip_") and f.endswith(".mp4")
        )
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class StartJobRequest(BaseModel):
    youtube_url: str
    aspect_ratio: str = "9:16"     # "9:16" | "1:1" | "16:9"
    mode: str = "split"            # split | cut | speaker_switch | gaming | gaming_split_noface | reaction


@app.post("/jobs")
def start_job(req: StartJobRequest):
    job_id = str(uuid.uuid4())
    job = Job(job_id)
    JOBS[job_id] = job
    job.thread = threading.Thread(
        target=_run_job, args=(job, req.youtube_url, req.aspect_ratio, req.mode), daemon=True
    )
    job.thread.start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id,
        "status": job.status,
        "error": job.error,
        "question": job.question if job.status == "awaiting_input" else None,
        "frames": [f"/jobs/{job.id}/frame/{f}" for f in job.frames] if job.status == "awaiting_input" else [],
        "clips": [f"/jobs/{job.id}/clip/{c}" for c in job.clips] if job.status == "done" else [],
    }


class RespondRequest(BaseModel):
    # Expected format: "x,y,w,h" as fractions 0.0-1.0 (or "" to mean
    # "no change" for the mid-clip cut re-prompt).
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


@app.get("/jobs/{job_id}/frame/{filename}")
def get_frame(job_id: str, filename: str):
    path = os.path.join(core.JOBS_ROOT, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "frame not found")
    return FileResponse(path, media_type="image/png")


@app.get("/jobs/{job_id}/clip/{filename}")
def get_clip(job_id: str, filename: str):
    path = os.path.join(core.JOBS_ROOT, job_id, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/", response_class=HTMLResponse)
def index():
    picker_path = os.path.join(os.path.dirname(__file__), "static", "picker.html")
    with open(picker_path, "r") as f:
        return f.read()


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")