FROM python:3.11-slim

# ffmpeg for the render/silence-trim pipeline; libgl1 + libglib2.0-0 are
# needed for opencv-contrib-python's GUI-less bindings to import at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yt-dlp as a standalone binary tends to stay more current than the pip
# package alone for extractor fixes -- installing both, pip version is the
# one actually imported/used if your code shells out to `yt-dlp` on PATH
# this picks up whichever resolves first, which will be this one.
ADD https://pypi.org/pypi/yt-dlp/json /tmp/ytdlp_version.json
RUN pip install --no-cache-dir -U yt-dlp

COPY . .

# jobs/ and models/ should be a mounted volume in production so job output
# and the downloaded face_landmarker.task survive restarts/redeploys.
RUN mkdir -p jobs models

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
