FROM python:3.11-slim

# ffmpeg for the render/silence-trim pipeline; libgl1 + libglib2.0-0 are
# needed for opencv-contrib-python's GUI-less bindings to import at all.
# libegl1 + libgles2 are needed for mediapipe's FaceLandmarker C bindings to
# load at all, even headless/CPU-only (it dlopens libEGL.so.1 unconditionally).
# curl + unzip are needed to install the Deno JS runtime below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Deno: yt-dlp now needs a JS runtime to solve YouTube's "n challenge" for
# the web/mweb player clients (required for cookie-authenticated requests).
# See: https://github.com/yt-dlp/yt-dlp/wiki/EJS
RUN curl -fsSL https://deno.land/install.sh | sh \
    && cp /root/.deno/bin/deno /usr/local/bin/deno \
    && chmod 755 /usr/local/bin/deno \
    && deno --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache-buster: ADD from a URL forces Docker to re-check freshness on every
# build (unlike a bare RUN, which gets cached indefinitely if nothing above
# it changed). This ensures `pip install -U yt-dlp` actually pulls the
# latest version each deploy instead of silently reusing a stale layer.
ADD https://pypi.org/pypi/yt-dlp/json /tmp/ytdlp_version.json
RUN pip install --no-cache-dir -U "yt-dlp[default]" yt-dlp-get-pot yt-dlp-get-pot-rustypipe

COPY . .

# jobs/ and models/ should be a mounted volume in production so job output
# and the downloaded face_landmarker.task survive restarts/redeploys.
RUN mkdir -p jobs models

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]