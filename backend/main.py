"""
AudioSentry Backend Server
FastAPI real-time audio sentinel service with WebSocket streaming ingestion at 16kHz.
"""

import os
import json
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Flexible import for dsp_engine depending on execution working directory
try:
    from backend.dsp_engine import AudioSentryDSP
except ImportError:
    from dsp_engine import AudioSentryDSP

app = FastAPI(
    title="AudioSentry Real-time Voice Sentinel",
    description="Real-time spectral analysis and deepfake detection engine powered by DSP.",
    version="1.0.0",
)

# Enable CORS for local cross-origin development (e.g. Live Server, Vite, file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Identify directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "online",
        "service": "AudioSentry",
        "sample_rate_hz": 16000,
        "frame_duration_ms": 50,
        "timestamp": time.time(),
    }


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Real-time 16kHz audio stream ingestion endpoint.
    Accepts raw binary Float32 PCM audio chunks (e.g. 50ms = 800 samples)
    and returns real-time spectral metrics, dynamic verdict, and smoothed confidence.
    """
    await websocket.accept()
    # Instantiate dedicated DSP engine with EMA smoothing (alpha=0.08) for this stream
    dsp = AudioSentryDSP(sample_rate=16000, buffer_seconds=1.0, ema_alpha=0.08, rolloff_threshold_hz=5800.0)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Decode 32-bit float array (IEEE 754 little-endian standard in Web Audio)
                if len(raw_bytes) % 4 == 0:
                    samples = np.frombuffer(raw_bytes, dtype=np.float32)
                elif len(raw_bytes) % 2 == 0:
                    # Fallback to int16 PCM if client sent 16-bit integers
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    continue

                if len(samples) > 0:
                    dsp.push_audio(samples)
                    analysis = dsp.analyze()
                    analysis["timestamp"] = round(time.time() * 1000)
                    await websocket.send_text(json.dumps(analysis))

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    if msg_type == "audio_chunk" and "samples" in payload:
                        samples = np.array(payload["samples"], dtype=np.float32)
                        dsp.push_audio(samples)
                        analysis = dsp.analyze()
                        analysis["timestamp"] = round(time.time() * 1000)
                        await websocket.send_text(json.dumps(analysis))

                    elif msg_type == "reset":
                        # Reset ring buffer and state
                        dsp = AudioSentryDSP(sample_rate=16000, buffer_seconds=1.0, ema_alpha=0.08)
                        await websocket.send_text(json.dumps({"status": "reset_completed"}))

                    elif msg_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket session closed with error: {e}")


# Serve frontend directly if files exist in /frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"message": "AudioSentry backend running. Frontend index.html not found."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
