"""
AttendX Biometric API Server
============================
FastAPI server exposing face registration and verification endpoints.
Run on the development PC (LAN) and configure the mobile app with:
  EXPO_PUBLIC_BIOMETRIC_API_URL=http://<your-LAN-IP>:8000

Endpoints:
  GET  /health         — liveness probe
  POST /register       — accept 3-5 frames, return averaged embedding
  POST /verify         — compare one frame against a stored embedding
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from face_engine import FaceEngine

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("attendx.api")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AttendX Biometric API",
    version="2.0.0",
    description="InsightFace / ArcFace face recognition backend for AttendX.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Warm up the engine at startup so the first request isn't slow
@app.on_event("startup")
async def _warm_up() -> None:
    try:
        FaceEngine.get_instance()
        logger.info("FaceEngine warm-up complete.")
    except Exception as exc:
        logger.error("FaceEngine warm-up FAILED: %s", exc)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """
    Client sends 3-5 base64-encoded JPEG frames captured from the camera.
    The server validates quality, extracts ArcFace embeddings, and returns
    the L2-normalised average embedding (512-dim float list).
    """

    frames: list[str]

    @field_validator("frames")
    @classmethod
    def validate_frame_count(cls, v: list[str]) -> list[str]:
        if not (1 <= len(v) <= 20):
            raise ValueError("frames must contain 1–20 images")
        return v


class RegisterResponse(BaseModel):
    success: bool
    embedding: list[float] | None = None
    frames_used: int = 0
    message: str = ""


class VerifyRequest(BaseModel):
    """
    Client sends one base64 frame + the stored 512-dim embedding retrieved
    from Firebase. Server computes cosine similarity and returns the score.
    """

    frame: str
    stored_embedding: list[float]

    @field_validator("stored_embedding")
    @classmethod
    def validate_embedding(cls, v: list[float]) -> list[float]:
        if len(v) != 512:
            raise ValueError("stored_embedding must be 512-dimensional")
        return v


class VerifyResponse(BaseModel):
    success: bool
    verified: bool = False
    similarity: float = 0.0
    message: str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

VERIFY_THRESHOLD = float(os.getenv("FACE_VERIFY_THRESHOLD", "0.40"))


def _engine() -> FaceEngine:
    return FaceEngine.get_instance()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict[str, Any]:
    """Liveness probe — returns 200 if the server and model are ready."""
    return {"status": "ok", "model": "buffalo_s"}


@app.post("/register", response_model=RegisterResponse, tags=["Biometric"])
async def register(req: RegisterRequest) -> RegisterResponse:
    """
    Accept 1-20 base64 frames, validate quality, extract embeddings,
    and return the averaged L2-normalised 512-dim ArcFace embedding.
    """
    engine = _engine()
    good_embeddings: list[list[float]] = []
    errors: list[str] = []

    for idx, b64 in enumerate(req.frames):
        img = engine.decode_image(b64)
        quality = engine.check_quality(img)
        if not quality["valid"]:
            errors.append(f"Frame {idx}: {quality['reason']}")
            continue

        result = engine.get_face_embedding(img)
        if not result["success"]:
            errors.append(f"Frame {idx}: {result['error']}")
            continue

        good_embeddings.append(result["embedding"])

    if not good_embeddings:
        detail = "; ".join(errors) if errors else "No usable frames"
        raise HTTPException(status_code=422, detail=detail)

    avg_embedding = engine.average_embeddings(good_embeddings)

    logger.info(
        "register: %d/%d frames used (%.0f%% yield)",
        len(good_embeddings),
        len(req.frames),
        100 * len(good_embeddings) / len(req.frames),
    )

    return RegisterResponse(
        success=True,
        embedding=avg_embedding,
        frames_used=len(good_embeddings),
        message=f"Registered from {len(good_embeddings)} frame(s).",
    )


@app.post("/verify", response_model=VerifyResponse, tags=["Biometric"])
async def verify(req: VerifyRequest) -> VerifyResponse:
    """
    Accept one base64 frame + the stored embedding from Firebase.
    Returns cosine similarity and a boolean `verified` flag.
    """
    engine = _engine()

    img = engine.decode_image(req.frame)
    quality = engine.check_quality(img)
    if not quality["valid"]:
        return VerifyResponse(
            success=True,
            verified=False,
            similarity=0.0,
            message=quality["reason"],
        )

    result = engine.get_face_embedding(img)
    if not result["success"]:
        return VerifyResponse(
            success=True,
            verified=False,
            similarity=0.0,
            message=result["error"],
        )

    similarity = engine.cosine_similarity(result["embedding"], req.stored_embedding)
    verified = similarity >= VERIFY_THRESHOLD

    logger.info(
        "verify: similarity=%.4f threshold=%.4f → %s",
        similarity,
        VERIFY_THRESHOLD,
        "PASS" if verified else "FAIL",
    )

    return VerifyResponse(
        success=True,
        verified=verified,
        similarity=round(similarity, 4),
        message="Match" if verified else f"Mismatch (score {similarity:.2f})",
    )


# ─── Entry point (python main.py) ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting AttendX Biometric API on %s:%d", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
