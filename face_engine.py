"""
AttendX Face Recognition Engine
================================
Uses InsightFace (ArcFace buffalo_l) + OpenCV for:
  - Face detection (RetinaFace)
  - Embedding extraction (ArcFace, 512-dim)
  - Image quality validation (brightness, sharpness)
  - Cosine similarity matching
  - Multi-frame embedding averaging
"""

import os
import base64
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Configurable thresholds (override via environment variables) ─────────────
MATCH_THRESHOLD    = float(os.getenv("FACE_MATCH_THRESHOLD",  "0.40"))
DET_SCORE_MIN      = float(os.getenv("FACE_DET_SCORE_MIN",   "0.65"))
BLUR_THRESHOLD     = float(os.getenv("BLUR_THRESHOLD",        "35.0"))
BRIGHTNESS_MIN     = float(os.getenv("BRIGHTNESS_MIN",        "25.0"))
BRIGHTNESS_MAX     = float(os.getenv("BRIGHTNESS_MAX",       "240.0"))
INSIGHTFACE_MODEL  = os.getenv("INSIGHTFACE_MODEL",          "buffalo_l")
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID",    "-1"))   # -1=CPU, 0=GPU


# ─── Singleton Face Engine ────────────────────────────────────────────────────
class FaceEngine:
    """
    Singleton wrapper around InsightFace FaceAnalysis.
    Loads the model once on first call to get_instance().
    """

    _instance: "FaceEngine | None" = None

    @classmethod
    def get_instance(cls) -> "FaceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        logger.info(
            "Initializing InsightFace engine: model=%s, ctx_id=%d",
            INSIGHTFACE_MODEL,
            INSIGHTFACE_CTX_ID,
        )
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. "
                "Run: pip install insightface onnxruntime"
            ) from exc

        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            allowed_modules=["detection", "recognition"],
        )
        # det_size=(640, 640) gives best accuracy for typical portrait selfies
        self.app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=(640, 640))
        logger.info("InsightFace engine ready.")

    # ─── Image decoding ───────────────────────────────────────────────────────
    def decode_image(self, b64_string: str) -> "np.ndarray | None":
        """Decode a base64 JPEG/PNG string to an OpenCV BGR image."""
        try:
            # Strip data-URI prefix if present (data:image/jpeg;base64,...)
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_string)
            buf = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return img
        except Exception as exc:
            logger.warning("Image decode failed: %s", exc)
            return None

    # ─── Quality validation ───────────────────────────────────────────────────
    def check_quality(self, img: "np.ndarray | None") -> dict:
        """
        Validate image quality before running the face model.
        Returns {"valid": True} or {"valid": False, "reason": str}.
        """
        if img is None:
            return {"valid": False, "reason": "Failed to decode image"}

        h, w = img.shape[:2]
        if h < 80 or w < 80:
            return {"valid": False, "reason": "Image resolution too low"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightness (mean pixel intensity, 0–255)
        brightness = float(np.mean(gray))
        if brightness < BRIGHTNESS_MIN:
            return {"valid": False, "reason": "Image too dark — improve lighting"}
        if brightness > BRIGHTNESS_MAX:
            return {"valid": False, "reason": "Image overexposed — reduce brightness"}

        # Sharpness via Laplacian variance (higher = sharper)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < BLUR_THRESHOLD:
            return {"valid": False, "reason": "Image blurry — hold the device still"}

        return {"valid": True, "brightness": round(brightness, 1), "sharpness": round(sharpness, 1)}

    # ─── Face embedding extraction ────────────────────────────────────────────
    def get_face_embedding(self, img: np.ndarray) -> dict:
        """
        Detect the face in `img` and extract a 512-dim L2-normalised ArcFace embedding.
        Returns {"success": True, "embedding": list[float], "det_score": float}
             or {"success": False, "error": str}.
        """
        try:
            faces = self.app.get(img)
        except Exception as exc:
            logger.error("InsightFace detection error: %s", exc)
            return {"success": False, "error": "Face detection model error"}

        if len(faces) == 0:
            return {"success": False, "error": "No face detected — position face in frame"}
        if len(faces) > 1:
            return {
                "success": False,
                "error": f"Multiple faces detected ({len(faces)}) — ensure only one face is visible",
            }

        face = faces[0]
        det_score = float(face.det_score)
        if det_score < DET_SCORE_MIN:
            return {
                "success": False,
                "error": f"Low detection confidence ({det_score:.2f}) — improve lighting or reposition",
            }

        if face.embedding is None:
            return {"success": False, "error": "Embedding extraction failed"}

        # L2-normalise for cosine similarity via simple dot product
        emb = face.embedding.astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return {
            "success": True,
            "embedding": emb.tolist(),
            "det_score": round(det_score, 4),
        }

    # ─── Embedding utilities ──────────────────────────────────────────────────
    def average_embeddings(self, embeddings: list) -> list:
        """Average a list of embeddings then re-normalise to unit length."""
        arr = np.array(embeddings, dtype=np.float64)
        avg = np.mean(arr, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg.tolist()

    def cosine_similarity(self, a: list, b: list) -> float:
        """
        Cosine similarity between two embeddings.
        Assumes both are already L2-normalised (dot product = cosine sim).
        Re-normalises defensively just in case.
        """
        va = np.array(a, dtype=np.float64)
        vb = np.array(b, dtype=np.float64)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va / na, vb / nb))
