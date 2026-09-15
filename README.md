# AttendX Backend — README

## Setup

1. **Install Python 3.10+** if you haven't already.

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > The first run downloads the InsightFace `buffalo_l` model (~300 MB).

4. **Start the server**:
   ```bash
   python start.py
   ```
   The API will be available at `http://0.0.0.0:8000`.

5. **Find your LAN IP** (so the phone can connect):
   ```bash
   # Windows
   ipconfig
   # macOS/Linux
   ifconfig
   ```
   Look for your IPv4 address under the Wi-Fi adapter (e.g., `192.168.1.105`).

6. **Set the mobile app URL** in `attendx/.env`:
   ```
   EXPO_PUBLIC_BIOMETRIC_API_URL=http://192.168.1.105:8000
   ```

## API Reference

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Server + model liveness probe        |
| POST   | `/register` | Accept frames, return embedding      |
| POST   | `/verify`   | Compare frame to stored embedding    |

### POST `/register`
```json
{
  "frames": ["<base64-jpeg>", "<base64-jpeg>", "<base64-jpeg>"]
}
```
Response:
```json
{
  "success": true,
  "embedding": [0.023, -0.411, ...],  // 512-dim float array
  "frames_used": 3,
  "message": "Registered from 3 frame(s)."
}
```

### POST `/verify`
```json
{
  "frame": "<base64-jpeg>",
  "stored_embedding": [0.023, -0.411, ...]  // 512-dim float array from Firebase
}
```
Response:
```json
{
  "success": true,
  "verified": true,
  "similarity": 0.8742,
  "message": "Match"
}
```

## Environment Variables

| Variable               | Default    | Description                          |
|------------------------|------------|--------------------------------------|
| `HOST`                 | `0.0.0.0`  | Bind address                         |
| `PORT`                 | `8000`     | Port                                 |
| `FACE_VERIFY_THRESHOLD`| `0.40`     | Min cosine similarity to verify      |
| `FACE_MATCH_THRESHOLD` | `0.40`     | Same as above (face_engine.py)       |
| `FACE_DET_SCORE_MIN`   | `0.65`     | Min RetinaFace detection confidence  |
| `BLUR_THRESHOLD`       | `35.0`     | Min Laplacian variance (sharpness)   |
| `INSIGHTFACE_CTX_ID`   | `-1`       | `-1`=CPU, `0`=GPU                    |
