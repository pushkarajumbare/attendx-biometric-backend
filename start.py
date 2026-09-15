"""
AttendX — start the biometric API server

Usage:
  python start.py

Environment variables (optional):
  HOST  — bind address (default: 0.0.0.0)
  PORT  — port number  (default: 8000)
"""

import os
import uvicorn  # type: ignore

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"\n🚀  AttendX Biometric API — listening on {host}:{port}\n")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,   # auto-reload on code changes during development
        log_level="info",
    )
