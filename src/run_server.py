"""
run_server.py - Launches the FastAPI backend server with Uvicorn.
"""
import os
import sys
import uvicorn
from dotenv import load_dotenv

# Ensure src and root directories are in sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

# Load .env from root or current dir
load_dotenv(os.path.join(_root_dir, ".env"))
load_dotenv(".env")

HOST    = os.getenv("HOST",    "0.0.0.0")
PORT    = int(os.getenv("PORT",    "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))

if __name__ == "__main__":
    print(f"Starting Adaptive Risk Engine Server on http://{HOST}:{PORT}...")
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
    )
