"""
run_server.py - Launches the FastAPI backend server with Uvicorn.
"""
import os
import uvicorn
from dotenv import load_dotenv

# Load .env before reading any config
load_dotenv()

HOST    = os.getenv("HOST",    "0.0.0.0")
PORT    = int(os.getenv("PORT",    "8000"))
WORKERS = int(os.getenv("WORKERS", "4"))

if __name__ == "__main__":
    print(f"Starting Adaptive Risk Engine Server on http://{HOST}:{PORT}...")
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
    )
