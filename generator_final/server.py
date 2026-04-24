"""
HTTP API for sketch → real image (wraps img2skch.convert_sketch_bytes).

Run from repo root or this folder:
  pip install -r requirements-server.txt
  python server.py

Default: http://127.0.0.1:5002
"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from img2skch import convert_sketch_bytes

app = FastAPI(title="Sketch to Real Image")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
async def convert_sketch(file: UploadFile = File(...)):
    # Browsers often send application/octet-stream for blob uploads
    ct = (file.content_type or "").lower()
    if ct and not (ct.startswith("image/") or ct == "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail="Expected an image file or octet-stream body",
        )
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        jpeg_bytes = convert_sketch_bytes(raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(content=jpeg_bytes, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "5002"))
    uvicorn.run(app, host="127.0.0.1", port=port)
