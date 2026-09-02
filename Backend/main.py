from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os

from yolo_analyzer import analyze_video

app = FastAPI(title="AI Football Analytics API")


@app.get("/")
def home():
    return {"message": "AI Football Analytics API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    input_path = os.path.join("uploads", video.filename)
    output_path = os.path.join("results", "analyzed_" + video.filename)

    # Save uploaded video
    with open(input_path, "wb") as buffer:
        while chunk := await video.read(1024 * 1024):
            buffer.write(chunk)

    # Run YOLO
    analyze_video(input_path, output_path)

    return {
        "message": "Video analyzed successfully!",
        "filename": video.filename,
        "result": output_path
    }


@app.get("/results/{filename}")
def get_result(filename: str):
    path = os.path.join("results", filename)

    if not os.path.exists(path):
        return {"error": "Result not found"}

    return FileResponse(path, media_type="video/mp4")