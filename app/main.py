from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .predictor import EmotionPredictor

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Facial Expression Recognition API",
    version="1.0.0",
    description="FastAPI + browser webcam facial expression recognition"
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

predictor = EmotionPredictor(
    model_path=BASE_DIR / "models" / "best_sentiment_model.h5"
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor.model is not None,
        "model_path": str(predictor.model_path)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty image.")

        image_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode the uploaded image."
            )

        result = predictor.predict(frame)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )
