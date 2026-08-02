from typing import Dict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import config
from .inference import predict_image_bytes
from .model import build_model


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]


def create_app(checkpoint_path: str = config.CHECKPOINT_PATH, device: torch.device = config.DEVICE) -> FastAPI:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    app = FastAPI(title="Brain MRI Tumor Classifier API")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(device)}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        try:
            result = predict_image_bytes(model, image_bytes, class_names, device=device)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
        return PredictionResponse(**result)

    return app


try:
    app = create_app()
except FileNotFoundError:
    app = None