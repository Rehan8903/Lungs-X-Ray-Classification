import sys

import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.logger import logger
from src.exception import CustomException
from src.components.model import build_model
from src.components.model_registry import ModelRegistry
from app.preprocessing_utility import load_inference_config, build_eval_transform, preprocess_image_bytes

CONFIG_PATH = "config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    _raw_config = yaml.safe_load(f)

CORS_ORIGINS = _raw_config["app"]["cors_origins"]

app = FastAPI(
    title="Pneumonia X-Ray Classifier API",
    description="Serves the current production model from the model registry.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated at startup — holds the loaded model + metadata about which
# registry version is currently being served.
model_state = {"model": None, "device": None, "registry_entry": None, "eval_transform": None, "class_names": None}

# Serves style.css and script.js at /static/style.css and /static/script.js
app.mount("/static", StaticFiles(directory="app/template"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("app/template/index.html")


@app.on_event("startup")
def load_production_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        registry = ModelRegistry(config_path=CONFIG_PATH)
        entry = registry.get_production_model()

        if entry is None:
            raise RuntimeError(
                "No production model found in the registry. "
                "Run the DVC pipeline (data_ingestion -> train -> evaluate -> registry) first."
            )

        checkpoint = torch.load(entry["checkpoint_path"], map_location=device)
        model = build_model(
            checkpoint["model_type"],
            num_classes=checkpoint["num_classes"],
            freeze_backbone=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        inference_config = load_inference_config(CONFIG_PATH)
        eval_transform = build_eval_transform(
            inference_config["img_size"],
            inference_config["normalize_mean"],
            inference_config["normalize_std"],
        )

        model_state["model"] = model
        model_state["device"] = device
        model_state["registry_entry"] = entry
        model_state["eval_transform"] = eval_transform
        model_state["class_names"] = inference_config["class_names"]

        logger.info(
            "Loaded production model %s (%s) from registry. %s=%.4f",
            entry["version"], entry["model_type"],
            entry["primary_metric"], entry["primary_metric_value"],
        )

    except Exception as e:
        # Fail loudly at startup rather than serving an app with no model loaded
        raise CustomException(e, sys)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_state["model"] is not None,
    }


@app.get("/model-info")
def model_info():
    entry = model_state["registry_entry"]
    if entry is None:
        raise HTTPException(status_code=503, detail="No model currently loaded.")
    return {
        "version": entry["version"],
        "model_type": entry["model_type"],
        "stage": entry["stage"],
        "primary_metric": entry["primary_metric"],
        "primary_metric_value": entry["primary_metric_value"],
        "test_accuracy": entry.get("test_accuracy"),
        "registered_at": entry["registered_at"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image_bytes(image_bytes, model_state["eval_transform"])
        input_tensor = input_tensor.to(model_state["device"])
    except CustomException:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    try:
        model = model_state["model"]
        class_names = model_state["class_names"]

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(probs.argmax().item())

        result = {
            "prediction": class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {
                class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
            },
            "model_version": model_state["registry_entry"]["version"],
        }

        logger.info(
            "Prediction served — file=%s prediction=%s confidence=%.4f",
            file.filename, result["prediction"], result["confidence"],
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.app:app", host=_raw_config["app"]["host"], port=_raw_config["app"]["port"], reload=True)