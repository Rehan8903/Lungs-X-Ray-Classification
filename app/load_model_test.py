"""
Standalone sanity check — run this BEFORE starting the FastAPI server to confirm
the production model loads correctly and (optionally) produces a sensible
prediction on a real X-ray image.

Usage:
    python -m app.load_model_test
    python -m app.load_model_test --image path/to/xray.jpeg
"""
import argparse
import sys

import torch

from src.logger import logger
from src.exception import CustomException
from src.components.model import build_model
from src.components.model_registry import ModelRegistry
from app.preprocessing_utility import load_inference_config, build_eval_transform, preprocess_image_path

CONFIG_PATH = "config/config.yaml"


def load_production_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        registry = ModelRegistry(config_path=CONFIG_PATH)
        entry = registry.get_production_model()

        if entry is None:
            raise RuntimeError(
                "No production model found in the registry. "
                "Run 'dvc repro' first (data_ingestion -> train -> evaluate -> registry)."
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

        logger.info(
            "Loaded production model %s (%s) — %s=%.4f, test_accuracy=%.4f",
            entry["version"], entry["model_type"],
            entry["primary_metric"], entry["primary_metric_value"],
            entry.get("test_accuracy", float("nan")),
        )
        return model, device, entry

    except Exception as e:
        raise CustomException(e, sys)


def run_test_prediction(model, device, image_path: str):
    try:
        config = load_inference_config(CONFIG_PATH)
        transform = build_eval_transform(
            config["img_size"], config["normalize_mean"], config["normalize_std"]
        )
        input_tensor = preprocess_image_path(image_path, transform).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(probs.argmax().item())

        class_names = config["class_names"]
        logger.info("Test image: %s", image_path)
        logger.info("Prediction: %s (confidence=%.4f)", class_names[pred_idx], probs[pred_idx].item())
        for i, cls in enumerate(class_names):
            logger.info("  %s: %.4f", cls, probs[i].item())

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check the production model before serving it.")
    parser.add_argument("--image", type=str, default=None, help="Optional path to a test X-ray image.")
    args = parser.parse_args()

    model, device, entry = load_production_model()
    print(f"\nModel loaded successfully: {entry['version']} ({entry['model_type']}), stage={entry['stage']}\n")

    if args.image:
        run_test_prediction(model, device, args.image)
    else:
        print("No --image passed — skipping test prediction. Model load check passed.")
