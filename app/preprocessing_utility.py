import io
import sys

import yaml
from PIL import Image
from torchvision import transforms

from src.logger import logger
from src.exception import CustomException


def load_inference_config(config_path: str = "config/config.yaml") -> dict:
    """
    Pulls the exact settings inference needs to stay consistent with training:
    class names, image size, and normalization stats.
    """
    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        return {
            "class_names": raw["data_ingestion"]["class_names"],
            "img_size": raw["image_preprocessing"]["img_size"],
            "normalize_mean": raw["image_preprocessing"]["normalize_mean"],
            "normalize_std": raw["image_preprocessing"]["normalize_std"],
        }
    except Exception as e:
        raise CustomException(e, sys)


def build_eval_transform(img_size: int, normalize_mean: list, normalize_std: list) -> transforms.Compose:
    """
    Builds the exact same eval-time transform pipeline used in training
    (grayscale -> 3-channel, resize, normalize). Must never drift from
    src/components/data_preprocessing.py's eval transform, or inference
    results will be wrong in a way that's hard to notice.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])


def preprocess_image_bytes(image_bytes: bytes, transform: transforms.Compose):
    """
    Takes raw uploaded image bytes, returns a model-ready tensor with a
    batch dimension of 1 (shape: [1, 3, img_size, img_size]).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        return tensor
    except Exception as e:
        raise CustomException(e, sys)


def preprocess_image_path(image_path: str, transform: transforms.Compose):
    """Same as preprocess_image_bytes, but reads from a file path — used by load_model_test.py."""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return preprocess_image_bytes(image_bytes, transform)
    except Exception as e:
        raise CustomException(e, sys)
