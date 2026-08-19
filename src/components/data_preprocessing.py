import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from src.logger import logger
from src.exception import CustomException


@dataclass
class ImagePreprocessingConfig:
    img_size: int
    batch_size: int
    num_workers: int
    normalize_mean: list
    normalize_std: list
    use_weighted_sampler: bool
    rotation_degrees: int
    flip_prob: float
    jitter_brightness: float
    jitter_contrast: float
    class_names: list = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])


class ImagePreprocessing:
    """
    Handles:
      1. Building train/eval transform pipelines (grayscale -> 3-channel, resize,
         augmentation for train, normalization for both)
      2. Wrapping train/val/test directories into ImageFolder datasets
      3. Building DataLoaders, with an optional WeightedRandomSampler on the
         train loader to counter class imbalance (more PNEUMONIA than NORMAL)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = self._load_config(config_path)
            self.train_transforms = self._build_train_transforms()
            self.eval_transforms = self._build_eval_transforms()
            logger.info("ImagePreprocessing initialized with config: %s", self.config)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> ImagePreprocessingConfig:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        prep_cfg = raw["image_preprocessing"]
        aug_cfg = prep_cfg.get("augmentation", {})
        class_names = raw.get("data_ingestion", {}).get("class_names", ["NORMAL", "PNEUMONIA"])

        return ImagePreprocessingConfig(
            img_size=prep_cfg["img_size"],
            batch_size=prep_cfg["batch_size"],
            num_workers=prep_cfg["num_workers"],
            normalize_mean=prep_cfg["normalize_mean"],
            normalize_std=prep_cfg["normalize_std"],
            use_weighted_sampler=prep_cfg.get("use_weighted_sampler", True),
            rotation_degrees=aug_cfg.get("random_rotation_degrees", 10),
            flip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
            jitter_brightness=aug_cfg.get("color_jitter_brightness", 0.1),
            jitter_contrast=aug_cfg.get("color_jitter_contrast", 0.1),
            class_names=class_names,
        )

    def _build_train_transforms(self) -> transforms.Compose:
        cfg = self.config
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.RandomRotation(cfg.rotation_degrees),
            transforms.RandomHorizontalFlip(p=cfg.flip_prob),
            transforms.ColorJitter(brightness=cfg.jitter_brightness, contrast=cfg.jitter_contrast),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
        ])

    def _build_eval_transforms(self) -> transforms.Compose:
        cfg = self.config
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
        ])

    def create_datasets(self, train_dir: str, val_dir: str, test_dir: str) -> dict:
        """Wraps train/val/test directories into torchvision ImageFolder datasets."""
        try:
            train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transforms)
            val_dataset = datasets.ImageFolder(val_dir, transform=self.eval_transforms)
            test_dataset = datasets.ImageFolder(test_dir, transform=self.eval_transforms)

            self._validate_class_order(train_dataset)

            logger.info(
                "Datasets created — train: %d, val: %d, test: %d",
                len(train_dataset), len(val_dataset), len(test_dataset),
            )
            return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        except Exception as e:
            raise CustomException(e, sys)

    def _validate_class_order(self, dataset: datasets.ImageFolder) -> None:
        """
        ImageFolder assigns class indices alphabetically. Confirms this matches
        config.class_names so label indices stay consistent across ingestion,
        training, and inference — a mismatch here silently corrupts metrics.
        """
        if dataset.classes != self.config.class_names:
            raise ValueError(
                f"Class order mismatch: dataset gives {dataset.classes}, "
                f"config expects {self.config.class_names}. "
                f"Update config.yaml's class_names to match, or your labels will be flipped."
            )

    def create_dataloaders(self, datasets_dict: dict) -> dict:
        """Builds DataLoaders from the datasets dict returned by create_datasets()."""
        try:
            cfg = self.config
            train_dataset = datasets_dict["train"]

            if cfg.use_weighted_sampler:
                targets = [label for _, label in train_dataset.samples]
                class_counts = np.bincount(targets)
                class_weights = 1.0 / class_counts
                sample_weights = [class_weights[t] for t in targets]
                sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(sample_weights), replacement=True
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=cfg.batch_size,
                    sampler=sampler, num_workers=cfg.num_workers,
                )
                logger.info(
                    "Train loader built with WeightedRandomSampler. Class counts: %s",
                    dict(zip(train_dataset.classes, class_counts)),
                )
            else:
                train_loader = DataLoader(
                    train_dataset, batch_size=cfg.batch_size,
                    shuffle=True, num_workers=cfg.num_workers,
                )
                logger.info("Train loader built with plain shuffling (no weighted sampler).")

            val_loader = DataLoader(
                datasets_dict["val"], batch_size=cfg.batch_size,
                shuffle=False, num_workers=cfg.num_workers,
            )
            test_loader = DataLoader(
                datasets_dict["test"], batch_size=cfg.batch_size,
                shuffle=False, num_workers=cfg.num_workers,
            )

            logger.info(
                "Dataloaders built — batch_size=%d, num_workers=%d",
                cfg.batch_size, cfg.num_workers,
            )
            return {"train": train_loader, "val": val_loader, "test": test_loader}

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_image_preprocessing(self, train_dir: str, val_dir: str, test_dir: str) -> dict:
        """
        Full preprocessing entrypoint: build datasets -> build dataloaders.
        Returns a dict with both datasets and dataloaders for downstream use
        (dataloaders for training, datasets for e.g. dataset-level inspection).
        """
        try:
            logger.info("===== Starting Image Preprocessing =====")
            datasets_dict = self.create_datasets(train_dir, val_dir, test_dir)
            dataloaders_dict = self.create_dataloaders(datasets_dict)
            logger.info("===== Image Preprocessing Completed Successfully =====")

            return {"datasets": datasets_dict, "dataloaders": dataloaders_dict}

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example: chain directly off data_ingestion.py output
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion(config_path="config/config.yaml")
    paths = ingestion.initiate_data_ingestion(force_resplit=False)

    preprocessing = ImagePreprocessing(config_path="config/config.yaml")
    result = preprocessing.initiate_image_preprocessing(
        train_dir=paths["train_dir"],
        val_dir=paths["val_dir"],
        test_dir=paths["test_dir"],
    )

    print("Train batches:", len(result["dataloaders"]["train"]))
    print("Val batches:", len(result["dataloaders"]["val"]))
    print("Test batches:", len(result["dataloaders"]["test"]))