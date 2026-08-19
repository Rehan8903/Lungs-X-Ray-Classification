import os
import sys
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    kaggle_dataset: str
    raw_data_dir: Path
    extracted_folder_name: str
    processed_data_dir: Path
    class_names: list = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    val_split_ratio: float = 0.15
    random_seed: int = 42


class DataIngestion:
    """
    Handles:
      1. Downloading the Chest X-Ray Pneumonia dataset from Kaggle (if not already present)
      2. Extracting it and resolving the raw folder structure
      3. Re-splitting train/ into train/val (stratified) since the official val/ folder
         only has 16 images — test/ is left untouched
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = self._load_config(config_path)
            logger.info("DataIngestion initialized with config: %s", self.config)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> DataIngestionConfig:
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)["data_ingestion"]
        return DataIngestionConfig(
            kaggle_dataset=raw_cfg["kaggle_dataset"],
            raw_data_dir=Path(raw_cfg["raw_data_dir"]),
            extracted_folder_name=raw_cfg["extracted_folder_name"],
            processed_data_dir=Path(raw_cfg["processed_data_dir"]),
            class_names=raw_cfg.get("class_names", ["NORMAL", "PNEUMONIA"]),
            val_split_ratio=raw_cfg.get("val_split_ratio", 0.15),
            random_seed=raw_cfg.get("random_seed", 42),
        )

    def download_dataset(self) -> Path:
        """
        Downloads and extracts the dataset via the Kaggle API.
        Requires kaggle.json to be set up (~/.kaggle/kaggle.json on Linux/Mac,
        C:\\Users\\<you>\\.kaggle\\kaggle.json on Windows).
        Skips download if the raw data already exists.
        """
        try:
            resolved_dir = self._resolve_extracted_dir()
            if resolved_dir is not None:
                logger.info("Dataset already present at %s — skipping download.", resolved_dir)
                return resolved_dir

            self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
            zip_path = self.config.raw_data_dir / "chest-xray-pneumonia.zip"

            logger.info("Downloading dataset '%s' from Kaggle...", self.config.kaggle_dataset)
            exit_code = os.system(
                f"kaggle datasets download -d {self.config.kaggle_dataset} "
                f"-p {self.config.raw_data_dir}"
            )
            if exit_code != 0:
                raise RuntimeError(
                    "Kaggle CLI download failed. Check that kaggle.json is configured "
                    "correctly and the 'kaggle' package is installed."
                )

            logger.info("Extracting %s ...", zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.config.raw_data_dir)

            zip_path.unlink(missing_ok=True)

            # Kaggle's zip sometimes contains junk/system files — clean them up
            macosx_junk = self.config.raw_data_dir / "__MACOSX"
            if macosx_junk.exists():
                shutil.rmtree(macosx_junk)

            resolved_dir = self._resolve_extracted_dir()
            if resolved_dir is None:
                raise RuntimeError(
                    f"Could not locate '{self.config.extracted_folder_name}' folder "
                    f"after extraction. Check the zip contents manually."
                )

            logger.info("Dataset downloaded and extracted to %s", resolved_dir)
            return resolved_dir

        except Exception as e:
            raise CustomException(e, sys)

    def _resolve_extracted_dir(self) -> Path | None:
        """
        The Kaggle zip for this dataset extracts into a nested
        chest_xray/chest_xray/{train,val,test} structure. This resolves to
        whichever level actually contains train/val/test.
        """
        candidates = [
            self.config.raw_data_dir / self.config.extracted_folder_name,
            self.config.raw_data_dir / self.config.extracted_folder_name / self.config.extracted_folder_name,
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "train").exists():
                return candidate
        return None

    def validate_raw_structure(self, raw_dir: Path) -> None:
        """Sanity-check that train/test + both class folders exist with images in them."""
        try:
            required_splits = ["train", "test"]
            for split in required_splits:
                split_dir = raw_dir / split
                if not split_dir.exists():
                    raise FileNotFoundError(f"Missing expected folder: {split_dir}")
                for cls in self.config.class_names:
                    cls_dir = split_dir / cls
                    if not cls_dir.exists():
                        raise FileNotFoundError(f"Missing expected folder: {cls_dir}")
                    n_images = len(list(cls_dir.glob("*.jpeg"))) + len(list(cls_dir.glob("*.jpg")))
                    if n_images == 0:
                        raise ValueError(f"No images found in {cls_dir}")
                    logger.info("%s/%s: %d images", split, cls, n_images)

            logger.info("Raw dataset structure validated successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def create_train_val_split(self, raw_dir: Path, force: bool = False) -> tuple[Path, Path, Path]:
        """
        Re-splits raw_dir/train into processed_data_dir/train and processed_data_dir/val
        (stratified per class). raw_dir/test is referenced directly, never copied/modified.

        Returns (train_dir, val_dir, test_dir).
        """
        try:
            new_train = self.config.processed_data_dir / "train"
            new_val = self.config.processed_data_dir / "val"
            test_dir = raw_dir / "test"

            if new_train.exists() and not force:
                logger.info("Processed train/val split already exists — skipping. Use force=True to rebuild.")
                return new_train, new_val, test_dir

            if new_train.exists():
                shutil.rmtree(new_train)
            if new_val.exists():
                shutil.rmtree(new_val)

            raw_train = raw_dir / "train"

            for cls in self.config.class_names:
                files = list((raw_train / cls).glob("*.jpeg")) + list((raw_train / cls).glob("*.jpg"))
                if not files:
                    raise ValueError(f"No source images found for class '{cls}' in {raw_train / cls}")

                train_files, val_files = train_test_split(
                    files,
                    test_size=self.config.val_split_ratio,
                    random_state=self.config.random_seed,
                )

                (new_train / cls).mkdir(parents=True, exist_ok=True)
                (new_val / cls).mkdir(parents=True, exist_ok=True)

                for f in train_files:
                    shutil.copy(f, new_train / cls / f.name)
                for f in val_files:
                    shutil.copy(f, new_val / cls / f.name)

                logger.info(
                    "Class '%s': %d train images, %d val images",
                    cls, len(train_files), len(val_files),
                )

            logger.info("Train/val split complete. train=%s val=%s test=%s", new_train, new_val, test_dir)
            return new_train, new_val, test_dir

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self, force_resplit: bool = False) -> dict:
        """
        Full ingestion entrypoint: download (if needed) -> validate -> re-split.
        Returns a dict of resolved paths for downstream pipeline stages.
        """
        try:
            logger.info("===== Starting Data Ingestion =====")
            raw_dir = self.download_dataset()
            self.validate_raw_structure(raw_dir)
            train_dir, val_dir, test_dir = self.create_train_val_split(raw_dir, force=force_resplit)
            logger.info("===== Data Ingestion Completed Successfully =====")

            return {
                "raw_dir": str(raw_dir),
                "train_dir": str(train_dir),
                "val_dir": str(val_dir),
                "test_dir": str(test_dir),
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion(config_path="config/config.yaml")
    paths = ingestion.initiate_data_ingestion(force_resplit=False)
    print(paths)
