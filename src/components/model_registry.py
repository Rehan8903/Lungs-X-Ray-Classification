import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from src.logger import logger
from src.exception import CustomException


@dataclass
class ModelRegistryConfig:
    registry_dir: Path
    registry_index_filename: str
    primary_metric: str


class ModelRegistry:
    """
    Lightweight file-based model registry.

    It DOES NOT train or evaluate models.

    It takes an already-trained checkpoint and an already-generated
    metrics JSON, then registers them as a version.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = self._load_config(config_path)

            self.config.registry_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            self.index_path = (
                self.config.registry_dir
                / self.config.registry_index_filename
            )

            if not self.index_path.exists():
                self._save_index({"versions": []})

            logger.info(
                "ModelRegistry initialized at %s",
                self.config.registry_dir
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(
        self,
        config_path: str
    ) -> ModelRegistryConfig:

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)["model_registry"]

        return ModelRegistryConfig(
            registry_dir=Path(
                raw["registry_dir"]
            ),
            registry_index_filename=raw[
                "registry_index_filename"
            ],
            primary_metric=raw.get(
                "primary_metric",
                "roc_auc"
            ),
        )

    def _load_index(self) -> dict:

        with open(self.index_path, "r") as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:

        with open(self.index_path, "w") as f:
            json.dump(
                index,
                f,
                indent=2
            )

    def _next_version(self, index: dict) -> str:

        existing = [
            int(v["version"][1:])
            for v in index["versions"]
            if v["version"].startswith("v")
        ]

        next_num = (
            max(existing) + 1
            if existing
            else 1
        )

        return f"v{next_num}"

    def register_model(
        self,
        checkpoint_path: str,
        metrics_json_path: str,
        model_type: str
    ) -> dict:

        try:

            checkpoint_path = Path(checkpoint_path)
            metrics_json_path = Path(metrics_json_path)

            # -------------------------------------------------
            # Validate existing files
            # -------------------------------------------------

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            if not metrics_json_path.exists():
                raise FileNotFoundError(
                    f"Metrics file not found: {metrics_json_path}"
                )

            # -------------------------------------------------
            # Load existing registry
            # -------------------------------------------------

            index = self._load_index()

            version = self._next_version(index)

            version_dir = (
                self.config.registry_dir
                / version
            )

            version_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # -------------------------------------------------
            # Copy EXISTING trained model
            # -------------------------------------------------

            new_checkpoint_path = (
                version_dir / "model.pth"
            )

            shutil.copy2(
                checkpoint_path,
                new_checkpoint_path
            )

            # -------------------------------------------------
            # Copy EXISTING evaluation metrics
            # -------------------------------------------------

            new_metrics_path = (
                version_dir / "metrics.json"
            )

            shutil.copy2(
                metrics_json_path,
                new_metrics_path
            )

            # -------------------------------------------------
            # Read metrics
            # -------------------------------------------------

            with open(metrics_json_path, "r") as f:
                metrics = json.load(f)

            primary_value = metrics.get(
                self.config.primary_metric
            )

            if primary_value is None:
                raise ValueError(
                    f"Metric '{self.config.primary_metric}' "
                    f"not found in {metrics_json_path}. "
                    f"Available keys: {list(metrics.keys())}"
                )

            # -------------------------------------------------
            # Create registry entry
            # -------------------------------------------------

            entry = {
                "version": version,
                "model_type": model_type,

                "checkpoint_path": str(
                    new_checkpoint_path
                ),

                "metrics_path": str(
                    new_metrics_path
                ),

                "primary_metric": (
                    self.config.primary_metric
                ),

                "primary_metric_value": (
                    primary_value
                ),

                "test_accuracy": metrics.get(
                    "test_accuracy"
                ),

                "registered_at": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),

                "stage": "staging",
            }

            index["versions"].append(entry)

            self._save_index(index)

            logger.info(
                "Registered %s (%s) — %s=%.4f",
                version,
                model_type,
                self.config.primary_metric,
                primary_value,
            )

            # -------------------------------------------------
            # Try promotion
            # -------------------------------------------------

            promoted = self._maybe_promote(
                version
            )

            # Reload index because _maybe_promote()
            # modifies it.
            index = self._load_index()

            final_entry = next(
                v for v in index["versions"]
                if v["version"] == version
            )

            logger.info(
                "Model %s registered with stage: %s",
                version,
                final_entry["stage"]
            )

            return final_entry

        except Exception as e:
            raise CustomException(e, sys)

    def _maybe_promote(
        self,
        version: str
    ) -> bool:

        try:

            index = self._load_index()

            current_prod = next(
                (
                    v for v in index["versions"]
                    if v["stage"] == "production"
                ),
                None
            )

            candidate = next(
                (
                    v for v in index["versions"]
                    if v["version"] == version
                ),
                None
            )

            if candidate is None:
                raise ValueError(
                    f"Version {version} not found."
                )

            should_promote = (
                current_prod is None
                or candidate["primary_metric_value"]
                > current_prod["primary_metric_value"]
            )

            if should_promote:

                # Archive old production model
                for v in index["versions"]:

                    if v["stage"] == "production":
                        v["stage"] = "archived"

                        logger.info(
                            "Demoted %s to archived.",
                            v["version"]
                        )

                # Promote candidate
                candidate["stage"] = "production"

                self._save_index(index)

                logger.info(
                    "Promoted %s to production (%s=%.4f).",
                    version,
                    self.config.primary_metric,
                    candidate[
                        "primary_metric_value"
                    ],
                )

                return True

            logger.info(
                "%s not promoted — %s=%.4f does not beat "
                "current production (%.4f).",
                version,
                self.config.primary_metric,
                candidate[
                    "primary_metric_value"
                ],
                current_prod[
                    "primary_metric_value"
                ],
            )

            return False

        except Exception as e:
            raise CustomException(e, sys)

    def get_production_model(self) -> dict:

        try:

            index = self._load_index()

            production = next(
                (
                    v for v in index["versions"]
                    if v["stage"] == "production"
                ),
                None
            )

            if production is None:
                logger.info(
                    "No production model registered yet."
                )

            return production

        except Exception as e:
            raise CustomException(e, sys)

    def list_versions(self) -> list:

        try:
            return self._load_index()["versions"]

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_registry(
        self,
        checkpoint_path: str,
        metrics_json_path: str,
        model_type: str
    ) -> dict:

        try:

            logger.info(
                "===== Starting Model Registry ====="
            )

            entry = self.register_model(
                checkpoint_path=checkpoint_path,
                metrics_json_path=metrics_json_path,
                model_type=model_type
            )

            logger.info(
                "===== Model Registry Completed Successfully ====="
            )

            return entry

        except Exception as e:
            raise CustomException(e, sys)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    registry = ModelRegistry(
        config_path="config/config.yaml"
    )

    # ---------------------------------------------------------
    # USE YOUR ALREADY-TRAINED MODEL
    # ---------------------------------------------------------

    checkpoint_path = (
        "models/resnet18_best.pth"
    )

    # ---------------------------------------------------------
    # USE YOUR ALREADY-GENERATED EVALUATION METRICS
    # ---------------------------------------------------------

    metrics_json_path = (
        "metrics/test_metrics.json"
    )

    # ---------------------------------------------------------
    # REGISTER EXISTING MODEL
    # ---------------------------------------------------------

    registry_entry = (
        registry.initiate_model_registry(
            checkpoint_path=checkpoint_path,
            metrics_json_path=metrics_json_path,
            model_type="resnet18"
        )
    )

    print(
        json.dumps(
            registry_entry,
            indent=2
        )
    )