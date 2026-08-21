import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from src.logger import logger
from src.exception import CustomException
from src.components.model import build_model


@dataclass
class ModelEvaluationConfig:
    metrics_dir: Path
    confusion_matrix_filename: str
    metrics_filename: str
    class_names: list


class ModelEvaluation:
    """
    Handles:
      1. Loading a trained model from a checkpoint
      2. Running inference on the test set
      3. Computing evaluation metrics
      4. Saving confusion matrix and metrics JSON
    """

    def __init__(self, config_path: str = "config/config.yaml", device: torch.device = None):
        try:
            self.config = self._load_config(config_path)
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.config.metrics_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "ModelEvaluation initialized. Device: %s | Config: %s",
                self.device,
                self.config,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> ModelEvaluationConfig:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        eval_cfg = raw["model_evaluation"]

        class_names = raw.get(
            "data_ingestion", {}
        ).get(
            "class_names",
            ["NORMAL", "PNEUMONIA"]
        )

        return ModelEvaluationConfig(
            metrics_dir=Path(eval_cfg["metrics_dir"]),
            confusion_matrix_filename=eval_cfg["confusion_matrix_filename"],
            metrics_filename=eval_cfg["metrics_filename"],
            class_names=class_names,
        )

    def load_model_from_checkpoint(self, checkpoint_path: str) -> nn.Module:
        """Load an already-trained model from a checkpoint."""

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

            model_type = checkpoint["model_type"]
            num_classes = checkpoint["num_classes"]

            model = build_model(
                model_type,
                num_classes=num_classes,
                freeze_backbone=False
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            model = model.to(self.device)
            model.eval()

            logger.info(
                "Loaded '%s' checkpoint from %s",
                model_type,
                checkpoint_path
            )

            return model

        except Exception as e:
            raise CustomException(e, sys)

    @torch.no_grad()
    def _run_inference(self, model: nn.Module, test_loader) -> dict:

        model.eval()

        criterion = nn.CrossEntropyLoss()

        running_loss = 0.0
        total = 0

        y_true = []
        y_pred = []
        y_prob = []

        for images, labels in test_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            preds = outputs.argmax(dim=1)

            y_true.extend(
                labels.cpu().numpy().tolist()
            )

            y_pred.extend(
                preds.cpu().numpy().tolist()
            )

            y_prob.extend(
                probs.cpu().numpy().tolist()
            )

        return {
            "test_loss": running_loss / total,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def _compute_metrics(self, inference_result: dict) -> dict:

        y_true = inference_result["y_true"]
        y_pred = inference_result["y_pred"]
        y_prob = inference_result["y_prob"]

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.config.class_names,
            output_dict=True
        )

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

        accuracy = report["accuracy"]

        metrics = {
            "test_loss": inference_result["test_loss"],
            "test_accuracy": accuracy,
            "roc_auc": roc_auc,

            "per_class": {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1_score": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                }
                for cls in self.config.class_names
            },

            "macro_avg": {
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1_score": report["macro avg"]["f1-score"],
            },

            "weighted_avg": {
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"],
            },
        }

        logger.info(
            "Test metrics — loss=%.4f accuracy=%.4f roc_auc=%.4f",
            metrics["test_loss"],
            metrics["test_accuracy"],
            metrics["roc_auc"],
        )

        for cls in self.config.class_names:

            logger.info(
                "  %s — precision=%.4f recall=%.4f f1=%.4f",
                cls,
                metrics["per_class"][cls]["precision"],
                metrics["per_class"][cls]["recall"],
                metrics["per_class"][cls]["f1_score"],
            )

        return metrics

    def _save_confusion_matrix(self, inference_result: dict) -> Path:

        cm = confusion_matrix(
            inference_result["y_true"],
            inference_result["y_pred"]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.config.class_names
        )

        disp.plot(cmap="Blues")

        plt.title("Test Set — Confusion Matrix")

        save_path = (
            self.config.metrics_dir
            / self.config.confusion_matrix_filename
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "Confusion matrix saved to %s",
            save_path
        )

        return save_path

    def _save_metrics_json(
        self,
        metrics: dict,
        model_type: str
    ) -> Path:

        save_path = (
            self.config.metrics_dir
            / self.config.metrics_filename
        )

        payload = {
            "model_type": model_type,
            **metrics
        }

        with open(save_path, "w") as f:
            json.dump(
                payload,
                f,
                indent=2
            )

        logger.info(
            "Metrics JSON saved to %s",
            save_path
        )

        return save_path

    def initiate_model_evaluation(
        self,
        test_loader,
        model: nn.Module = None,
        checkpoint_path: str = None,
        model_type: str = "model"
    ) -> dict:

        try:

            logger.info(
                "===== Starting Model Evaluation ====="
            )

            # Load existing trained model
            if model is None:

                if checkpoint_path is None:
                    raise ValueError(
                        "Provide either a `model` or a `checkpoint_path`."
                    )

                model = self.load_model_from_checkpoint(
                    checkpoint_path
                )

            inference_result = self._run_inference(
                model,
                test_loader
            )

            metrics = self._compute_metrics(
                inference_result
            )

            cm_path = self._save_confusion_matrix(
                inference_result
            )

            metrics_path = self._save_metrics_json(
                metrics,
                model_type
            )

            logger.info(
                "===== Model Evaluation Completed Successfully ====="
            )

            return {
                "metrics": metrics,
                "confusion_matrix_path": str(cm_path),
                "metrics_json_path": str(metrics_path),
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_preprocessing import ImagePreprocessing

    # ---------------------------------------------------------
    # 1. DATA INGESTION
    # ---------------------------------------------------------

    ingestion = DataIngestion(
        config_path="config/config.yaml"
    )

    paths = ingestion.initiate_data_ingestion(
        force_resplit=False
    )

    # ---------------------------------------------------------
    # 2. PREPROCESSING
    # ---------------------------------------------------------

    preprocessing = ImagePreprocessing(
        config_path="config/config.yaml"
    )

    prep_result = preprocessing.initiate_image_preprocessing(
        train_dir=paths["train_dir"],
        val_dir=paths["val_dir"],
        test_dir=paths["test_dir"],
    )

    # ---------------------------------------------------------
    # 3. EVALUATION
    # ---------------------------------------------------------

    evaluator = ModelEvaluation(
        config_path="config/config.yaml"
    )

    # CHANGE THIS TO YOUR ACTUAL CHECKPOINT
    checkpoint_path = "models/resnet18_best.pth"

    eval_result = evaluator.initiate_model_evaluation(
        test_loader=prep_result["dataloaders"]["test"],
        checkpoint_path=checkpoint_path,
    )

    # ---------------------------------------------------------
    # 4. PRINT RESULTS
    # ---------------------------------------------------------

    print(
        json.dumps(
            eval_result["metrics"],
            indent=2
        )
    )