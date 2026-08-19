import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.logger import logger
from src.exception import CustomException
from src.components.model import build_model, unfreeze_all


@dataclass
class ModelTrainerConfig:
    model_type: str
    num_classes: int
    checkpoint_dir: Path
    weight_decay: float
    early_stopping_patience: int
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    random_seed: int
    # resnet-specific
    freeze_backbone_epochs: int
    freeze_backbone_lr: float
    finetune_epochs: int
    finetune_lr: float
    # baseline_cnn-specific
    baseline_num_epochs: int
    baseline_lr: float


class ModelTrainer:
    """
    Handles:
      1. Building the model architecture from config (baseline_cnn or resnet18)
      2. Training with early stopping + LR scheduling
      3. For resnet18: a frozen-backbone phase followed by an optional fine-tune phase
      4. Saving the best checkpoint (by val_loss) with the metadata needed to reload it
    """

    def __init__(self, config_path: str = "config/config.yaml", device: torch.device = None):
        try:
            self.config = self._load_config(config_path)
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(self.config.random_seed)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info("ModelTrainer initialized. Device: %s | Config: %s", self.device, self.config)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_config(self, config_path: str) -> ModelTrainerConfig:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)["model_training"]
        resnet_cfg = raw.get("resnet", {})
        baseline_cfg = raw.get("baseline_cnn", {})

        return ModelTrainerConfig(
            model_type=raw["model_type"],
            num_classes=raw["num_classes"],
            checkpoint_dir=Path(raw["checkpoint_dir"]),
            weight_decay=raw.get("weight_decay", 1e-4),
            early_stopping_patience=raw.get("early_stopping_patience", 3),
            lr_scheduler_patience=raw.get("lr_scheduler_patience", 2),
            lr_scheduler_factor=raw.get("lr_scheduler_factor", 0.5),
            random_seed=raw.get("random_seed", 42),
            freeze_backbone_epochs=resnet_cfg.get("freeze_backbone_epochs", 8),
            freeze_backbone_lr=resnet_cfg.get("freeze_backbone_lr", 1e-3),
            finetune_epochs=resnet_cfg.get("finetune_epochs", 5),
            finetune_lr=resnet_cfg.get("finetune_lr", 1e-5),
            baseline_num_epochs=baseline_cfg.get("num_epochs", 10),
            baseline_lr=baseline_cfg.get("learning_rate", 1e-3),
        )

    def _train_one_epoch(self, model, loader, criterion, optimizer) -> tuple[float, float]:
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self, model, loader, criterion) -> tuple[float, float]:
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    def _run_training_phase(self, model, train_loader, val_loader, num_epochs, lr, phase_name) -> dict:
        """
        Runs one training phase (either the only phase for baseline_cnn, or one of
        the two phases — frozen / fine-tune — for resnet18). Applies early stopping
        and keeps the best-by-val_loss weights loaded into `model` at the end.
        """
        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=self.config.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                patience=self.config.lr_scheduler_patience,
                factor=self.config.lr_scheduler_factor,
            )

            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
            best_val_loss = float("inf")
            best_state = None
            epochs_without_improvement = 0

            for epoch in range(1, num_epochs + 1):
                train_loss, train_acc = self._train_one_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                scheduler.step(val_loss)

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                logger.info(
                    "[%s] Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                    phase_name, epoch, num_epochs, train_loss, train_acc, val_loss, val_acc,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.config.early_stopping_patience:
                        logger.info(
                            "[%s] Early stopping at epoch %d (no val_loss improvement for %d epochs).",
                            phase_name, epoch, self.config.early_stopping_patience,
                        )
                        break

            model.load_state_dict(best_state)
            return {"history": history, "best_val_loss": best_val_loss}

        except Exception as e:
            raise CustomException(e, sys)

    def train(self, train_loader, val_loader) -> dict:
        """
        Builds and trains the model specified by config.model_type.
        For resnet18: frozen-backbone phase, then fine-tune phase (both with
        early stopping independently).
        Returns the trained model plus training history for both phases.
        """
        try:
            cfg = self.config
            logger.info("===== Starting Model Training (%s) =====", cfg.model_type)

            if cfg.model_type == "baseline_cnn":
                model = build_model("baseline_cnn", num_classes=cfg.num_classes).to(self.device)
                result = self._run_training_phase(
                    model, train_loader, val_loader,
                    num_epochs=cfg.baseline_num_epochs, lr=cfg.baseline_lr,
                    phase_name="baseline_cnn",
                )
                all_history = {"baseline_cnn": result["history"]}

            elif cfg.model_type == "resnet18":
                model = build_model("resnet18", num_classes=cfg.num_classes, freeze_backbone=True)
                model = model.to(self.device)

                frozen_result = self._run_training_phase(
                    model, train_loader, val_loader,
                    num_epochs=cfg.freeze_backbone_epochs, lr=cfg.freeze_backbone_lr,
                    phase_name="resnet18_frozen",
                )

                model = unfreeze_all(model)
                finetune_result = self._run_training_phase(
                    model, train_loader, val_loader,
                    num_epochs=cfg.finetune_epochs, lr=cfg.finetune_lr,
                    phase_name="resnet18_finetuned",
                )

                all_history = {
                    "resnet18_frozen": frozen_result["history"],
                    "resnet18_finetuned": finetune_result["history"],
                }

            else:
                raise ValueError(f"Unknown model_type in config: '{cfg.model_type}'")

            checkpoint_path = self._save_checkpoint(model, cfg.model_type)
            logger.info("===== Model Training Completed Successfully =====")

            return {
                "model": model,
                "history": all_history,
                "checkpoint_path": str(checkpoint_path),
                "model_type": cfg.model_type,
            }

        except Exception as e:
            raise CustomException(e, sys)

    def _save_checkpoint(self, model, model_type: str) -> Path:
        checkpoint_path = self.config.checkpoint_dir / f"{model_type}_best.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "num_classes": self.config.num_classes,
        }, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)
        return checkpoint_path


if __name__ == "__main__":
    # Example: chain off ingestion -> preprocessing -> training
    from src.components.data_ingestion import DataIngestion
    from src.components.data_preprocessing import ImagePreprocessing

    ingestion = DataIngestion(config_path="config/config.yaml")
    paths = ingestion.initiate_data_ingestion(force_resplit=False)

    preprocessing = ImagePreprocessing(config_path="config/config.yaml")
    prep_result = preprocessing.initiate_image_preprocessing(
        train_dir=paths["train_dir"], val_dir=paths["val_dir"], test_dir=paths["test_dir"],
    )

    trainer = ModelTrainer(config_path="config/config.yaml")
    train_result = trainer.train(
        train_loader=prep_result["dataloaders"]["train"],
        val_loader=prep_result["dataloaders"]["val"],
    )

    print("Trained model_type:", train_result["model_type"])
    print("Checkpoint saved at:", train_result["checkpoint_path"])