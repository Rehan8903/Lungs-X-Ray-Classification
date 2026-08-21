import json
from pathlib import Path

from src.components.data_preprocessing import ImagePreprocessing
from src.components.model_trainer import ModelTrainer

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
TEST_DIR = "data/raw/chest_xray/test"  # adjust if your resolved raw dir differs

if __name__ == "__main__":
    preprocessing = ImagePreprocessing(config_path="config/config.yaml")
    prep_result = preprocessing.initiate_image_preprocessing(
        train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR,
    )

    trainer = ModelTrainer(config_path="config/config.yaml")
    train_result = trainer.train(
        train_loader=prep_result["dataloaders"]["train"],
        val_loader=prep_result["dataloaders"]["val"],
    )

    # save training history alongside the checkpoint so DVC tracks it as a metric/plot
    history_path = Path("metrics/train_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(train_result["history"], f, indent=2)

    print("Checkpoint:", train_result["checkpoint_path"])
    print("History saved to:", history_path)
