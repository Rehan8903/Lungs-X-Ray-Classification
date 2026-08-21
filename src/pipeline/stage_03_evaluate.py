import yaml

from src.components.data_preprocessing import ImagePreprocessing
from src.components.model_evaluation import ModelEvaluation

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
TEST_DIR = "data/raw/chest_xray/test"  # adjust if your resolved raw dir differs

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        model_type = yaml.safe_load(f)["model_training"]["model_type"]

    preprocessing = ImagePreprocessing(config_path="config/config.yaml")
    prep_result = preprocessing.initiate_image_preprocessing(
        train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR,
    )

    checkpoint_path = f"models/{model_type}_best.pth"

    evaluator = ModelEvaluation(config_path="config/config.yaml")
    eval_result = evaluator.initiate_model_evaluation(
        test_loader=prep_result["dataloaders"]["test"],
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )

    print("Metrics:", eval_result["metrics"])
