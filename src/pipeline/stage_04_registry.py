import yaml

from src.components.model_registry import ModelRegistry

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        model_type = yaml.safe_load(f)["model_training"]["model_type"]

    checkpoint_path = f"models/{model_type}_best.pth"
    metrics_json_path = "metrics/test_metrics.json"

    registry = ModelRegistry(config_path="config/config.yaml")
    entry = registry.initiate_model_registry(
        checkpoint_path=checkpoint_path,
        metrics_json_path=metrics_json_path,
        model_type=model_type,
    )

    print(entry)
