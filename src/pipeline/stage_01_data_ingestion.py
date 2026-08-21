from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion(config_path="config/config.yaml")
    paths = ingestion.initiate_data_ingestion(force_resplit=False)
    print(paths)
