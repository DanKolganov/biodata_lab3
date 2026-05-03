import os
from pathlib import Path

class Config:
    # Base paths - detect if running in Docker or locally
    if os.getenv("DOCKER_ENV"):
        BASE_DIR = Path("/app")
    else:
        BASE_DIR = Path(__file__).parent.parent
        
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MLFLOW_DIR = BASE_DIR / "mlflow"
    
    # Lakehouse paths
    BRONZE_PATH = DATA_DIR / "bronze"
    SILVER_PATH = DATA_DIR / "silver"
    GOLD_PATH = DATA_DIR / "gold"
    
    # Source data
    SOURCE_CSV = DATA_DIR / "flight_data_2018_2024.csv"
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Delta Lake settings
    DELTA_MAX_WORKERS = 4
    
    # Processing
    BATCH_SIZE = 10000
    
    # ML
    DELAY_THRESHOLD_MINUTES = 15  # For classification
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
