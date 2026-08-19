import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# also stream to console so logs show up during interactive runs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)
