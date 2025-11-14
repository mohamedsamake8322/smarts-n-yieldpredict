import logging
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "../logs/fertilizer.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_prediction(inputs: dict, prediction: str):
    logging.info(f"Inputs: {inputs} → Prediction: {prediction}")
