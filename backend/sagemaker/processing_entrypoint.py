"""SageMaker Processing entry point for reproducible offline evaluation."""
import os
from pathlib import Path

# SageMaker mounts the supplied S3 dataset and output directories at these paths.
INPUT_DATASET = Path("/opt/ml/processing/input/synthetic_conversations.json")
if INPUT_DATASET.exists():
    os.environ["CLEARCALL_DATA_PATH"] = str(INPUT_DATASET)
os.environ.setdefault("EVALUATION_OUTPUT_DIR", "/opt/ml/processing/output")

from app.evaluation import run_evaluation


if __name__ == "__main__":
    result = run_evaluation(log_to_mlflow=True)
    print(result["metrics"])
