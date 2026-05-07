from pathlib import Path
import subprocess
from datetime import datetime


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Run pipeline step
# -------------------------------------------------
def run_step(command: list):
    print(f"\nRunning: {' '.join(command)}")

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {' '.join(command)}")


# -------------------------------------------------
# Main retraining pipeline
# -------------------------------------------------
def main():

    print("\nStarting retraining pipeline...")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")

    run_step(["python", "src/features/build_features.py"])

    run_step(["python", "src/models/train.py"])

    run_step(["python", "src/models/evaluate.py"])

    run_step(["python", "src/models/select_model.py"])

    print("\nRetraining pipeline completed successfully.")


if __name__ == "__main__":
    main()