from pathlib import Path
import subprocess


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# -------------------------------------------------
# Run pipeline step
# -------------------------------------------------
def run_step(command: list):
    print(f"\nRunning: {' '.join(command)}")

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: {' '.join(command)}"
        )


# -------------------------------------------------
# Main execution
# -------------------------------------------------
def main():

    root = get_project_root()

    commands = [
        ["pip install requirements.txt"],
        ["python", str(root / "src" / "data" / "ingest.py")],
        ["python", str(root / "src" / "eda" / "eda.py")],
        ["python", str(root / "src" / "features" / "build_features.py")],
        ["python", str(root / "src" / "models" / "train.py")],
        ["python", str(root / "src" / "models" / "evaluate.py")],
        ["python", str(root / "src" / "models" / "select_model.py")],
        ["python", str(root / "src" / "versioning" / "register_production_model.py")]
    ]

    for command in commands:
        run_step(command)

    print("\nFull ML pipeline completed successfully.")


if __name__ == "__main__":
    main()