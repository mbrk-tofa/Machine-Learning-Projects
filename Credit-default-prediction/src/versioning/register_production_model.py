from pathlib import Path
import json
import shutil

from src.versioning.version_manager import register_model


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Main execution
# -------------------------------------------------
def main():

    root = get_project_root()

    selection_path = root / "artifacts" / "model_selection.json"
    evaluation_path = root / "artifacts" / "evaluation_results.json"

    with open(selection_path, "r") as f:
        selection = json.load(f)

    with open(evaluation_path, "r") as f:
        evaluation = json.load(f)

    best_model = min(
        selection.items(),
        key=lambda x: x[1]["expected_cost"]
    )[0]

    expected_cost = selection[best_model]["expected_cost"]
    roc_auc = evaluation[best_model]["roc_auc"]

    versioned_model_name, version = register_model(
        model_name=best_model,
        expected_cost=expected_cost,
        roc_auc=roc_auc
    )

    source_model = root / "artifacts" / "models" / f"{best_model}.joblib"

    target_model = ( root / "artifacts" / "models" / f"{versioned_model_name}.joblib")

    shutil.copy(source_model, target_model)

    print("\nRegistered production model:")
    print(versioned_model_name)


if __name__ == "__main__":
    main()