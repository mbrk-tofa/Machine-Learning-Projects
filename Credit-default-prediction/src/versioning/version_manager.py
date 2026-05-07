from pathlib import Path
import json
from datetime import datetime


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Registry path
# -------------------------------------------------
def get_registry_path(root: Path):
    registry_dir = root / "artifacts" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    return registry_dir / "model_registry.json"


# -------------------------------------------------
# Load registry
# -------------------------------------------------
def load_registry(path: Path):

    if not path.exists():
        return {
            "latest_version": 0,
            "production_model": None,
            "models": []
        }

    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Save registry
# -------------------------------------------------
def save_registry(path: Path, registry: dict):

    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


# -------------------------------------------------
# Register model
# -------------------------------------------------
def register_model(
    model_name: str,
    expected_cost: float,
    roc_auc: float
):

    root = get_project_root()

    registry_path = get_registry_path(root)

    registry = load_registry(registry_path)

    new_version = registry["latest_version"] + 1

    versioned_model_name = f"{model_name}_v{new_version}"

    model_entry = {
        "version": new_version,
        "model_name": versioned_model_name,
        "created_at": datetime.utcnow().isoformat(),
        "expected_cost": expected_cost,
        "roc_auc": roc_auc
    }

    registry["latest_version"] = new_version
    registry["production_model"] = versioned_model_name
    registry["models"].append(model_entry)

    save_registry(registry_path, registry)

    return versioned_model_name, new_version