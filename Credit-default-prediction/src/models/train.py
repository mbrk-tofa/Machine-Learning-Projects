from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#Path resolution
def get_project_root()-> Path:
    return Path(__file__).resolve().parents[2]

#Loading data
def load_processed_data(root: Path) -> pd.DataFrame:
    data_path = root / "data" / "processed" / "credit_default_v1.parquet"
    return pd.read_parquet(data_path)

def load_feature_pipeline(root: Path):
    pip_path = root / "artifacts" / "feature_pipeline.joblib"
    return joblib.load(pip_path)

# Train / test split
# def split_features(df: pd.DataFrame):
#     train_df = df[df["split"]=="train"].drop(columns=["split"])
#     test_df = df[df["split"]=="test"].drop(columns=["split"])

#     X_train = train_df.drop(columns=["default"])
#     y_train = train_df["default"]

#     X_test = test_df.drop(columns=["default"])
#     y_test = test_df["default"]

#     return X_train, X_test, y_train, y_test

def split_features(df: pd.DataFrame):
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]

    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    return X_train, y_train, X_test, y_test

#Model definition
def build_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            random_state=24
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=42
        ),
    }

#Taining loop
def train_models(X_train, y_train, preprocessor, models, root: Path):
    model_dir = root / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        pipeline = Pipeline(
            steps = [
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )
        
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, model_dir / f"{name}.joblib")
        print(f"Saved model: {name}")

# Main execution
def main():
    root = get_project_root()

    df = load_processed_data(root)
    preprocessor = load_feature_pipeline(root)

    X_train, y_train, _, _ = split_features(df)

    models = build_models()

    train_models(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        models=models,
        root=root,
    )

    print("Model training completed sucessfully. ")

if __name__ == "__main__":
    main()