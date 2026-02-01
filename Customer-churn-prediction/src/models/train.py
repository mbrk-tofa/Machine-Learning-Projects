from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models"
MODEL_PATH.mkdir(exist_ok=True)


def load_artifacts():
    """Load preprocessing pipeline and training data."""
    preprocessor = joblib.load(ARTIFACTS_PATH / "preprocessor.pkl")
    X_train = joblib.load(ARTIFACTS_PATH / "X_train.pkl")
    y_train = joblib.load(ARTIFACTS_PATH / "y_train.pkl")
    return preprocessor, X_train, y_train


def build_model():
    """Create logistic regression model with L2 regularization."""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    return model


def build_training_pipeline(preprocessor, model):
    """Combine preprocessing and model into one pipeline."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def save_model(pipeline):
    """Persist trained model."""
    joblib.dump(pipeline, MODEL_PATH / "logistic_model.pkl")


def main():
    preprocessor, X_train, y_train = load_artifacts()
    model = build_model()
    pipeline = build_training_pipeline(preprocessor, model)

    pipeline.fit(X_train, y_train)

    save_model(pipeline)


if __name__ == "__main__":
    main()
