import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "processed"


def load_data(path: Path) -> pd.DataFrame:
    """Load cleaned dataset."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str):
    """Separate features and target."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def save_artifacts(preprocessor, X_train, X_test, y_train, y_test):
    """Persist preprocessing objects and data splits."""
    joblib.dump(preprocessor, ARTIFACTS_PATH / "preprocessor.pkl")
    joblib.dump(X_train, ARTIFACTS_PATH / "X_train.pkl")
    joblib.dump(X_test, ARTIFACTS_PATH / "X_test.pkl")
    joblib.dump(y_train, ARTIFACTS_PATH / "y_train.pkl")
    joblib.dump(y_test, ARTIFACTS_PATH / "y_test.pkl")


def main():
    df = load_data(DATA_PATH)

    X, y = split_features_target(df, target="churn")
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X)
    preprocessor.fit(X_train)

    save_artifacts(preprocessor, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
