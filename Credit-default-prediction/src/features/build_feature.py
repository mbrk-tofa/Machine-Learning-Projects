""" This script is meant for:
 - Loading the versioned processed dataset
 - Separating train/test splits
 - Defining feature groups
 - Building a preprocessing pipeline
 - Fitting the pipeline only on training data
 - Persisting the fitted pipeline as an artifact
"""


import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#Path resolution
#PROJECT_ROOT = Path(__file__).resolve().parents[2]
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

#Data Loading
def load_processed_data(root: Path) -> pd.DataFrame:
    data_path = root / 'data' / 'processed' / 'credit_default_v1.parquet'
    return pd.read_parquet(data_path)

#Grouping features:
def get_feature_group():
    numerical_features = [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]
    ordinal_features = [
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]
    categorical_features = [
        "SEX", "EDUCATION", "MARRIAGE"
    ]
    return numerical_features, ordinal_features, categorical_features

#Build preprocessor
def build_preprocessor(
        numerical_features,
        ordinal_features,
        categorical_features
)-> ColumnTransformer:
    
    numeric_tranformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tranformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("ord", "passthrough", ordinal_features)
        ]
    )

    return preprocessor

# Main Execution
def main():
    root = get_project_root()

    df = load_processed_data(root)

    train_df = df[df["split"] == "train"].drop(columns=["split"])

    X_train = train_df.drop(columns=["default"])

    numeric, ordinal, categorical = get_feature_group()

    preprocessor = build_preprocessor(
        numerical_features=numeric,
        ordinal_features=ordinal,
        categorical_features=categorical,
    )

    preprocessor.fit(X_train)

    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        preprocessor,
        artifact_dir / "feature_pipeline.joblib"
    )

    print("Feature pipeline built and saved successfully.")


if __name__ == "__main__":
    main()