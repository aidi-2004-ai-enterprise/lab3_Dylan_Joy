"""
train.py

Trains an XGBoost classifier on the Seaborn penguins dataset.

Features:
- One-hot encoding of categorical features
- Label encoding of target variable
- Train/test split with stratification
- Model training with overfitting prevention parameters
- Model evaluation using F1-score and classification report
- Saving model and preprocessing artifacts
- Logging and error handling
- CLI parameter support
"""

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Tuple

import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning, message=".*use_label_encoder.*")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_data() -> pd.DataFrame:
    logging.info("Loading penguins dataset from seaborn...")
    penguins = sns.load_dataset("penguins")
    if penguins is None or penguins.empty:
        raise ValueError("Failed to load penguins dataset or dataset is empty.")
    logging.info("Dataset loaded with shape {}.".format(penguins.shape))
    return penguins


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    logging.info("Preprocessing data...")

    df = df.dropna()
    logging.info("After dropping missing values: {} rows.".format(df.shape[0]))

    categorical_features = ["sex", "island"]
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    logging.info(
        "Applied one-hot encoding on {}, resulting shape {}.".format(
            categorical_features, df_encoded.shape
        )
    )

    label_encoder = LabelEncoder()
    df_encoded["species_encoded"] = label_encoder.fit_transform(df_encoded["species"])
    n_classes = len(label_encoder.classes_)
    logging.info(
        "Encoded target variable into {} classes: {}.".format(
            n_classes, list(label_encoder.classes_)
        )
    )

    os.makedirs("app/data", exist_ok=True)
    with open("app/data/labels.json", "w") as f:
        json.dump(label_encoder.classes_.tolist(), f)
    logging.info("Saved label classes to app/data/labels.json")

    X = df_encoded.drop(columns=["species", "species_encoded"])
    y = df_encoded["species_encoded"]

    with open("app/data/features.json", "w") as f:
        json.dump(X.columns.tolist(), f)
    logging.info("Saved feature columns to app/data/features.json")

    return X, y, label_encoder


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, max_depth: int, n_estimators: int
) -> XGBClassifier:
    logging.info("Training XGBoost model...")
    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model


def evaluate_model(
    model: XGBClassifier, X: pd.DataFrame, y: pd.Series, dataset_name: str = "dataset"
) -> float:
    logging.info("Evaluating model on {}...".format(dataset_name))
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average="weighted")
    logging.info("F1-score on {}: {:.4f}".format(dataset_name, f1))
    report = classification_report(y, y_pred)
    logging.info("Classification report on {}:\n{}".format(dataset_name, report))
    return f1


def save_model(model: XGBClassifier, path: str = "app/data/model.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    logging.info("Model saved to {}".format(path))


def main(test_size: float, max_depth: int, n_estimators: int) -> None:
    setup_logging()
    logging.info("Starting training pipeline...")

    try:
        df = load_data()
        X, y, label_encoder = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        logging.info(
            "Split data into train ({}) and test ({}).".format(len(X_train), len(X_test))
        )

        model = train_model(X_train, y_train, max_depth, n_estimators)

        evaluate_model(model, X_train, y_train, "train set")
        evaluate_model(model, X_test, y_test, "test set")

        save_model(model)

        logging.info("Training pipeline finished successfully.")

    except Exception as e:
        logging.error("An error occurred: {}".format(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on penguins dataset.")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test data proportion, default 0.2"
    )
    parser.add_argument(
        "--max_depth", type=int, default=3, help="XGBoost max tree depth, default 3"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100, help="Number of trees, default 100"
    )
    args = parser.parse_args()
    main(args.test_size, args.max_depth, args.n_estimators)
