# `src/components/model_trainer.py`


# =============================================================================
# Model Trainer Component
# =============================================================================
# This module is responsible for:
#
# 1. Loading transformed train and test datasets
# 2. Training multiple machine learning models
# 3. Evaluating each trained model
# 4. Logging experiments into MLflow
# 5. Selecting the best model based on configuration
# 6. Saving the best model locally
# 7. Saving model metadata
#
# =============================================================================

# -----------------------------------------------------------------------------
# Import standard libraries
# -----------------------------------------------------------------------------

# Used to save metadata in JSON format
import json

# Used for file path handling
import os
import shutil

# -----------------------------------------------------------------------------
# Import third-party libraries
# -----------------------------------------------------------------------------

# Used to save trained model
import joblib

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Metrics used for evaluation
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# Import project modules
# -----------------------------------------------------------------------------

# Data transformation component
from src.components.data_transformation import DataTransformation

# Model factory
from src.utils.model_factory import get_models

# Evaluation helper
from src.utils.evaluation import evaluate_model

# Configuration loader
from src.utils.config import load_schema


# =============================================================================
# Model Trainer Class
# =============================================================================
class ModelTrainer:

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self):

        # Artifact directory
        self.artifact_dir = "artifacts"

        # Path to save best model
        self.model_path = os.path.join(
            self.artifact_dir,
            "best_model.pkl"
        )

        # Path to save metadata
        self.metadata_path = os.path.join(
            self.artifact_dir,
            "model_metadata.json"
        )

    # -------------------------------------------------------------------------
    # Main Training Function
    # -------------------------------------------------------------------------
    def initiate_model_training(self):

        # Print header
        print("=" * 70)
        print("STARTING MODEL TRAINING")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Load configuration
        # ---------------------------------------------------------------------

        # Read schema.yaml
        schema = load_schema()

        # Read MLflow experiment name
        experiment_name = schema["mlflow_experiment_name"]

        # Read model selection metric
        selection_metric = schema["model_selection_metric"]

        # Configure MLflow experiment
        mlflow.set_experiment(experiment_name)

        # ---------------------------------------------------------------------
        # Load transformed datasets
        # ---------------------------------------------------------------------

        transformer = DataTransformation()

        X_train, X_test, y_train, y_test = (
            transformer.initiate_data_transformation()
        )

        # ---------------------------------------------------------------------
        # Load all available models
        # ---------------------------------------------------------------------

        models = get_models()

        # ---------------------------------------------------------------------
        # Variables to track best model
        # ---------------------------------------------------------------------

        best_model = None

        best_model_name = None

        best_score = float("-inf")

        # ---------------------------------------------------------------------
        # Train every model
        # ---------------------------------------------------------------------

        for model_name, model in models.items():

            print("\n" + "-" * 70)
            print(f"Training Model : {model_name}")
            print("-" * 70)

            try:

                # -------------------------------------------------------------
                # Start MLflow run
                # -------------------------------------------------------------
                with mlflow.start_run(run_name=model_name):

                    # ---------------------------------------------------------
                    # Train model
                    # ---------------------------------------------------------
                    model.fit(
                        X_train,
                        y_train
                    )

                    # ---------------------------------------------------------
                    # Generate predictions
                    # ---------------------------------------------------------
                    y_pred = model.predict(
                        X_test
                    )

                    # ---------------------------------------------------------
                    # Calculate evaluation metrics
                    # ---------------------------------------------------------
                    precision = precision_score(
                        y_test,
                        y_pred
                    )

                    recall = recall_score(
                        y_test,
                        y_pred
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred
                    )

                    metrics = evaluate_model(
                        model=model,
                        X_test=X_test,
                        y_test=y_test
                    )

                    report = metrics["classification_report"]
                    cm = metrics["confusion_matrix"]
                    precision = metrics["precision"]
                    recall = metrics["recall"]
                    f1 = metrics["f1_score"]
                    roc_auc = metrics["roc_auc"]

                    # ---------------------------------------------------------
                    # MLflow Logging
                    # ---------------------------------------------------------

                    # Log model name
                    mlflow.log_param(
                        "model_name",
                        model_name
                    )

                    # Log all hyperparameters
                    mlflow.log_params(
                        model.get_params()
                    )

                    # Log evaluation metrics
                    mlflow.log_metric(
                        "precision",
                        precision
                    )

                    mlflow.log_metric(
                        "recall",
                        recall
                    )

                    mlflow.log_metric(
                        "f1_score",
                        f1
                    )

                    mlflow.log_metric(
                        "roc_auc",
                        roc_auc
                    )

                    # Save model inside MLflow
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name=model_name
                    )

                    # ---------------------------------------------------------
                    # Print evaluation
                    # ---------------------------------------------------------

                    print("\nClassification Report:")
                    print(report)

                    print("\nConfusion Matrix:")
                    print(cm)

                    print(f"\nPrecision : {precision:.4f}")
                    print(f"Recall    : {recall:.4f}")
                    print(f"F1 Score  : {f1:.4f}")
                    print(f"ROC-AUC   : {roc_auc:.4f}")

                    # ---------------------------------------------------------
                    # Select score based on configuration
                    # ---------------------------------------------------------

                    if selection_metric == "roc_auc":

                        current_score = roc_auc

                    elif selection_metric == "precision":

                        current_score = precision

                    elif selection_metric == "recall":

                        current_score = recall

                    elif selection_metric == "f1_score":

                        current_score = f1

                    else:

                        raise ValueError(
                            f"Unsupported model_selection_metric: "
                            f"{selection_metric}"
                        )

                    # ---------------------------------------------------------
                    # Update best model
                    # ---------------------------------------------------------

                    if current_score > best_score:

                        best_score = current_score

                        best_model = model

                        best_model_name = model_name

            except Exception as error:

                print(
                    f"\nError while training "
                    f"{model_name}: {error}"
                )

                # Continue with remaining models
                continue

        # ---------------------------------------------------------------------
        # Save best model
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------

        if best_model is None:
            raise Exception("No model was successfully trained.")

        # Create model artifact directory
        model_dir = os.path.join("artifacts", "model")
        os.makedirs(model_dir, exist_ok=True)

        # Create metadata FIRST
        metadata = {
            "best_model_name": best_model_name,
            "selection_metric": selection_metric,
            "best_score": round(best_score, 6)
        }

        # Save best model
        joblib.dump(
            best_model,
            os.path.join(model_dir, "best_model.pkl")
        )

        # Copy preprocessor
        shutil.copy(
            "artifacts/preprocessor.pkl",
            os.path.join(model_dir, "preprocessor.pkl")
        )

        # Save metadata inside model folder
        with open(
            os.path.join(model_dir, "model_metadata.json"),
            "w"
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4
            )

        # Optional: also save metadata at original location
        with open(
            self.metadata_path,
            "w"
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4
            )

        print("\n" + "=" * 70)
        print("BEST MODEL SUMMARY")
        print("=" * 70)
        print(f"Best Model Name : {best_model_name}")
        print(f"Selection Metric: {selection_metric}")
        print(f"Best Score      : {best_score:.6f}")
        print(f"Model Directory : {model_dir}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    # Create trainer object
    trainer = ModelTrainer()

    # Run training
    trainer.initiate_model_training()

