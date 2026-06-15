# =============================================================================
# Data Transformation Component
# =============================================================================
# This module is responsible for:
#
# 1. Reading train and test datasets
# 2. Removing duplicate rows
# 3. Separating features and target
# 4. Scaling selected columns
# 5. Saving the preprocessing pipeline
#
# =============================================================================

# Import pandas
import pandas as pd

# Import joblib
import joblib

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Import ColumnTransformer
from sklearn.compose import ColumnTransformer

# Import configuration loader
from src.utils.config import load_schema


# =============================================================================
# Data Transformation Class
# =============================================================================
class DataTransformation:

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self):

        # Load schema configuration
        schema = load_schema()

        # Artifact paths
        self.train_path = schema["artifacts"]["train_data_path"]
        self.test_path = schema["artifacts"]["test_data_path"]
        self.preprocessor_path = schema["artifacts"]["preprocessor_path"]

        # Target column
        self.target_column = schema["target_column"]

        # Columns to scale
        self.scale_columns = schema["scale_columns"]

    # -------------------------------------------------------------------------
    # Main transformation function
    # -------------------------------------------------------------------------
    def initiate_data_transformation(self):

        print("=" * 70)
        print("STARTING DATA TRANSFORMATION")
        print("=" * 70)

        # Read datasets
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        print(f"Original train shape : {train_df.shape}")
        print(f"Original test shape  : {test_df.shape}")

        # Remove duplicates
        train_df = train_df.drop_duplicates()
        test_df = test_df.drop_duplicates()

        print(f"Train shape after duplicate removal : {train_df.shape}")
        print(f"Test shape after duplicate removal  : {test_df.shape}")

        # Separate features and target
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]

        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "scaler",
                    StandardScaler(),
                    self.scale_columns
                )
            ],
            remainder="passthrough"
        )

        # Fit on training data
        X_train = preprocessor.fit_transform(X_train)

        # Transform test data
        X_test = preprocessor.transform(X_test)

        # Save preprocessor
        joblib.dump(
            preprocessor,
            self.preprocessor_path
        )

        print("Preprocessor saved successfully.")

        print("=" * 70)
        print("DATA TRANSFORMATION COMPLETED")
        print("=" * 70)

        return X_train, X_test, y_train, y_test


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    transformer = DataTransformation()

    X_train, X_test, y_train, y_test = (
        transformer.initiate_data_transformation()
    )

    print("\nTransformation completed successfully.")
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)