# =============================================================================
# Data Validation Component
# =============================================================================
# This module is responsible for:
#
# 1. Reading the training dataset
# 2. Validating the schema
# 3. Checking for null values
# 4. Checking for duplicate rows
# 5. Validating the target column
#
# =============================================================================

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

# Import pandas for dataframe operations
import pandas as pd

# Import schema loader utility
from src.utils.config import load_schema


# =============================================================================
# Data Validation Class
# =============================================================================
class DataValidation:

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self):

        # Path of training dataset
        self.train_path = "artifacts/train.csv"

    # -------------------------------------------------------------------------
    # Main validation function
    # -------------------------------------------------------------------------
    def initiate_data_validation(self):

        # Print header
        print("=" * 70)
        print("STARTING DATA VALIDATION")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Load training dataset
        # ---------------------------------------------------------------------

        df = pd.read_csv(self.train_path)

        # ---------------------------------------------------------------------
        # Load schema configuration
        # ---------------------------------------------------------------------

        schema = load_schema()

        # Expected columns from schema
        expected_columns = schema["columns"]

        # Actual columns from dataset
        actual_columns = list(df.columns)

        # ---------------------------------------------------------------------
        # Validate schema
        # ---------------------------------------------------------------------

        if expected_columns != actual_columns:

            raise Exception(
                "Schema validation failed.\n"
                f"Expected Columns:\n{expected_columns}\n\n"
                f"Actual Columns:\n{actual_columns}"
            )

        print("Schema validation passed.")

        # ---------------------------------------------------------------------
        # Validate null values
        # ---------------------------------------------------------------------

        null_count = df.isnull().sum().sum()

        if null_count > 0:

            raise Exception(
                f"Dataset contains {null_count} null values."
            )

        print("Null value validation passed.")

        # ---------------------------------------------------------------------
        # Validate duplicate rows
        # ---------------------------------------------------------------------

        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:

            print(
                f"Warning: Found {duplicate_count} duplicate rows."
            )

        else:

            print("Duplicate validation passed.")

        # ---------------------------------------------------------------------
        # Validate target column
        # ---------------------------------------------------------------------

        target_column = schema["target_column"]

        unique_values = set(
            df[target_column].unique()
        )

        expected_target_values = {0, 1}

        if unique_values != expected_target_values:

            raise Exception(
                f"Invalid target values detected: {unique_values}"
            )

        print("Target validation passed.")

        # ---------------------------------------------------------------------
        # Validation successful
        # ---------------------------------------------------------------------

        print("=" * 70)
        print("DATA VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return True


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    # Create validator object
    validator = DataValidation()

    # Run validation
    validator.initiate_data_validation()