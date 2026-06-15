
# =============================================================================
# Data Ingestion Component
# =============================================================================

# Import required libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import schema loader
from src.utils.config import load_schema


# =============================================================================
# Data Ingestion Class
# =============================================================================
class DataIngestion:

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self):

        # Load schema configuration
        schema = load_schema()

        # ---------------------------------------------------------------------
        # Raw dataset path
        #
        # Local Development:
        #     artifacts/creditcard.csv
        #
        # Azure ML:
        #     RAW_DATA_PATH environment variable supplied by the job
        # ---------------------------------------------------------------------
        self.raw_data_path = os.getenv(
            "RAW_DATA_PATH",
            "artifacts/creditcard.csv"
        )

        # Artifact output paths
        self.train_data_path = schema["artifacts"]["train_data_path"]
        self.test_data_path = schema["artifacts"]["test_data_path"]

        # Train/test split configuration
        self.test_size = schema["test_size"]
        self.random_state = schema["random_state"]

        # Target column
        self.target_column = schema["target_column"]

    # -------------------------------------------------------------------------
    # Main ingestion function
    # -------------------------------------------------------------------------
    def initiate_data_ingestion(self):

        print("=" * 70)
        print("STARTING DATA INGESTION")
        print("=" * 70)

        print(f"Reading dataset from: {self.raw_data_path}")

        # Check dataset exists
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.raw_data_path}"
            )

        # Read dataset
        df = pd.read_csv(self.raw_data_path)

        print(f"Dataset Shape: {df.shape}")

        # Print original class distribution
        print("\nOriginal Class Distribution:")
        print(
            df[self.target_column]
            .value_counts(normalize=True)
        )

        # Train-test split with stratification
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column]
        )

        # Save train dataset
        train_df.to_csv(
            self.train_data_path,
            index=False
        )

        # Save test dataset
        test_df.to_csv(
            self.test_data_path,
            index=False
        )

        # Display distributions
        print("\nTrain Class Distribution:")
        print(
            train_df[self.target_column]
            .value_counts(normalize=True)
        )

        print("\nTest Class Distribution:")
        print(
            test_df[self.target_column]
            .value_counts(normalize=True)
        )

        print("\nTrain and Test datasets saved successfully.")

        return (
            self.train_data_path,
            self.test_data_path
        )


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    print(f"\nTrain File: {train_path}")
    print(f"Test File : {test_path}")

