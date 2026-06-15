
# =============================================================================
# Data Ingestion Component
# =============================================================================
# This module is responsible for:
#
# 1. Reading the raw dataset
# 2. Splitting the dataset into train and test sets
# 3. Preserving class distribution using stratification
# 4. Saving train and test datasets
#
# =============================================================================

# -----------------------------------------------------------------------------
# Import required libraries
# -----------------------------------------------------------------------------

# Import pandas for dataframe operations
import pandas as pd

# Import train_test_split
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

        # Load configuration
        schema = load_schema()

        # Artifact paths
        self.raw_data_path = "artifacts/creditcard.csv"

        self.train_data_path = schema["artifacts"]["train_data_path"]

        self.test_data_path = schema["artifacts"]["test_data_path"]

        # Training configuration
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

        # Read raw dataset
        df = pd.read_csv(self.raw_data_path)

        print(f"Dataset Shape : {df.shape}")

        # Display original class distribution
        print("\nOriginal Class Distribution:")
        print(
            df[self.target_column]
            .value_counts(normalize=True)
        )

        # Split dataset
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

        # Display train distribution
        print("\nTrain Class Distribution:")
        print(
            train_df[self.target_column]
            .value_counts(normalize=True)
        )

        # Display test distribution
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

    # Create ingestion object
    ingestion = DataIngestion()

    # Execute ingestion
    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    # Print saved locations
    print(f"\nTrain File : {train_path}")
    print(f"Test File  : {test_path}")

