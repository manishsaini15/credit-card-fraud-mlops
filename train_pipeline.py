# Import data ingestion component
from src.components.data_ingestion import DataIngestion

# Import data validation component
from src.components.data_validation import DataValidation

# Import model trainer component
from src.components.model_trainer import ModelTrainer


# Main function to run the complete training pipeline
def run_training_pipeline():

    # Print starting message
    print("=" * 60)
    print("Starting Credit Card Fraud Detection Training Pipeline")
    print("=" * 60)

    # -------------------------------
    # STEP 1 : Data Ingestion
    # -------------------------------
    print("\nStep 1 : Data Ingestion")

    ingestion = DataIngestion()

    # Download / split / save train-test data
    ingestion.initiate_data_ingestion()

    # -------------------------------
    # STEP 2 : Data Validation
    # -------------------------------
    print("\nStep 2 : Data Validation")

    validator = DataValidation()

    # Validate schema, nulls, target, etc.
    validator.initiate_data_validation()

    # -------------------------------
    # STEP 3 : Model Training
    # -------------------------------
    print("\nStep 3 : Model Training")

    trainer = ModelTrainer()

    # Train all models and save the best one
    trainer.initiate_model_training()

    # Print completion message
    print("\nTraining Pipeline Completed Successfully!")


# Entry point
if __name__ == "__main__":
    run_training_pipeline()