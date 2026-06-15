# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Import Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Import XGBoost model
from xgboost import XGBClassifier


# Function to return all models
def get_models():

    # Store all models in dictionary
    models = {

        # Logistic Regression baseline model
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        ),

        # Random Forest model
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        # XGBoost model
        "xgboost": XGBClassifier(

            # Number of boosting trees
            n_estimators=100,

            # Maximum depth of each tree
            max_depth=6,

            # Learning rate
            learning_rate=0.1,

            # Because classification problem
            objective="binary:logistic",

            # Random seed
            random_state=42,

            # Faster training
            n_jobs=-1,

            # Evaluation metric
            eval_metric="logloss"
        )
    }

    # Return all models
    return models