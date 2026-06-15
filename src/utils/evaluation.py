
# =============================================================================
# Model Evaluation Utility
# =============================================================================
# This module evaluates a trained classification model and returns
# commonly used performance metrics.
# =============================================================================

# Import evaluation metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate_model(model, X_test, y_test):

    # -------------------------------------------------------------------------
    # Generate predictions
    # -------------------------------------------------------------------------

    # Predict class labels
    y_pred = model.predict(X_test)

    # Predict class probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------------------

    # Classification report
    report = classification_report(
        y_test,
        y_pred
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Precision
    precision = precision_score(
        y_test,
        y_pred
    )

    # Recall
    recall = recall_score(
        y_test,
        y_pred
    )

    # F1-score
    f1 = f1_score(
        y_test,
        y_pred
    )

    # ROC-AUC
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # -------------------------------------------------------------------------
    # Return all evaluation metrics
    # -------------------------------------------------------------------------

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

