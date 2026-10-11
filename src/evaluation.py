from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import numpy as np


def find_best_threshold(y_true, anomaly_scores):
    """
    Find the threshold that gives the highest F1-score.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        anomaly_scores
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]

    best_precision = precision[best_index]
    best_recall = recall[best_index]
    best_f1 = f1_scores[best_index]

    return (
        best_threshold,
        best_precision,
        best_recall,
        best_f1
    )


def calculate_pr_auc(y_true, anomaly_scores):
    """
    Calculate Precision-Recall AUC (Average Precision).
    """

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores
    )

    return pr_auc


def evaluate_model(y_true, predictions):
    """
    Generate confusion matrix and classification report.
    """

    cm = confusion_matrix(
        y_true,
        predictions
    )

    report = classification_report(
        y_true,
        predictions
    )

    return cm, report


def calculate_confusion_metrics(y_true, predictions):
    """
    Extract TP, TN, FP and FN from the confusion matrix.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return tn, fp, fn, tp