import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import logging

logger = logging.getLogger(__name__)


def standard_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    classes = sorted(set(y_true))
    for i, cls in enumerate(classes):
        metrics[f"f1_{cls}"] = per_class_f1[i]
        metrics[f"precision_{cls}"] = per_class_precision[i]
        metrics[f"recall_{cls}"] = per_class_recall[i]

    return metrics


def business_metrics(y_true, y_pred) -> dict:
    """Compute business-oriented metrics for Airbnb stakeholders.

    Business Context:
    - Guests want reliable recommendations (don't over-promise)
    - Hosts want recognition for quality (don't undersell)
    - Platform wants to surface best listings
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Map class names to quality tiers
    # Assuming: 'High Rating' > 'Medium Rating', 'Very High Rating' > 'High Rating'
    quality_order = {"Medium Rating": 0, "High Rating": 1, "Very High Rating": 2}

    # Convert to numeric for comparison
    y_true_numeric = np.array([quality_order.get(y, 0) for y in y_true_arr])
    y_pred_numeric = np.array([quality_order.get(y, 0) for y in y_pred_arr])

    # Over-promise rate: Predicted higher quality than actual
    over_promise = np.sum(y_pred_numeric > y_true_numeric) / len(y_true_arr)

    # Undersell rate: Predicted lower quality than actual
    undersell = np.sum(y_pred_numeric < y_true_numeric) / len(y_true_arr)

    # High quality recall: % of "Very High" listings correctly identified
    very_high_mask = y_true_arr == "Very High Rating"
    if very_high_mask.sum() > 0:
        high_quality_recall = (
            np.sum(
                (y_true_arr == "Very High Rating") & (y_pred_arr == "Very High Rating")
            )
            / very_high_mask.sum()
        )
    else:
        high_quality_recall = 0.0

    # Severe misclassification rate: Off by 2 levels (Medium ↔ Very High)
    #    Most harmful predictions for both guests and hosts
    severe_error = np.sum(np.abs(y_pred_numeric - y_true_numeric) >= 2) / len(
        y_true_arr
    )

    # Host confidence score: For hosts, consistent quality recognition
    #    (1 - undersell_rate) = how often we don't undersell quality
    host_confidence = 1 - undersell

    return {
        "over_promise_rate": over_promise,
        "undersell_rate": undersell,
        "high_quality_recall": high_quality_recall,
        "severe_misclassification_rate": severe_error,
        "host_confidence_score": host_confidence,
    }


def error_analysis(y_true, y_pred, X_test=None, feature_names=None) -> dict:
    """Produce confusion matrix and identify worst failure modes."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    analysis = {
        "confusion_matrix": cm,
        "classification_report": report,
        "total_errors": np.sum(np.array(y_true) != np.array(y_pred)),
        "error_rate": 1 - accuracy_score(y_true, y_pred),
    }

    # If test data provided, analyze error patterns
    if X_test is not None and feature_names is not None:
        errors_mask = np.array(y_true) != np.array(y_pred)

        if errors_mask.sum() > 0:
            # Store error patterns for reporting
            analysis["error_patterns"] = {
                "total_errors": int(errors_mask.sum()),
                "error_percentage": float(errors_mask.sum() / len(y_true) * 100),
            }

            logger.info(
                f"  Total errors: {errors_mask.sum()} ({errors_mask.sum() / len(y_true) * 100:.2f}%)"
            )

    return analysis


def print_evaluation_summary(metrics_dict: dict, model_name: str = "Model"):
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("=" * 70)

    # Standard metrics
    print("\nSTANDARD METRICS:")
    print(f"  Accuracy:          {metrics_dict.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {metrics_dict.get('balanced_accuracy', 0):.4f}")
    print(f"  F1 Macro:          {metrics_dict.get('f1_macro', 0):.4f}")
    print(f"  F1 Weighted:       {metrics_dict.get('f1_weighted', 0):.4f}")

    # Per-class metrics
    print("\nPER-CLASS PERFORMANCE:")
    for cls in ["Medium Rating", "High Rating", "Very High Rating"]:
        f1 = metrics_dict.get(f"f1_{cls}", 0)
        recall = metrics_dict.get(f"recall_{cls}", 0)
        precision = metrics_dict.get(f"precision_{cls}", 0)
        print(f"  {cls}:")
        print(f"    F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # Business metrics
    print("\nBUSINESS METRICS:")
    print(f"  High Quality Recall:    {metrics_dict.get('high_quality_recall', 0):.4f}")
    print(f"  Over-Promise Rate:      {metrics_dict.get('over_promise_rate', 0):.4f}")
    print(f"  Undersell Rate:         {metrics_dict.get('undersell_rate', 0):.4f}")
    print(
        f"  Severe Misclass Rate:   {metrics_dict.get('severe_misclassification_rate', 0):.4f}"
    )
    print(
        f"  Host Confidence Score:  {metrics_dict.get('host_confidence_score', 0):.4f}"
    )

    print("=" * 70)
