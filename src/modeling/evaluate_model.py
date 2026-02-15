"""
Model evaluation metrics and utilities for churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


def evaluate_model(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: True labels
        threshold: Classification threshold

    Returns:
        Dict of evaluation metrics
    """
    # Get predictions
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= threshold).astype(int)
    y_test = y_test.reset_index(drop=True)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]

    # Business metrics
    metrics['total_churners'] = int(y_test.sum())
    metrics['identified_churners'] = int(((y_pred == 1) & (y_test == 1)).sum())
    metrics['missed_churners'] = int(((y_pred == 0) & (y_test == 1)).sum())
    metrics['false_alarms'] = int(((y_pred == 1) & (y_test == 0)).sum())

    # Capture rate at different thresholds
    for thresh in [0.3, 0.5, 0.7]:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        capture_rate = recall_score(y_test, y_pred_thresh, zero_division=0)
        metrics[f'capture_rate_{int(thresh * 100)}'] = capture_rate

    return metrics


def find_optimal_threshold(
        y_true: pd.Series,
        y_prob: np.ndarray,
        optimize_for: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        optimize_for: Metric to optimize ('f1', 'recall', 'precision', 'balanced')

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        if optimize_for == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif optimize_for == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif optimize_for == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif optimize_for == 'balanced':
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * (precision * recall) / (precision + recall + 1e-10)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Calculate metrics at optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    metrics = {
        'threshold': best_threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    return best_threshold, metrics


def calculate_business_impact(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        partner_values: Optional[pd.Series] = None
) -> Dict:
    """
    Calculate business impact of predictions.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        partner_values: Optional partner monthly values

    Returns:
        Dict with business impact metrics
    """
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    y_test_reset = y_test.reset_index(drop=True)

    # If no partner values provided, use tier-based estimates
    if partner_values is None:
        # Assume average monthly commission of $5000
        partner_values = pd.Series([5000] * len(y_test_reset))

    # Calculate potential saved revenue
    correctly_identified = (y_pred == 1) & (y_test_reset == 1)
    potential_save_value = partner_values[correctly_identified].sum() * 12  # Annual

    # Missed churners (false negatives)
    missed = (y_pred == 0) & (y_test_reset == 1)
    missed_value = partner_values[missed].sum() * 12

    # False alarms (false positives) - intervention cost
    false_alarms = (y_pred == 1) & (y_test_reset == 0)
    intervention_cost = len(false_alarms[false_alarms]) * Config.INTERVENTION_COSTS['account_review']

    # Net value
    net_value = potential_save_value - intervention_cost

    return {
        'potential_saved_revenue': potential_save_value,
        'missed_churn_value': missed_value,
        'intervention_cost': intervention_cost,
        'net_value': net_value,
        'roi': (net_value / intervention_cost) * 100 if intervention_cost > 0 else float('inf'),
        'partners_to_contact': int(y_pred.sum()),
        'expected_saves': int(correctly_identified.sum())
    }


def generate_evaluation_report(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = 'XGBoost Churn Predictor'
) -> str:
    """
    Generate a comprehensive evaluation report.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Name for the report

    Returns:
        Formatted report string
    """
    metrics = evaluate_model(model, X_test, y_test)
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    y_test = y_test.reset_index(drop=True)

    report = []
    report.append("=" * 60)
    report.append(f"MODEL EVALUATION REPORT: {model_name}")
    report.append("=" * 60)

    report.append("\n1. OVERALL PERFORMANCE")
    report.append("-" * 40)
    report.append(f"  Accuracy:      {metrics['accuracy']:.4f}")
    report.append(f"  Precision:     {metrics['precision']:.4f}")
    report.append(f"  Recall:        {metrics['recall']:.4f}")
    report.append(f"  F1 Score:      {metrics['f1']:.4f}")
    report.append(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
    report.append(f"  PR-AUC:        {metrics['pr_auc']:.4f}")

    report.append("\n2. CONFUSION MATRIX")
    report.append("-" * 40)
    report.append(f"  True Negatives:  {metrics['true_negatives']}")
    report.append(f"  False Positives: {metrics['false_positives']}")
    report.append(f"  False Negatives: {metrics['false_negatives']}")
    report.append(f"  True Positives:  {metrics['true_positives']}")

    report.append("\n3. CHURN DETECTION")
    report.append("-" * 40)
    report.append(f"  Total Churners:      {metrics['total_churners']}")
    report.append(f"  Identified (TP):     {metrics['identified_churners']}")
    report.append(f"  Missed (FN):         {metrics['missed_churners']}")
    report.append(f"  False Alarms (FP):   {metrics['false_alarms']}")

    report.append("\n4. CAPTURE RATE BY THRESHOLD")
    report.append("-" * 40)
    report.append(f"  At 30% threshold:  {metrics['capture_rate_30']:.2%}")
    report.append(f"  At 50% threshold:  {metrics['capture_rate_50']:.2%}")
    report.append(f"  At 70% threshold:  {metrics['capture_rate_70']:.2%}")

    # Find optimal threshold
    opt_thresh, opt_metrics = find_optimal_threshold(y_test, y_prob, 'f1')
    report.append("\n5. OPTIMAL THRESHOLD (for F1)")
    report.append("-" * 40)
    report.append(f"  Optimal Threshold: {opt_thresh:.2f}")
    report.append(f"  Precision:         {opt_metrics['precision']:.4f}")
    report.append(f"  Recall:            {opt_metrics['recall']:.4f}")
    report.append(f"  F1:                {opt_metrics['f1']:.4f}")

    # Business impact
    business = calculate_business_impact(model, X_test, y_test)
    report.append("\n6. BUSINESS IMPACT (Estimated)")
    report.append("-" * 40)
    report.append(f"  Potential Saved Revenue: ${business['potential_saved_revenue']:,.0f}")
    report.append(f"  Missed Churn Value:      ${business['missed_churn_value']:,.0f}")
    report.append(f"  Intervention Cost:       ${business['intervention_cost']:,.0f}")
    report.append(f"  Net Value:               ${business['net_value']:,.0f}")
    report.append(f"  ROI:                     {business['roi']:.0f}%")

    report.append("\n" + "=" * 60)

    # Classification report
    report.append("\n7. DETAILED CLASSIFICATION REPORT")
    report.append("-" * 40)
    report.append(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

    return '\n'.join(report)


def compare_models(
        models: Dict[str, object],
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        models: Dict of model_name -> model
        X_test: Test features
        y_test: True labels

    Returns:
        DataFrame comparing model metrics
    """
    results = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index('model')

    # Select key metrics for comparison
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    return df[key_metrics].sort_values('pr_auc', ascending=False)


if __name__ == "__main__":
    # Test evaluation with mock model
    from src.modeling.train_xgboost import ChurnPredictor

    # Load test data
    test_data = pd.read_csv(Config.get_data_path('test_data.csv'))
    X_test = test_data.drop('churned', axis=1)
    y_test = test_data['churned']

    # Load model
    model = ChurnPredictor().load()

    # Generate report
    report = generate_evaluation_report(model, X_test, y_test)
    print(report)
