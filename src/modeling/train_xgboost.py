"""
XGBoost model training for partner churn prediction.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Tuple, Dict, Optional, List
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class ChurnPredictor:
    """
    XGBoost-based churn prediction model.
    Handles training, prediction, and probability estimation.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the churn predictor.

        Args:
            params: XGBoost parameters (default: from Config)
        """
        self.params = params or Config.XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.is_trained = False

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: int = 20
    ) -> 'ChurnPredictor':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Initialize model
        self.model = XGBClassifier(**self.params)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Calculate feature importance
        self._calculate_feature_importance()

        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn labels.

        Args:
            X: Features DataFrame

        Returns:
            Array of predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Array of probabilities for class 1 (churn)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]

    def get_risk_tier(self, probabilities: np.ndarray) -> List[str]:
        """
        Assign risk tiers based on probabilities.

        Args:
            probabilities: Array of churn probabilities

        Returns:
            List of risk tier labels
        """
        tiers = []
        for prob in probabilities:
            if prob >= Config.RISK_TIERS['critical']['min_prob']:
                tiers.append('critical')
            elif prob >= Config.RISK_TIERS['high']['min_prob']:
                tiers.append('high')
            elif prob >= Config.RISK_TIERS['medium']['min_prob']:
                tiers.append('medium')
            elif prob >= Config.RISK_TIERS['low']['min_prob']:
                tiers.append('low')
            else:
                tiers.append('minimal')
        return tiers

    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.

        Args:
            top_n: Number of features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        return self.feature_importance.head(top_n)

    def cross_validate(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            cv: int = 5,
            scoring: str = 'roc_auc'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dict with mean and std of scores
        """
        model = XGBClassifier(**self.params)
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True,
                                       random_state=Config.RANDOM_SEED)

        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)

        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }

    def save(self, path: Optional[Path] = None):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        path = path or Config.get_model_path()
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'params': self.params
        }, path)
        print(f"Model saved to: {path}")

    def load(self, path: Optional[Path] = None) -> 'ChurnPredictor':
        """Load a trained model from disk."""
        path = path or Config.get_model_path()
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.feature_importance = data['feature_importance']
        self.params = data['params']
        self.is_trained = True
        print(f"Model loaded from: {path}")
        return self


def train_xgboost_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        save_model: bool = True
) -> Tuple[ChurnPredictor, Dict]:
    """
    Train XGBoost churn prediction model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for evaluation)
        y_test: Test labels (for evaluation)
        save_model: Whether to save the trained model

    Returns:
        Tuple of (trained ChurnPredictor, metrics dict)
    """
    print("Training XGBoost Churn Prediction Model...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Churn rate: {y_train.mean():.2%}")

    # Initialize predictor
    predictor = ChurnPredictor()

    # Perform cross-validation
    print("\n  Performing 5-fold cross-validation...")
    cv_results = predictor.cross_validate(X_train, y_train, cv=5, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

    # Train on full training set
    print("\n  Training on full training set...")
    predictor.train(X_train, y_train, X_test, y_test)

    # Calculate metrics
    metrics = {'cv_roc_auc_mean': cv_results['mean'], 'cv_roc_auc_std': cv_results['std']}

    if X_test is not None and y_test is not None:
        from src.modeling.evaluate_model import evaluate_model
        test_metrics = evaluate_model(predictor, X_test, y_test)
        metrics.update(test_metrics)

    # Print top features
    print("\n  Top 10 Most Important Features:")
    importance = predictor.get_feature_importance(10)
    for idx, row in importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # Save model
    if save_model:
        Config.ensure_directories()
        predictor.save()

    return predictor, metrics


if __name__ == "__main__":
    # Load preprocessed data
    train_data = pd.read_csv(Config.get_data_path('train_data.csv'))
    test_data = pd.read_csv(Config.get_data_path('test_data.csv'))

    X_train = train_data.drop('churned', axis=1)
    y_train = train_data['churned']
    X_test = test_data.drop('churned', axis=1)
    y_test = test_data['churned']

    # Train model
    predictor, metrics = train_xgboost_model(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
