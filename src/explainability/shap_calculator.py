"""
SHAP (SHapley Additive exPlanations) calculator for model explainability.
Provides feature importance and individual prediction explanations.
"""
import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.feature_schema import FeatureSchema


class SHAPCalculator:
    """
    Calculate and manage SHAP values for churn predictions.
    Provides both global and local (per-partner) explanations.
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize SHAP calculator.

        Args:
            model: Trained model (XGBoost)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.is_fitted = False

    def fit(
            self,
            X_background: pd.DataFrame,
            n_samples: int = None
    ) -> 'SHAPCalculator':
        """
        Fit the SHAP explainer with background data.

        Args:
            X_background: Background dataset for SHAP
            n_samples: Number of background samples to use

        Returns:
            self
        """
        n_samples = n_samples or Config.SHAP_CONFIG['background_samples']

        # Sample background data if needed
        if len(X_background) > n_samples:
            X_background = X_background.sample(n=n_samples, random_state=Config.RANDOM_SEED)

        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model.model)
        self.expected_value = self.explainer.expected_value

        # Handle different expected_value formats
        if isinstance(self.expected_value, np.ndarray):
            self.expected_value = self.expected_value[1]  # For binary classification

        self.is_fitted = True
        return self

    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for given data.

        Args:
            X: Features DataFrame

        Returns:
            Array of SHAP values
        """
        if not self.is_fitted:
            raise ValueError("Calculator must be fitted first")

        shap_values = self.explainer.shap_values(X)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        self.shap_values = shap_values
        return shap_values

    def get_top_drivers(
            self,
            partner_idx: int,
            X: pd.DataFrame,
            top_k: int = None
    ) -> pd.DataFrame:
        """
        Get top K drivers for a specific partner's prediction.

        Args:
            partner_idx: Index of partner in X
            X: Features DataFrame
            top_k: Number of top features to return

        Returns:
            DataFrame with feature names, values, and SHAP contributions
        """
        top_k = top_k or Config.SHAP_CONFIG['top_k_features']

        if self.shap_values is None:
            self.calculate_shap_values(X)

        partner_shap = self.shap_values[partner_idx]
        partner_values = X.iloc[partner_idx]

        # Create driver DataFrame
        drivers = pd.DataFrame({
            'feature': self.feature_names,
            'value': partner_values.values,
            'shap_value': partner_shap,
            'abs_shap': np.abs(partner_shap)
        })

        # Sort by absolute SHAP value and get top K
        drivers = drivers.sort_values('abs_shap', ascending=False).head(top_k)

        # Add direction indicator
        drivers['direction'] = drivers['shap_value'].apply(
            lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
        )

        return drivers[['feature', 'value', 'shap_value', 'direction']].reset_index(drop=True)

    def get_global_importance(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.

        Args:
            X: Optional features DataFrame to calculate on

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.shap_values is None:
            if X is not None:
                self.calculate_shap_values(X)
            else:
                raise ValueError("No SHAP values available. Provide X to calculate.")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        # Normalize to percentages
        importance['importance_pct'] = importance['importance'] / importance['importance'].sum() * 100

        return importance.reset_index(drop=True)

    def explain_prediction(
            self,
            partner_idx: int,
            X: pd.DataFrame,
            probability: float
    ) -> Dict:
        """
        Generate a complete explanation for a single prediction.

        Args:
            partner_idx: Index of partner
            X: Features DataFrame
            probability: Predicted churn probability

        Returns:
            Dict with explanation components
        """
        top_drivers = self.get_top_drivers(partner_idx, X)

        # Get category for each driver
        from src.explainability.driver_categorizer import DriverCategorizer
        categorizer = DriverCategorizer()

        categorized_drivers = categorizer.categorize_drivers(top_drivers)

        explanation = {
            'partner_idx': partner_idx,
            'churn_probability': probability,
            'risk_level': self._get_risk_level(probability),
            'base_probability': 1 / (1 + np.exp(-self.expected_value)),
            'top_drivers': categorized_drivers.to_dict('records'),
            'summary': self._generate_summary(categorized_drivers, probability)
        }

        return explanation

    def _get_risk_level(self, probability: float) -> str:
        """Map probability to risk level."""
        for tier, config in Config.RISK_TIERS.items():
            if probability >= config['min_prob']:
                return tier
        return 'minimal'

    def _generate_summary(
            self,
            drivers: pd.DataFrame,
            probability: float
    ) -> str:
        """Generate human-readable summary of prediction."""
        risk_level = self._get_risk_level(probability)

        # Get top increasing and decreasing factors
        increasing = drivers[drivers['direction'] == 'Increases Risk']
        decreasing = drivers[drivers['direction'] == 'Decreases Risk']

        summary_parts = [
            f"This partner has a {probability:.1%} probability of churning ({risk_level} risk)."
        ]

        if len(increasing) > 0:
            top_risk = increasing.iloc[0]
            summary_parts.append(
                f"The main risk factor is {top_risk['feature']} "
                f"(current value: {top_risk['value']:.2f})."
            )

        if len(decreasing) > 0:
            top_protect = decreasing.iloc[0]
            summary_parts.append(
                f"A protective factor is {top_protect['feature']} "
                f"(current value: {top_protect['value']:.2f})."
            )

        return " ".join(summary_parts)

    def batch_explain(
            self,
            X: pd.DataFrame,
            probabilities: np.ndarray,
            partner_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate explanations for multiple partners.

        Args:
            X: Features DataFrame
            probabilities: Array of churn probabilities
            partner_ids: Optional list of partner IDs

        Returns:
            DataFrame with explanations for all partners
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)

        records = []

        for idx in range(len(X)):
            top_drivers = self.get_top_drivers(idx, X)

            record = {
                'partner_id': partner_ids[idx] if partner_ids else idx,
                'churn_probability': probabilities[idx],
                'risk_level': self._get_risk_level(probabilities[idx]),
            }

            # Add top 5 drivers
            for i, (_, driver) in enumerate(top_drivers.iterrows(), 1):
                record[f'driver_{i}'] = driver['feature']
                record[f'driver_{i}_shap'] = driver['shap_value']
                record[f'driver_{i}_direction'] = driver['direction']

            records.append(record)

        return pd.DataFrame(records)

    def get_feature_interaction(
            self,
            feature1: str,
            feature2: str,
            X: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate SHAP interaction values between two features.

        Args:
            feature1: First feature name
            feature2: Second feature name
            X: Features DataFrame

        Returns:
            Array of interaction values
        """
        if not self.is_fitted:
            raise ValueError("Calculator must be fitted first")

        # Calculate interaction values
        shap_interaction = self.explainer.shap_interaction_values(X)

        if isinstance(shap_interaction, list):
            shap_interaction = shap_interaction[1]

        idx1 = self.feature_names.index(feature1)
        idx2 = self.feature_names.index(feature2)

        return shap_interaction[:, idx1, idx2]


if __name__ == "__main__":
    # Test SHAP calculator
    from src.modeling.train_xgboost import ChurnPredictor

    # Load model and data
    model = ChurnPredictor().load()
    test_data = pd.read_csv(Config.get_data_path('test_data.csv'))
    X_test = test_data.drop('churned', axis=1)
    y_test = test_data['churned']

    # Initialize calculator
    calculator = SHAPCalculator(model, list(X_test.columns))
    calculator.fit(X_test)

    # Calculate SHAP values
    shap_values = calculator.calculate_shap_values(X_test)

    # Get global importance
    print("Global Feature Importance:")
    importance = calculator.get_global_importance()
    print(importance.head(10))

    # Explain a single prediction
    print("\n\nExplaining prediction for partner 0:")
    probs = model.predict_proba(X_test)
    explanation = calculator.explain_prediction(0, X_test, probs[0])
    print(f"Probability: {explanation['churn_probability']:.1%}")
    print(f"Risk Level: {explanation['risk_level']}")
    print(f"\nSummary: {explanation['summary']}")
    print("\nTop Drivers:")
    for driver in explanation['top_drivers']:
        print(f"  - {driver['feature']}: {driver['shap_value']:.4f} ({driver['direction']})")
