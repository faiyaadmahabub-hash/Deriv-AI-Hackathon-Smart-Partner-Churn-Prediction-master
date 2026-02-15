"""
Pattern mining for discovering new churn signals.
Identifies patterns not explicitly engineered in the feature set.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class PatternMiner:
    """
    Discover new churn patterns from historical data.
    Uses SHAP interactions and segment analysis to find hidden signals.
    """

    def __init__(self):
        """Initialize pattern miner."""
        self.min_sample_size = 10  # Minimum samples to consider pattern valid
        self.significance_threshold = 1.5  # Minimum ratio for pattern significance

    def mine_shap_interactions(
            self,
            shap_df: pd.DataFrame,
            feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Dict]:
        """
        Find feature interactions that strongly predict churn.

        Args:
            shap_df: DataFrame with SHAP values and churn labels
            feature_pairs: Optional specific pairs to analyze

        Returns:
            List of discovered interaction patterns
        """
        patterns = []

        if 'churned' not in shap_df.columns:
            return patterns

        churned = shap_df[shap_df['churned'] == 1]
        retained = shap_df[shap_df['churned'] == 0]

        if len(churned) < self.min_sample_size or len(retained) < self.min_sample_size:
            return patterns

        # Default feature pairs to check
        if feature_pairs is None:
            feature_pairs = [
                ('commission_decline_30d', 'payout_delay_avg_days_90d'),
                ('days_since_last_login', 'commission_trend_30d'),
                ('competitor_mention_count_90d', 'pricing_complaint_count_90d'),
                ('churned_clients_30d', 'client_trading_volume_trend'),
                ('unresolved_tickets_count', 'days_oldest_unresolved_ticket')
            ]

        for feat1, feat2 in feature_pairs:
            if feat1 not in shap_df.columns or feat2 not in shap_df.columns:
                continue

            # Check if both features being high is associated with churn
            pattern = self._analyze_feature_pair(shap_df, churned, retained, feat1, feat2)
            if pattern:
                patterns.append(pattern)

        return patterns

    def _analyze_feature_pair(
            self,
            df: pd.DataFrame,
            churned: pd.DataFrame,
            retained: pd.DataFrame,
            feat1: str,
            feat2: str
    ) -> Optional[Dict]:
        """Analyze a specific feature pair for interaction effects."""
        # Get median thresholds
        med1 = df[feat1].median()
        med2 = df[feat2].median()

        # Count partners with both features above median
        churned_both_high = ((churned[feat1] > med1) & (churned[feat2] > med2)).sum()
        retained_both_high = ((retained[feat1] > med1) & (retained[feat2] > med2)).sum()

        churned_rate_both = churned_both_high / len(churned) if len(churned) > 0 else 0
        retained_rate_both = retained_both_high / len(retained) if len(retained) > 0 else 0

        # Check significance
        if retained_rate_both > 0:
            ratio = churned_rate_both / retained_rate_both
        else:
            ratio = churned_rate_both * 10 if churned_rate_both > 0 else 0

        if ratio >= self.significance_threshold and churned_both_high >= 5:
            return {
                'type': 'feature_interaction',
                'features': [feat1, feat2],
                'pattern': f'Both {feat1} and {feat2} elevated',
                'churned_rate': round(churned_rate_both, 3),
                'retained_rate': round(retained_rate_both, 3),
                'lift_ratio': round(ratio, 2),
                'affected_churners': int(churned_both_high),
                'insight': (
                    f"Partners with elevated {feat1} AND {feat2} are "
                    f"{ratio:.1f}x more likely to churn"
                ),
                'recommendation': f"Flag partners with combined {feat1} and {feat2} issues urgently"
            }

        return None

    def mine_segment_patterns(
            self,
            df: pd.DataFrame,
            prediction_col: str = 'predicted_churn',
            actual_col: str = 'churned'
    ) -> List[Dict]:
        """
        Find segments where the model underperforms.

        Args:
            df: DataFrame with predictions and actuals
            prediction_col: Column with model predictions
            actual_col: Column with actual outcomes

        Returns:
            List of segment patterns with poor model performance
        """
        patterns = []

        if prediction_col not in df.columns or actual_col not in df.columns:
            return patterns

        # Check segments by tier + region
        segment_cols = []
        if 'tier' in df.columns:
            segment_cols.append('tier')
        if 'partner_tier' in df.columns:
            segment_cols.append('partner_tier')
        if 'region' in df.columns:
            segment_cols.append('region')
        if 'partner_region' in df.columns:
            segment_cols.append('partner_region')

        if len(segment_cols) < 2:
            return patterns

        # Use first two segment columns
        col1, col2 = segment_cols[0], segment_cols[1]

        for segment_values, group in df.groupby([col1, col2]):
            if len(group) < self.min_sample_size:
                continue

            # Calculate accuracy
            correct = (group[prediction_col] == group[actual_col]).sum()
            accuracy = correct / len(group)

            # Check for significant underperformance
            if accuracy < 0.70:
                # Analyze why model might be failing
                churn_rate = group[actual_col].mean()

                patterns.append({
                    'type': 'model_underperformance',
                    'segment': f"{segment_values[0]}_{segment_values[1]}",
                    'segment_values': {col1: segment_values[0], col2: segment_values[1]},
                    'sample_size': len(group),
                    'accuracy': round(accuracy, 3),
                    'actual_churn_rate': round(churn_rate, 3),
                    'insight': (
                        f"Model accuracy is only {accuracy:.0%} for "
                        f"{segment_values[0]} {segment_values[1]} partners. "
                        f"Actual churn rate: {churn_rate:.0%}"
                    ),
                    'recommendation': 'Investigate segment-specific churn drivers'
                })

        return patterns

    def mine_temporal_patterns(
            self,
            intervention_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Find temporal patterns in churn behavior.

        Args:
            intervention_df: DataFrame with intervention and outcome data

        Returns:
            List of temporal patterns
        """
        patterns = []

        if intervention_df.empty:
            return patterns

        # Check if we have date information
        if 'intervention_date' not in intervention_df.columns:
            return patterns

        try:
            intervention_df['date'] = pd.to_datetime(intervention_df['intervention_date'])
            intervention_df['month'] = intervention_df['date'].dt.month
            intervention_df['day_of_week'] = intervention_df['date'].dt.dayofweek
        except Exception:
            return patterns

        # Analyze success rates by timing
        if 'outcome_30d' in intervention_df.columns:
            completed = intervention_df[intervention_df['outcome_30d'].notna()].copy()

            if len(completed) >= 20:
                completed['success'] = completed['outcome_30d'].isin(['Retained', 'Stabilized'])

                # Check day of week patterns
                dow_success = completed.groupby('day_of_week')['success'].agg(['mean', 'count'])
                dow_success = dow_success[dow_success['count'] >= 3]

                if len(dow_success) > 0:
                    best_day = dow_success['mean'].idxmax()
                    worst_day = dow_success['mean'].idxmin()

                    if dow_success.loc[best_day, 'mean'] - dow_success.loc[worst_day, 'mean'] > 0.15:
                        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        patterns.append({
                            'type': 'temporal_pattern',
                            'pattern': 'day_of_week_effect',
                            'best_day': day_names[best_day],
                            'best_success_rate': round(dow_success.loc[best_day, 'mean'], 3),
                            'worst_day': day_names[worst_day],
                            'worst_success_rate': round(dow_success.loc[worst_day, 'mean'], 3),
                            'insight': (
                                f"Interventions on {day_names[best_day]} have "
                                f"{dow_success.loc[best_day, 'mean']:.0%} success rate vs "
                                f"{dow_success.loc[worst_day, 'mean']:.0%} on {day_names[worst_day]}"
                            ),
                            'recommendation': f"Prioritize interventions on {day_names[best_day]}"
                        })

        return patterns

    def get_all_patterns(
            self,
            shap_df: Optional[pd.DataFrame] = None,
            prediction_df: Optional[pd.DataFrame] = None,
            intervention_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run all pattern mining analyses.

        Args:
            shap_df: DataFrame with SHAP values
            prediction_df: DataFrame with predictions and actuals
            intervention_df: DataFrame with intervention data

        Returns:
            All discovered patterns organized by type
        """
        all_patterns = {
            'shap_interactions': [],
            'segment_patterns': [],
            'temporal_patterns': [],
            'total_patterns': 0
        }

        if shap_df is not None:
            all_patterns['shap_interactions'] = self.mine_shap_interactions(shap_df)

        if prediction_df is not None:
            all_patterns['segment_patterns'] = self.mine_segment_patterns(prediction_df)

        if intervention_df is not None:
            all_patterns['temporal_patterns'] = self.mine_temporal_patterns(intervention_df)

        all_patterns['total_patterns'] = (
            len(all_patterns['shap_interactions']) +
            len(all_patterns['segment_patterns']) +
            len(all_patterns['temporal_patterns'])
        )

        return all_patterns

    def get_patterns_summary_df(self, patterns: Dict) -> pd.DataFrame:
        """Create summary DataFrame of all patterns."""
        records = []

        for pattern in patterns.get('shap_interactions', []):
            records.append({
                'Type': 'Feature Interaction',
                'Pattern': pattern['pattern'],
                'Lift': f"{pattern['lift_ratio']:.1f}x",
                'Insight': pattern['insight'][:80] + '...' if len(pattern['insight']) > 80 else pattern['insight']
            })

        for pattern in patterns.get('segment_patterns', []):
            records.append({
                'Type': 'Model Gap',
                'Pattern': pattern['segment'],
                'Lift': f"{pattern['accuracy']:.0%} acc",
                'Insight': pattern['insight'][:80] + '...' if len(pattern['insight']) > 80 else pattern['insight']
            })

        for pattern in patterns.get('temporal_patterns', []):
            records.append({
                'Type': 'Timing',
                'Pattern': pattern['pattern'],
                'Lift': f"{pattern.get('best_success_rate', 0):.0%}",
                'Insight': pattern['insight'][:80] + '...' if len(pattern['insight']) > 80 else pattern['insight']
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    # Test pattern miner
    miner = PatternMiner()

    print("Pattern Miner Test")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 200

    sample_shap = pd.DataFrame({
        'partner_id': [f'P{i:03d}' for i in range(n)],
        'churned': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'tier': np.random.choice(['gold', 'silver', 'bronze'], n),
        'region': np.random.choice(['SEA', 'MENA', 'Europe'], n),
        'commission_decline_30d': np.random.uniform(0, 1, n),
        'payout_delay_avg_days_90d': np.random.uniform(0, 30, n),
        'days_since_last_login': np.random.uniform(0, 60, n),
        'competitor_mention_count_90d': np.random.poisson(1, n)
    })

    # Make churned partners have higher values for key features
    churned_mask = sample_shap['churned'] == 1
    sample_shap.loc[churned_mask, 'commission_decline_30d'] *= 1.5
    sample_shap.loc[churned_mask, 'payout_delay_avg_days_90d'] *= 1.8

    patterns = miner.mine_shap_interactions(sample_shap)

    print(f"\nDiscovered {len(patterns)} interaction patterns:")
    for p in patterns:
        print(f"\n  {p['pattern']}")
        print(f"  Lift: {p['lift_ratio']:.1f}x")
        print(f"  Insight: {p['insight']}")
