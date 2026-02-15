"""
Effectiveness scorer for intervention success rate analysis.
Calculates and tracks which interventions work best for different situations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class EffectivenessScorer:
    """
    Calculate success rates for different intervention types.
    Provides historical success rate data for ROI calculations.
    """

    def __init__(self):
        """Initialize effectiveness scorer."""
        self.success_outcomes = ['Retained', 'Stabilized', 'Recovered']
        self.failure_outcomes = ['Churned', 'Declined', 'Lost']

    def calculate_effectiveness(
            self,
            outcomes_df: pd.DataFrame
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Calculate success rates for different intervention types by driver category.

        Args:
            outcomes_df: DataFrame from InterventionTracker

        Returns:
            Effectiveness metrics by (intervention_type, driver_category)
        """
        # Filter for completed interventions
        completed = outcomes_df[outcomes_df['outcome_30d'].notna()].copy()

        if len(completed) == 0:
            return {}

        # Define success
        completed['success'] = completed['outcome_30d'].isin(self.success_outcomes)

        # Group by intervention type and driver category
        effectiveness = {}

        for (int_type, driver_cat), group in completed.groupby(['intervention_type', 'driver_category']):
            if pd.isna(int_type) or pd.isna(driver_cat):
                continue

            success_count = group['success'].sum()
            total_count = len(group)
            success_rate = success_count / total_count

            # Calculate average metrics
            avg_commission_change = group['commission_change_30d_pct'].mean()
            avg_cost = group['intervention_cost'].mean()

            # Calculate average risk reduction
            risk_at_start = group['risk_score_at_trigger'].mean()
            risk_at_30d = group['risk_score_30d'].mean()
            avg_risk_reduction = risk_at_start - risk_at_30d if not pd.isna(risk_at_30d) else 0

            # Calculate ROI (commission recovery over cost)
            avg_monthly_value = avg_commission_change * 100  # Rough estimate
            roi = (avg_monthly_value - avg_cost) / avg_cost if avg_cost > 0 else 0

            effectiveness[(int_type, driver_cat)] = {
                'successful_count': int(success_count),
                'total_interventions': int(total_count),
                'success_rate': round(success_rate, 3),
                'avg_commission_recovery_pct': round(avg_commission_change, 1) if not pd.isna(avg_commission_change) else 0,
                'avg_cost': round(avg_cost, 2),
                'avg_risk_reduction': round(avg_risk_reduction, 3),
                'estimated_roi': round(roi, 2),
                'confidence': self._get_confidence_level(total_count)
            }

        return effectiveness

    def _get_confidence_level(self, sample_size: int) -> str:
        """Determine confidence level based on sample size."""
        if sample_size >= 30:
            return 'HIGH'
        elif sample_size >= 10:
            return 'MEDIUM'
        elif sample_size >= 3:
            return 'LOW'
        return 'VERY_LOW'

    def get_best_intervention(
            self,
            effectiveness: Dict,
            driver_category: str
    ) -> Optional[Dict]:
        """
        Get the best intervention type for a given driver category.

        Args:
            effectiveness: Output from calculate_effectiveness
            driver_category: The driver category to look up

        Returns:
            Best intervention with its metrics
        """
        matching = {
            k: v for k, v in effectiveness.items()
            if k[1] == driver_category
        }

        if not matching:
            return None

        # Find highest success rate with reasonable confidence
        best_key = max(
            matching.items(),
            key=lambda x: x[1]['success_rate'] * (1 if x[1]['confidence'] != 'VERY_LOW' else 0.5)
        )

        return {
            'intervention_type': best_key[0][0],
            'driver_category': best_key[0][1],
            **best_key[1]
        }

    def get_effectiveness_summary_df(
            self,
            effectiveness: Dict
    ) -> pd.DataFrame:
        """
        Create summary DataFrame for dashboard display.

        Args:
            effectiveness: Output from calculate_effectiveness

        Returns:
            Summary DataFrame
        """
        records = []

        for (int_type, driver_cat), metrics in effectiveness.items():
            records.append({
                'Intervention': int_type.replace('_', ' ').title(),
                'Driver': driver_cat,
                'Success Rate': f"{metrics['success_rate']:.0%}",
                'Interventions': metrics['total_interventions'],
                'Avg Recovery': f"{metrics['avg_commission_recovery_pct']:+.1f}%",
                'Avg Cost': f"${metrics['avg_cost']:,.0f}",
                'ROI': f"{metrics['estimated_roi']:.1f}x",
                'Confidence': metrics['confidence']
            })

        df = pd.DataFrame(records)
        if not df.empty:
            # Sort by success rate descending
            df = df.sort_values('Success Rate', ascending=False)

        return df

    def calculate_tier_effectiveness(
            self,
            outcomes_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Calculate effectiveness by partner tier.

        Args:
            outcomes_df: DataFrame from InterventionTracker

        Returns:
            Effectiveness metrics by tier
        """
        completed = outcomes_df[outcomes_df['outcome_30d'].notna()].copy()

        if len(completed) == 0:
            return {}

        completed['success'] = completed['outcome_30d'].isin(self.success_outcomes)

        tier_effectiveness = {}

        for tier in completed['risk_tier_at_trigger'].dropna().unique():
            tier_data = completed[completed['risk_tier_at_trigger'] == tier]

            tier_effectiveness[tier] = {
                'total_interventions': len(tier_data),
                'success_rate': round(tier_data['success'].mean(), 3),
                'avg_cost': round(tier_data['intervention_cost'].mean(), 2),
                'best_intervention': tier_data.groupby('intervention_type')['success'].mean().idxmax() if len(tier_data) > 0 else None
            }

        return tier_effectiveness

    def get_intervention_recommendations(
            self,
            driver_category: str,
            risk_tier: str,
            effectiveness: Dict,
            tier_effectiveness: Dict
    ) -> Dict:
        """
        Get intervention recommendation based on driver and tier.

        Args:
            driver_category: Primary driver category
            risk_tier: Partner's risk tier
            effectiveness: Output from calculate_effectiveness
            tier_effectiveness: Output from calculate_tier_effectiveness

        Returns:
            Recommendation with expected success rate
        """
        # Find best intervention for driver
        best = self.get_best_intervention(effectiveness, driver_category)

        if best is None:
            # Use default recommendations
            defaults = {
                'Financial': 'dispute_resolution',
                'Engagement': 'personal_outreach',
                'Competitive': 'rate_review',
                'Client-base': 'client_reactivation',
                'Support': 'escalation_resolution'
            }

            return {
                'intervention_type': defaults.get(driver_category, 'personal_outreach'),
                'expected_success_rate': 0.50,
                'confidence': 'DEFAULT',
                'source': 'default_mapping'
            }

        # Adjust for tier
        tier_data = tier_effectiveness.get(risk_tier, {})
        tier_modifier = 1.0

        if tier_data.get('success_rate'):
            # Adjust based on tier performance
            overall_avg = sum(t.get('success_rate', 0.5) for t in tier_effectiveness.values()) / len(tier_effectiveness)
            tier_modifier = tier_data['success_rate'] / overall_avg if overall_avg > 0 else 1.0

        adjusted_rate = min(0.95, best['success_rate'] * tier_modifier)

        return {
            'intervention_type': best['intervention_type'],
            'expected_success_rate': round(adjusted_rate, 3),
            'base_success_rate': best['success_rate'],
            'tier_adjustment': round(tier_modifier, 2),
            'confidence': best['confidence'],
            'estimated_cost': best['avg_cost'],
            'historical_sample_size': best['total_interventions'],
            'source': 'historical_data'
        }


if __name__ == "__main__":
    # Test effectiveness scorer
    scorer = EffectivenessScorer()

    print("Effectiveness Scorer Test")
    print("=" * 60)

    # Create sample outcomes data
    np.random.seed(42)
    sample_outcomes = pd.DataFrame({
        'intervention_id': [f'INT_{i}' for i in range(50)],
        'partner_id': [f'P{i:03d}' for i in range(50)],
        'intervention_type': np.random.choice(
            ['dispute_resolution', 'personal_outreach', 'rate_review', 'client_reactivation'],
            50
        ),
        'driver_category': np.random.choice(
            ['Financial', 'Engagement', 'Competitive', 'Client-base'],
            50
        ),
        'risk_tier_at_trigger': np.random.choice(['high', 'medium'], 50),
        'risk_score_at_trigger': np.random.uniform(0.5, 0.9, 50),
        'risk_score_30d': np.random.uniform(0.2, 0.6, 50),
        'intervention_cost': np.random.choice([100, 150, 300, 500, 800], 50),
        'commission_change_30d_pct': np.random.uniform(-10, 25, 50),
        'outcome_30d': np.random.choice(
            ['Retained', 'Stabilized', 'Churned', 'Declined'],
            50,
            p=[0.35, 0.25, 0.25, 0.15]
        )
    })

    effectiveness = scorer.calculate_effectiveness(sample_outcomes)

    print("\nEffectiveness by (Intervention, Driver):")
    for key, metrics in list(effectiveness.items())[:5]:
        print(f"  {key}: {metrics['success_rate']:.0%} ({metrics['total_interventions']} interventions)")

    print("\nBest intervention for 'Financial' drivers:")
    best = scorer.get_best_intervention(effectiveness, 'Financial')
    if best:
        print(f"  {best['intervention_type']}: {best['success_rate']:.0%} success rate")

    print("\nSummary DataFrame:")
    summary = scorer.get_effectiveness_summary_df(effectiveness)
    print(summary.head(5).to_string(index=False))
