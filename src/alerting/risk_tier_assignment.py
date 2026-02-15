"""
Risk tier assignment for partner churn predictions.
Categorizes partners into actionable risk tiers.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class RiskTierAssigner:
    """
    Assign risk tiers to partners based on churn probability and other factors.
    Provides actionable categorization for relationship managers.
    """

    def __init__(self):
        """Initialize with default tier thresholds from config."""
        self.tiers = Config.RISK_TIERS
        self.tier_order = ['critical', 'high', 'medium', 'low', 'minimal']

    def assign_tier(self, probability: float) -> str:
        """
        Assign a single probability to a risk tier.

        Args:
            probability: Churn probability (0-1)

        Returns:
            Risk tier name
        """
        for tier in self.tier_order:
            if probability >= self.tiers[tier]['min_prob']:
                return tier
        return 'minimal'

    def assign_tiers(self, probabilities: np.ndarray) -> List[str]:
        """
        Assign risk tiers to an array of probabilities.

        Args:
            probabilities: Array of churn probabilities

        Returns:
            List of risk tier names
        """
        return [self.assign_tier(p) for p in probabilities]

    def get_tier_info(self, tier: str) -> Dict:
        """
        Get display information for a tier.

        Args:
            tier: Tier name

        Returns:
            Dict with tier details
        """
        info = self.tiers.get(tier, self.tiers['minimal']).copy()
        info['name'] = tier.title()
        info['description'] = self._get_tier_description(tier)
        info['action'] = self._get_tier_action(tier)
        return info

    def _get_tier_description(self, tier: str) -> str:
        """Get human-readable tier description."""
        descriptions = {
            'critical': 'Immediate action required. Partner at very high risk of churning within 2 weeks.',
            'high': 'Urgent attention needed. Partner showing strong churn signals.',
            'medium': 'Proactive intervention recommended. Early warning signs detected.',
            'low': 'Monitor closely. Some minor concerns but generally stable.',
            'minimal': 'Healthy partner relationship. Continue standard engagement.'
        }
        return descriptions.get(tier, 'Unknown tier')

    def _get_tier_action(self, tier: str) -> str:
        """Get recommended action for tier."""
        actions = {
            'critical': 'Escalate to senior RM. Personal call within 24 hours.',
            'high': 'Schedule priority review. Contact within 48 hours.',
            'medium': 'Add to watch list. Reach out within 1 week.',
            'low': 'Include in next regular check-in cycle.',
            'minimal': 'No special action needed.'
        }
        return actions.get(tier, 'Review case')

    def create_tier_summary(
            self,
            df: pd.DataFrame,
            probability_col: str = 'churn_probability'
    ) -> pd.DataFrame:
        """
        Create a summary of partners by risk tier.

        Args:
            df: DataFrame with partner data and probabilities
            probability_col: Column name for churn probability

        Returns:
            DataFrame with tier summary statistics
        """
        df = df.copy()
        df['risk_tier'] = self.assign_tiers(df[probability_col].values)

        summary = df.groupby('risk_tier').agg({
            probability_col: ['count', 'mean', 'min', 'max'],
            'commission_current_month': 'sum' if 'commission_current_month' in df.columns else 'count'
        }).round(4)

        summary.columns = ['count', 'avg_probability', 'min_probability',
                           'max_probability', 'total_commission']

        # Add tier order for sorting
        summary['tier_order'] = summary.index.map(
            lambda x: self.tier_order.index(x) if x in self.tier_order else 99
        )
        summary = summary.sort_values('tier_order').drop('tier_order', axis=1)

        # Add percentage
        total = summary['count'].sum()
        summary['percentage'] = (summary['count'] / total * 100).round(1)

        return summary

    def get_partners_by_tier(
            self,
            df: pd.DataFrame,
            tier: str,
            probability_col: str = 'churn_probability'
    ) -> pd.DataFrame:
        """
        Get all partners in a specific risk tier.

        Args:
            df: DataFrame with partner data
            tier: Risk tier to filter
            probability_col: Column name for churn probability

        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        df['risk_tier'] = self.assign_tiers(df[probability_col].values)
        return df[df['risk_tier'] == tier].sort_values(probability_col, ascending=False)

    def get_tier_distribution(
            self,
            probabilities: np.ndarray
    ) -> Dict[str, int]:
        """
        Get count of partners in each tier.

        Args:
            probabilities: Array of churn probabilities

        Returns:
            Dict mapping tier name to count
        """
        tiers = self.assign_tiers(probabilities)
        distribution = {tier: 0 for tier in self.tier_order}
        for tier in tiers:
            distribution[tier] += 1
        return distribution

    def get_tier_transitions(
            self,
            current_probs: np.ndarray,
            previous_probs: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze tier transitions between two time periods.

        Args:
            current_probs: Current churn probabilities
            previous_probs: Previous churn probabilities

        Returns:
            DataFrame with transition analysis
        """
        current_tiers = self.assign_tiers(current_probs)
        previous_tiers = self.assign_tiers(previous_probs)

        transitions = []
        for i, (prev, curr) in enumerate(zip(previous_tiers, current_tiers)):
            prev_order = self.tier_order.index(prev)
            curr_order = self.tier_order.index(curr)

            if curr_order < prev_order:
                direction = 'worsened'
            elif curr_order > prev_order:
                direction = 'improved'
            else:
                direction = 'stable'

            transitions.append({
                'index': i,
                'previous_tier': prev,
                'current_tier': curr,
                'direction': direction,
                'probability_change': current_probs[i] - previous_probs[i]
            })

        return pd.DataFrame(transitions)


if __name__ == "__main__":
    # Test risk tier assignment
    assigner = RiskTierAssigner()

    # Sample probabilities
    probs = np.array([0.92, 0.75, 0.55, 0.35, 0.15, 0.05])

    print("Risk Tier Assignment Test")
    print("=" * 50)

    for p in probs:
        tier = assigner.assign_tier(p)
        info = assigner.get_tier_info(tier)
        print(f"\nProbability: {p:.0%}")
        print(f"  Tier: {info['name']}")
        print(f"  Color: {info['color']}")
        print(f"  Description: {info['description']}")
        print(f"  Action: {info['action']}")

    # Test distribution
    print("\n" + "=" * 50)
    print("Tier Distribution:")
    distribution = assigner.get_tier_distribution(probs)
    for tier, count in distribution.items():
        print(f"  {tier}: {count}")
