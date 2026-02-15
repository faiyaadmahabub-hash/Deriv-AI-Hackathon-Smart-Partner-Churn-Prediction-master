"""
Priority scoring for partner interventions.
Combines churn risk with partner value to prioritize actions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class PriorityScorer:
    """
    Calculate priority scores for partner interventions.
    Balances churn risk with partner value for optimal resource allocation.
    """

    def __init__(
            self,
            risk_weight: float = 0.5,
            value_weight: float = 0.3,
            trend_weight: float = 0.2
    ):
        """
        Initialize priority scorer.

        Args:
            risk_weight: Weight for churn probability
            value_weight: Weight for partner value
            trend_weight: Weight for risk trend
        """
        self.risk_weight = risk_weight
        self.value_weight = value_weight
        self.trend_weight = trend_weight
        self.tier_multipliers = Config.TIER_REVENUE_MULTIPLIERS

    def calculate_priority_score(
            self,
            churn_probability: float,
            monthly_value: float,
            tier: str,
            trend_direction: str = 'stable'
    ) -> float:
        """
        Calculate priority score for a single partner.

        Args:
            churn_probability: Predicted churn probability
            monthly_value: Monthly commission/revenue
            tier: Partner tier
            trend_direction: 'improving', 'stable', or 'worsening'

        Returns:
            Priority score (0-100)
        """
        # Risk component (0-100)
        risk_score = churn_probability * 100

        # Value component (0-100, normalized)
        tier_mult = self.tier_multipliers.get(tier, 1.0)
        value_score = min(100, (monthly_value / 50000) * 100 * tier_mult)

        # Trend component (-20 to +20)
        trend_scores = {
            'worsening': 20,
            'stable': 0,
            'improving': -20
        }
        trend_score = trend_scores.get(trend_direction, 0)

        # Combine scores
        base_score = (
                self.risk_weight * risk_score +
                self.value_weight * value_score
        )

        # Apply trend adjustment
        final_score = min(100, max(0, base_score + self.trend_weight * trend_score))

        return final_score

    def calculate_batch_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate priority scores for all partners in DataFrame.

        Args:
            df: DataFrame with partner data

        Returns:
            DataFrame with added priority_score column
        """
        df = df.copy()

        # Determine trend direction from probability change if available
        if 'probability_change' in df.columns:
            df['trend_direction'] = df['probability_change'].apply(
                lambda x: 'worsening' if x > 0.05 else ('improving' if x < -0.05 else 'stable')
            )
        else:
            df['trend_direction'] = 'stable'

        # Calculate scores
        scores = []
        for _, row in df.iterrows():
            score = self.calculate_priority_score(
                churn_probability=row.get('churn_probability', 0.5),
                monthly_value=row.get('commission_current_month', 5000),
                tier=row.get('tier', 'bronze'),
                trend_direction=row.get('trend_direction', 'stable')
            )
            scores.append(score)

        df['priority_score'] = scores

        return df.sort_values('priority_score', ascending=False)

    def get_priority_tier(self, score: float) -> str:
        """
        Convert priority score to tier.

        Args:
            score: Priority score (0-100)

        Returns:
            Priority tier name
        """
        if score >= 80:
            return 'urgent'
        elif score >= 60:
            return 'high'
        elif score >= 40:
            return 'medium'
        elif score >= 20:
            return 'low'
        return 'minimal'

    def calculate_intervention_capacity(
            self,
            df: pd.DataFrame,
            rm_capacity: int = 20
    ) -> Dict:
        """
        Calculate how many partners can be addressed with current capacity.

        Args:
            df: DataFrame with priority scores
            rm_capacity: Number of partners an RM can handle per week

        Returns:
            Dict with capacity analysis
        """
        if 'priority_score' not in df.columns:
            df = self.calculate_batch_scores(df)

        df_sorted = df.sort_values('priority_score', ascending=False)

        # Partners that can be addressed
        addressable = df_sorted.head(rm_capacity)
        backlog = df_sorted.iloc[rm_capacity:]

        # Calculate value at risk
        total_value = df_sorted['commission_current_month'].sum() if 'commission_current_month' in df_sorted.columns else 0
        addressable_value = addressable['commission_current_month'].sum() if 'commission_current_month' in addressable.columns else 0

        return {
            'total_at_risk': len(df_sorted),
            'addressable': len(addressable),
            'backlog': len(backlog),
            'capacity_utilization': min(100, len(df_sorted) / rm_capacity * 100),
            'total_value_at_risk': total_value * 12,  # Annual
            'addressable_value': addressable_value * 12,
            'coverage_rate': len(addressable) / len(df_sorted) * 100 if len(df_sorted) > 0 else 0,
            'value_coverage_rate': addressable_value / total_value * 100 if total_value > 0 else 0
        }

    def recommend_resource_allocation(
            self,
            df: pd.DataFrame,
            total_rm_hours: int = 40
    ) -> pd.DataFrame:
        """
        Recommend how to allocate RM time across priority tiers.

        Args:
            df: DataFrame with priority scores
            total_rm_hours: Available RM hours per week

        Returns:
            DataFrame with allocation recommendations
        """
        if 'priority_score' not in df.columns:
            df = self.calculate_batch_scores(df)

        df['priority_tier'] = df['priority_score'].apply(self.get_priority_tier)

        # Time per intervention by tier
        time_per_intervention = {
            'urgent': 4,  # hours
            'high': 2,
            'medium': 1,
            'low': 0.5,
            'minimal': 0.25
        }

        summary = df.groupby('priority_tier').agg({
            'priority_score': 'count',
            'commission_current_month': 'sum' if 'commission_current_month' in df.columns else 'count'
        }).rename(columns={'priority_score': 'count', 'commission_current_month': 'value'})

        # Add allocation
        allocations = []
        remaining_hours = total_rm_hours

        for tier in ['urgent', 'high', 'medium', 'low', 'minimal']:
            if tier in summary.index:
                count = summary.loc[tier, 'count']
                hours_needed = count * time_per_intervention[tier]
                allocated_hours = min(hours_needed, remaining_hours)
                partners_addressed = int(allocated_hours / time_per_intervention[tier])
                remaining_hours -= allocated_hours

                allocations.append({
                    'tier': tier,
                    'total_partners': count,
                    'hours_per_partner': time_per_intervention[tier],
                    'hours_allocated': allocated_hours,
                    'partners_addressed': partners_addressed,
                    'coverage': partners_addressed / count * 100 if count > 0 else 0
                })

        return pd.DataFrame(allocations)


if __name__ == "__main__":
    # Test priority scoring
    scorer = PriorityScorer()

    print("Priority Scoring Test")
    print("=" * 50)

    # Sample partners
    test_cases = [
        {'prob': 0.85, 'value': 45000, 'tier': 'gold', 'trend': 'worsening'},
        {'prob': 0.85, 'value': 5000, 'tier': 'bronze', 'trend': 'stable'},
        {'prob': 0.40, 'value': 80000, 'tier': 'platinum', 'trend': 'worsening'},
        {'prob': 0.30, 'value': 10000, 'tier': 'silver', 'trend': 'improving'},
    ]

    print("\nPriority Scores:")
    for case in test_cases:
        score = scorer.calculate_priority_score(
            case['prob'], case['value'], case['tier'], case['trend']
        )
        priority_tier = scorer.get_priority_tier(score)
        print(f"\n  Probability: {case['prob']:.0%}, Value: ${case['value']:,}, "
              f"Tier: {case['tier']}, Trend: {case['trend']}")
        print(f"  > Priority Score: {score:.1f} ({priority_tier})")
