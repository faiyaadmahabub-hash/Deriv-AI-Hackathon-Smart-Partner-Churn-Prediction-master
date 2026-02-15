"""
Intervention ROI calculator for partner retention.
Calculates expected return on investment for retention interventions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class InterventionROICalculator:
    """
    Calculate ROI for retention interventions.
    Helps prioritize which partners to target and which interventions to use.
    """

    def __init__(self):
        """Initialize with intervention costs and success rates."""
        self.intervention_costs = Config.INTERVENTION_COSTS
        self.base_success_rates = Config.INTERVENTION_SUCCESS_RATES

    def calculate_intervention_roi(
            self,
            partner_value: float,
            churn_probability: float,
            intervention_type: str,
            tier: str = 'silver',
            custom_success_rate: Optional[float] = None
    ) -> Dict:
        """
        Calculate ROI for a specific intervention.

        Args:
            partner_value: Annual partner value
            churn_probability: Current churn probability
            intervention_type: Type of intervention
            tier: Partner tier
            custom_success_rate: Override default success rate

        Returns:
            Dict with ROI metrics
        """
        cost = self.intervention_costs.get(intervention_type, 100)
        base_success = self.base_success_rates.get(intervention_type, 0.2)

        # Adjust success rate by tier (higher tiers respond better)
        tier_adjustments = {
            'platinum': 1.3,
            'gold': 1.15,
            'silver': 1.0,
            'bronze': 0.85
        }
        adjusted_success = base_success * tier_adjustments.get(tier, 1.0)

        if custom_success_rate is not None:
            adjusted_success = custom_success_rate

        # Clamp success rate
        adjusted_success = min(0.9, max(0.1, adjusted_success))

        # Calculate expected value saved
        expected_value_saved = partner_value * churn_probability * adjusted_success

        # Calculate ROI
        net_value = expected_value_saved - cost
        roi_pct = (net_value / cost * 100) if cost > 0 else 0

        # Break-even probability
        break_even_prob = cost / (partner_value * adjusted_success) if partner_value * adjusted_success > 0 else 1

        return {
            'intervention_type': intervention_type,
            'cost': cost,
            'success_rate': adjusted_success,
            'expected_value_saved': expected_value_saved,
            'net_value': net_value,
            'roi_percent': roi_pct,
            'break_even_probability': break_even_prob,
            'is_worthwhile': net_value > 0
        }

    def recommend_intervention(
            self,
            partner_value: float,
            churn_probability: float,
            tier: str = 'silver',
            budget_limit: Optional[float] = None
    ) -> Dict:
        """
        Recommend the best intervention for a partner.

        Args:
            partner_value: Annual partner value
            churn_probability: Current churn probability
            tier: Partner tier
            budget_limit: Maximum intervention budget

        Returns:
            Dict with recommended intervention details
        """
        best_intervention = None
        best_roi = float('-inf')

        for intervention_type in self.intervention_costs.keys():
            cost = self.intervention_costs[intervention_type]

            # Skip if over budget
            if budget_limit and cost > budget_limit:
                continue

            roi_analysis = self.calculate_intervention_roi(
                partner_value, churn_probability, intervention_type, tier
            )

            if roi_analysis['roi_percent'] > best_roi and roi_analysis['is_worthwhile']:
                best_roi = roi_analysis['roi_percent']
                best_intervention = roi_analysis

        if best_intervention is None:
            return {
                'recommendation': 'no_action',
                'reason': 'No intervention provides positive ROI',
                'all_options': self._get_all_options(partner_value, churn_probability, tier)
            }

        return {
            'recommendation': best_intervention['intervention_type'],
            'analysis': best_intervention,
            'alternative_options': self._get_alternatives(
                partner_value, churn_probability, tier, best_intervention['intervention_type']
            )
        }

    def _get_all_options(
            self,
            partner_value: float,
            churn_probability: float,
            tier: str
    ) -> List[Dict]:
        """Get all intervention options with their ROI."""
        options = []
        for intervention_type in self.intervention_costs.keys():
            analysis = self.calculate_intervention_roi(
                partner_value, churn_probability, intervention_type, tier
            )
            options.append(analysis)

        return sorted(options, key=lambda x: x['roi_percent'], reverse=True)

    def _get_alternatives(
            self,
            partner_value: float,
            churn_probability: float,
            tier: str,
            exclude: str
    ) -> List[Dict]:
        """Get alternative intervention options."""
        alternatives = []
        for intervention_type in self.intervention_costs.keys():
            if intervention_type == exclude:
                continue
            analysis = self.calculate_intervention_roi(
                partner_value, churn_probability, intervention_type, tier
            )
            if analysis['is_worthwhile']:
                alternatives.append({
                    'type': intervention_type,
                    'cost': analysis['cost'],
                    'roi': analysis['roi_percent']
                })

        return sorted(alternatives, key=lambda x: x['roi'], reverse=True)[:3]

    def calculate_portfolio_roi(
            self,
            df: pd.DataFrame,
            intervention_type: str = 'account_review',
            probability_col: str = 'churn_probability',
            value_col: Optional[str] = None
    ) -> Dict:
        """
        Calculate portfolio-wide ROI for an intervention type.

        Args:
            df: DataFrame with partner data
            intervention_type: Type of intervention to apply
            probability_col: Column with churn probability
            value_col: Column with partner annual value

        Returns:
            Dict with portfolio ROI metrics
        """
        total_cost = 0
        total_expected_saved = 0
        partners_treated = 0

        for _, row in df.iterrows():
            prob = row[probability_col]

            # Only intervene if probability > threshold
            if prob < 0.3:
                continue

            partner_value = row.get(value_col, row.get('commission_current_month', 5000) * 12)
            tier = row.get('tier', 'silver')

            roi_analysis = self.calculate_intervention_roi(
                partner_value, prob, intervention_type, tier
            )

            if roi_analysis['is_worthwhile']:
                total_cost += roi_analysis['cost']
                total_expected_saved += roi_analysis['expected_value_saved']
                partners_treated += 1

        net_value = total_expected_saved - total_cost
        portfolio_roi = (net_value / total_cost * 100) if total_cost > 0 else 0

        return {
            'intervention_type': intervention_type,
            'partners_treated': partners_treated,
            'total_cost': total_cost,
            'expected_value_saved': total_expected_saved,
            'net_value': net_value,
            'portfolio_roi_percent': portfolio_roi,
            'cost_per_partner': total_cost / partners_treated if partners_treated > 0 else 0,
            'value_saved_per_partner': total_expected_saved / partners_treated if partners_treated > 0 else 0
        }

    def compare_interventions(
            self,
            df: pd.DataFrame,
            probability_col: str = 'churn_probability'
    ) -> pd.DataFrame:
        """
        Compare all intervention types across portfolio.

        Args:
            df: DataFrame with partner data
            probability_col: Column with churn probability

        Returns:
            DataFrame comparing intervention types
        """
        comparisons = []

        for intervention_type in self.intervention_costs.keys():
            portfolio_roi = self.calculate_portfolio_roi(
                df, intervention_type, probability_col
            )
            comparisons.append(portfolio_roi)

        return pd.DataFrame(comparisons).sort_values('portfolio_roi_percent', ascending=False)


if __name__ == "__main__":
    # Test intervention ROI calculator
    calculator = InterventionROICalculator()

    print("Intervention ROI Calculator Test")
    print("=" * 60)

    # Single partner analysis
    print("\nSingle Partner Analysis:")
    print("Partner: $60,000 annual value, 65% churn probability, Gold tier")

    for intervention in ['personal_call', 'account_review', 'custom_incentive', 'retention_package']:
        roi = calculator.calculate_intervention_roi(
            partner_value=60000,
            churn_probability=0.65,
            intervention_type=intervention,
            tier='gold'
        )
        print(f"\n  {intervention}:")
        print(f"    Cost: ${roi['cost']}")
        print(f"    Success Rate: {roi['success_rate']:.0%}")
        print(f"    Expected Value Saved: ${roi['expected_value_saved']:,.0f}")
        print(f"    ROI: {roi['roi_percent']:.0f}%")
        print(f"    Worthwhile: {roi['is_worthwhile']}")

    # Recommendation
    print("\n" + "=" * 60)
    print("\nRecommended Intervention:")
    recommendation = calculator.recommend_intervention(
        partner_value=60000,
        churn_probability=0.65,
        tier='gold'
    )
    print(f"  Best Option: {recommendation['recommendation']}")
    if 'analysis' in recommendation:
        print(f"  Expected ROI: {recommendation['analysis']['roi_percent']:.0f}%")

    # Portfolio comparison
    print("\n" + "=" * 60)
    print("\nPortfolio Analysis:")

    np.random.seed(42)
    df = pd.DataFrame({
        'partner_id': [f'P{i:04d}' for i in range(100)],
        'tier': np.random.choice(['platinum', 'gold', 'silver', 'bronze'], 100),
        'churn_probability': np.random.beta(2, 4, 100),
        'commission_current_month': np.random.gamma(3, 5000, 100)
    })

    comparison = calculator.compare_interventions(df)
    print("\nIntervention Comparison:")
    print(comparison[['intervention_type', 'partners_treated', 'total_cost',
                       'net_value', 'portfolio_roi_percent']].to_string(index=False))
