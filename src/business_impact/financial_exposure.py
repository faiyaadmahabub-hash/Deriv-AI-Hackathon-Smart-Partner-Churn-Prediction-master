"""
Financial exposure calculator for partner churn risk.
Quantifies the monetary impact of potential churn.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class FinancialExposureCalculator:
    """
    Calculate financial exposure from partner churn risk.
    Provides executive-level financial impact metrics.
    """

    def __init__(self):
        """Initialize with configuration."""
        self.tier_multipliers = Config.TIER_REVENUE_MULTIPLIERS
        self.avg_commission = Config.AVG_MONTHLY_COMMISSION

    def calculate_partner_exposure(
            self,
            monthly_commission: float,
            churn_probability: float,
            tier: str,
            tenure_months: int = 12
    ) -> Dict:
        """
        Calculate financial exposure for a single partner.

        Args:
            monthly_commission: Current monthly commission
            churn_probability: Predicted churn probability
            tier: Partner tier
            tenure_months: Partner tenure in months

        Returns:
            Dict with exposure metrics
        """
        # Annual value
        annual_value = monthly_commission * 12

        # Expected loss (probability weighted)
        expected_annual_loss = annual_value * churn_probability

        # Lifetime value remaining (assuming 3-year average lifetime)
        avg_remaining_months = max(0, 36 - tenure_months)
        lifetime_value_remaining = monthly_commission * avg_remaining_months

        # Risk-adjusted lifetime value
        risk_adjusted_ltv = lifetime_value_remaining * (1 - churn_probability)

        # Value at risk (what we could lose)
        value_at_risk = lifetime_value_remaining * churn_probability

        return {
            'monthly_value': monthly_commission,
            'annual_value': annual_value,
            'expected_annual_loss': expected_annual_loss,
            'lifetime_value_remaining': lifetime_value_remaining,
            'risk_adjusted_ltv': risk_adjusted_ltv,
            'value_at_risk': value_at_risk,
            'tier_weight': self.tier_multipliers.get(tier, 1.0)
        }

    def calculate_portfolio_exposure(
            self,
            df: pd.DataFrame,
            probability_col: str = 'churn_probability'
    ) -> Dict:
        """
        Calculate total financial exposure across portfolio.

        Args:
            df: DataFrame with partner data
            probability_col: Column with churn probability

        Returns:
            Dict with portfolio-level exposure metrics
        """
        results = {
            'total_partners': len(df),
            'at_risk_partners': 0,
            'total_monthly_value': 0,
            'total_annual_value': 0,
            'expected_annual_loss': 0,
            'total_value_at_risk': 0,
            'exposure_by_tier': {},
            'exposure_by_risk_level': {}
        }

        # Calculate per-partner exposure
        commission_col = 'commission_current_month'
        if commission_col not in df.columns:
            commission_col = 'commission_3month_avg' if 'commission_3month_avg' in df.columns else None

        for _, row in df.iterrows():
            prob = row[probability_col]
            commission = row.get(commission_col, self.avg_commission.get(row.get('tier', 'bronze'), 3000))
            tier = row.get('tier', 'bronze')
            tenure = row.get('tenure_months', 12)

            exposure = self.calculate_partner_exposure(commission, prob, tier, tenure)

            results['total_monthly_value'] += exposure['monthly_value']
            results['total_annual_value'] += exposure['annual_value']
            results['expected_annual_loss'] += exposure['expected_annual_loss']
            results['total_value_at_risk'] += exposure['value_at_risk']

            if prob >= 0.5:
                results['at_risk_partners'] += 1

            # Aggregate by tier
            if tier not in results['exposure_by_tier']:
                results['exposure_by_tier'][tier] = {
                    'count': 0, 'value_at_risk': 0, 'expected_loss': 0
                }
            results['exposure_by_tier'][tier]['count'] += 1
            results['exposure_by_tier'][tier]['value_at_risk'] += exposure['value_at_risk']
            results['exposure_by_tier'][tier]['expected_loss'] += exposure['expected_annual_loss']

            # Aggregate by risk level
            if prob >= 0.7:
                risk_level = 'high'
            elif prob >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            if risk_level not in results['exposure_by_risk_level']:
                results['exposure_by_risk_level'][risk_level] = {
                    'count': 0, 'value_at_risk': 0
                }
            results['exposure_by_risk_level'][risk_level]['count'] += 1
            results['exposure_by_risk_level'][risk_level]['value_at_risk'] += exposure['value_at_risk']

        return results

    def get_exposure_summary(
            self,
            df: pd.DataFrame,
            probability_col: str = 'churn_probability'
    ) -> pd.DataFrame:
        """
        Get detailed exposure summary per partner.

        Args:
            df: DataFrame with partner data
            probability_col: Column with churn probability

        Returns:
            DataFrame with exposure details
        """
        records = []

        commission_col = 'commission_current_month'
        if commission_col not in df.columns:
            commission_col = 'commission_3month_avg' if 'commission_3month_avg' in df.columns else None

        for _, row in df.iterrows():
            prob = row[probability_col]
            commission = row.get(commission_col, 3000)
            tier = row.get('tier', 'bronze')

            exposure = self.calculate_partner_exposure(
                commission, prob, tier, row.get('tenure_months', 12)
            )

            records.append({
                'partner_id': row.get('partner_id', 'Unknown'),
                'tier': tier,
                'churn_probability': prob,
                'monthly_value': exposure['monthly_value'],
                'annual_value': exposure['annual_value'],
                'expected_loss': exposure['expected_annual_loss'],
                'value_at_risk': exposure['value_at_risk']
            })

        result = pd.DataFrame(records)
        return result.sort_values('value_at_risk', ascending=False)

    def format_currency(self, value: float) -> str:
        """Format value as currency string."""
        if value >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"${value / 1_000:.0f}K"
        return f"${value:.0f}"


if __name__ == "__main__":
    # Test financial exposure calculator
    calculator = FinancialExposureCalculator()

    print("Financial Exposure Calculator Test")
    print("=" * 60)

    # Single partner test
    exposure = calculator.calculate_partner_exposure(
        monthly_commission=25000,
        churn_probability=0.75,
        tier='gold',
        tenure_months=18
    )

    print("\nSingle Partner Exposure (Gold, $25K/month, 75% churn risk):")
    for key, value in exposure.items():
        if isinstance(value, float):
            print(f"  {key}: {calculator.format_currency(value)}")
        else:
            print(f"  {key}: {value}")

    # Portfolio test
    np.random.seed(42)
    df = pd.DataFrame({
        'partner_id': [f'P{i:04d}' for i in range(100)],
        'tier': np.random.choice(['platinum', 'gold', 'silver', 'bronze'], 100,
                                  p=[0.05, 0.15, 0.35, 0.45]),
        'churn_probability': np.random.beta(2, 5, 100),
        'commission_current_month': np.random.gamma(3, 5000, 100),
        'tenure_months': np.random.randint(1, 60, 100)
    })

    portfolio = calculator.calculate_portfolio_exposure(df)

    print("\n\nPortfolio Exposure Summary:")
    print(f"  Total Partners: {portfolio['total_partners']}")
    print(f"  At-Risk Partners (>50% prob): {portfolio['at_risk_partners']}")
    print(f"  Total Annual Value: {calculator.format_currency(portfolio['total_annual_value'])}")
    print(f"  Expected Annual Loss: {calculator.format_currency(portfolio['expected_annual_loss'])}")
    print(f"  Total Value at Risk: {calculator.format_currency(portfolio['total_value_at_risk'])}")

    print("\n  By Tier:")
    for tier, data in portfolio['exposure_by_tier'].items():
        print(f"    {tier}: {data['count']} partners, "
              f"VAR: {calculator.format_currency(data['value_at_risk'])}")
