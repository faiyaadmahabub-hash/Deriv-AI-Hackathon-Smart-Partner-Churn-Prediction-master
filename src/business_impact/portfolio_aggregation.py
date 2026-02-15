"""
Portfolio aggregation for high-risk partner exposure.
Provides executive-level portfolio view and tier breakdowns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class PortfolioAggregator:
    """
    Aggregate exposure across all high-risk partners.
    Provides executive-level portfolio metrics and prioritization.
    """

    def __init__(self):
        """Initialize portfolio aggregator."""
        # Extract min_prob thresholds from RISK_TIERS
        self.risk_thresholds = {
            tier: info['min_prob'] for tier, info in Config.RISK_TIERS.items()
        }

    def aggregate_exposure(
            self,
            partner_exposures: List[Dict]
    ) -> Dict:
        """
        Aggregate exposure across all partners.

        Args:
            partner_exposures: List of dicts from FinancialExposureCalculator

        Returns:
            Portfolio-level metrics
        """
        if not partner_exposures:
            return self._empty_portfolio()

        df = pd.DataFrame(partner_exposures)

        # Total exposure metrics
        total_exposure = df['value_at_risk'].sum() if 'value_at_risk' in df.columns else 0
        total_monthly_risk = df['monthly_value'].sum() if 'monthly_value' in df.columns else 0
        total_annual_risk = df['annual_value'].sum() if 'annual_value' in df.columns else 0
        total_expected_loss = df['expected_loss'].sum() if 'expected_loss' in df.columns else 0

        # Tier breakdown
        tier_breakdown = {}
        if 'tier' in df.columns:
            tier_breakdown = df.groupby('tier').agg({
                'partner_id': 'count',
                'value_at_risk': 'sum',
                'monthly_value': 'sum',
                'expected_loss': 'sum'
            }).rename(columns={'partner_id': 'partner_count'}).to_dict('index')

        # Top 10 partners by exposure
        top_10 = df.nlargest(10, 'value_at_risk')[
            ['partner_id', 'tier', 'value_at_risk', 'churn_probability', 'monthly_value']
        ].to_dict('records') if len(df) > 0 else []

        # Risk distribution
        risk_distribution = self._calculate_risk_distribution(df)

        # Region breakdown if available
        region_breakdown = {}
        if 'region' in df.columns:
            region_breakdown = df.groupby('region').agg({
                'partner_id': 'count',
                'value_at_risk': 'sum'
            }).rename(columns={'partner_id': 'partner_count'}).to_dict('index')

        return {
            'total_partners_at_risk': len(df),
            'total_value_at_risk': round(total_exposure, 2),
            'total_monthly_risk': round(total_monthly_risk, 2),
            'total_annual_value': round(total_annual_risk, 2),
            'total_expected_loss': round(total_expected_loss, 2),
            'avg_exposure_per_partner': round(total_exposure / len(df), 2) if len(df) > 0 else 0,
            'tier_breakdown': tier_breakdown,
            'region_breakdown': region_breakdown,
            'top_10_partners': top_10,
            'risk_distribution': risk_distribution
        }

    def _calculate_risk_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution across risk levels."""
        if 'churn_probability' not in df.columns:
            return {'high': 0, 'medium': 0, 'low': 0}

        high_threshold = self.risk_thresholds.get('high', 0.7)
        medium_threshold = self.risk_thresholds.get('medium', 0.4)

        return {
            'high': int((df['churn_probability'] >= high_threshold).sum()),
            'medium': int(((df['churn_probability'] >= medium_threshold) & 
                          (df['churn_probability'] < high_threshold)).sum()),
            'low': int((df['churn_probability'] < medium_threshold).sum()),
            'total': len(df)
        }

    def _empty_portfolio(self) -> Dict:
        """Return empty portfolio structure."""
        return {
            'total_partners_at_risk': 0,
            'total_value_at_risk': 0,
            'total_monthly_risk': 0,
            'total_annual_value': 0,
            'total_expected_loss': 0,
            'avg_exposure_per_partner': 0,
            'tier_breakdown': {},
            'region_breakdown': {},
            'top_10_partners': [],
            'risk_distribution': {'high': 0, 'medium': 0, 'low': 0, 'total': 0}
        }

    def get_prioritized_partners(
            self,
            partner_exposures: List[Dict],
            limit: int = 20
    ) -> List[Dict]:
        """
        Get top partners prioritized by exposure and intervention feasibility.

        Args:
            partner_exposures: List of partner exposure data
            limit: Maximum partners to return

        Returns:
            Prioritized list of partners
        """
        if not partner_exposures:
            return []

        df = pd.DataFrame(partner_exposures)

        # Priority score: value_at_risk * probability (highest risk dollar impact)
        df['priority_score'] = df['value_at_risk'] * df.get('churn_probability', 0.5)

        # Sort by priority score
        prioritized = df.nlargest(limit, 'priority_score')

        return prioritized.to_dict('records')

    def get_tier_summary(
            self,
            partner_exposures: List[Dict]
    ) -> pd.DataFrame:
        """
        Get summary statistics by tier.

        Args:
            partner_exposures: List of partner exposure data

        Returns:
            DataFrame with tier-level summary
        """
        if not partner_exposures:
            return pd.DataFrame()

        df = pd.DataFrame(partner_exposures)

        if 'tier' not in df.columns:
            return pd.DataFrame()

        summary = df.groupby('tier').agg({
            'partner_id': 'count',
            'value_at_risk': ['sum', 'mean'],
            'churn_probability': 'mean',
            'monthly_value': 'sum'
        })

        summary.columns = ['partner_count', 'total_var', 'avg_var', 
                          'avg_churn_prob', 'total_monthly_value']

        # Order by tier importance
        tier_order = ['platinum', 'gold', 'silver', 'bronze']
        summary = summary.reindex([t for t in tier_order if t in summary.index])

        return summary.reset_index()


if __name__ == "__main__":
    # Test portfolio aggregator
    aggregator = PortfolioAggregator()

    print("Portfolio Aggregator Test")
    print("=" * 60)

    # Create sample partner exposures
    np.random.seed(42)
    sample_exposures = []
    for i in range(50):
        tier = np.random.choice(['platinum', 'gold', 'silver', 'bronze'],
                                p=[0.05, 0.15, 0.35, 0.45])
        prob = np.random.beta(2, 5)
        monthly = np.random.gamma(3, 5000)

        sample_exposures.append({
            'partner_id': f'P{i:04d}',
            'tier': tier,
            'region': np.random.choice(['SEA', 'MENA', 'Europe', 'Africa']),
            'churn_probability': prob,
            'monthly_value': monthly,
            'annual_value': monthly * 12,
            'value_at_risk': monthly * 12 * prob,
            'expected_loss': monthly * 12 * prob * 0.8
        })

    # Test aggregation
    portfolio = aggregator.aggregate_exposure(sample_exposures)

    print(f"\nTotal Partners: {portfolio['total_partners_at_risk']}")
    print(f"Total Value at Risk: ${portfolio['total_value_at_risk']:,.0f}")
    print(f"Average per Partner: ${portfolio['avg_exposure_per_partner']:,.0f}")

    print("\nRisk Distribution:")
    for level, count in portfolio['risk_distribution'].items():
        print(f"  {level}: {count}")

    print("\nTop 5 Partners:")
    for p in portfolio['top_10_partners'][:5]:
        print(f"  {p['partner_id']} ({p['tier']}): ${p['value_at_risk']:,.0f}")

    # Test tier summary
    tier_summary = aggregator.get_tier_summary(sample_exposures)
    print("\nTier Summary:")
    print(tier_summary.to_string(index=False))
