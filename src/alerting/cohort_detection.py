"""
Cohort detection for identifying systemic churn patterns.
Detects when multiple partners with common characteristics are at risk.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


@dataclass
class CohortAlert:
    """Represents a detected cohort pattern."""
    cohort_id: str
    dimension: str
    value: str
    size: int
    avg_churn_probability: float
    baseline_probability: float
    lift: float  # How much higher than baseline
    severity: str
    affected_partners: List[str]
    description: str


class CohortDetector:
    """
    Detect cohorts of at-risk partners with common characteristics.
    Identifies systemic issues vs. individual partner problems.
    """

    def __init__(
            self,
            min_cohort_size: int = None,
            lift_threshold: float = 1.5
    ):
        """
        Initialize cohort detector.

        Args:
            min_cohort_size: Minimum partners for cohort detection
            lift_threshold: Minimum lift to flag as significant
        """
        config = Config.COHORT_CONFIG
        self.min_cohort_size = min_cohort_size or config['min_cohort_size']
        self.grouping_dimensions = config['grouping_dimensions']
        self.lift_threshold = lift_threshold

    def detect_cohorts(
            self,
            df: pd.DataFrame,
            probability_col: str = 'churn_probability',
            threshold: float = 0.5
    ) -> List[CohortAlert]:
        """
        Detect at-risk cohorts across all dimensions.

        Args:
            df: DataFrame with partner data and probabilities
            probability_col: Column with churn probability
            threshold: Probability threshold for "at risk"

        Returns:
            List of CohortAlert objects
        """
        alerts = []
        baseline_prob = df[probability_col].mean()

        for dimension in self.grouping_dimensions:
            if dimension not in df.columns:
                continue

            dimension_alerts = self._analyze_dimension(
                df, dimension, probability_col, baseline_prob, threshold
            )
            alerts.extend(dimension_alerts)

        # Also check for multi-dimensional cohorts
        multi_alerts = self._detect_multi_dimensional_cohorts(
            df, probability_col, baseline_prob, threshold
        )
        alerts.extend(multi_alerts)

        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

        return alerts

    def _analyze_dimension(
            self,
            df: pd.DataFrame,
            dimension: str,
            probability_col: str,
            baseline_prob: float,
            threshold: float
    ) -> List[CohortAlert]:
        """Analyze a single dimension for cohort patterns."""
        alerts = []

        # Group by dimension
        grouped = df.groupby(dimension).agg({
            probability_col: ['mean', 'count'],
            'partner_id': list
        }).reset_index()

        grouped.columns = [dimension, 'avg_prob', 'size', 'partner_ids']

        # Filter to significant cohorts
        for _, row in grouped.iterrows():
            if row['size'] < self.min_cohort_size:
                continue

            lift = row['avg_prob'] / baseline_prob if baseline_prob > 0 else 1

            if lift >= self.lift_threshold:
                severity = self._calculate_severity(lift, row['size'], row['avg_prob'])

                at_risk = df[(df[dimension] == row[dimension]) &
                             (df[probability_col] >= threshold)]

                alert = CohortAlert(
                    cohort_id=f"{dimension}_{row[dimension]}",
                    dimension=dimension,
                    value=str(row[dimension]),
                    size=int(row['size']),
                    avg_churn_probability=float(row['avg_prob']),
                    baseline_probability=float(baseline_prob),
                    lift=float(lift),
                    severity=severity,
                    affected_partners=at_risk['partner_id'].tolist()[:20],  # Limit list size
                    description=self._generate_description(dimension, row, lift, baseline_prob)
                )
                alerts.append(alert)

        return alerts

    def _detect_multi_dimensional_cohorts(
            self,
            df: pd.DataFrame,
            probability_col: str,
            baseline_prob: float,
            threshold: float
    ) -> List[CohortAlert]:
        """Detect cohorts defined by multiple dimensions."""
        alerts = []

        # Check common combinations
        dimension_pairs = [
            ('region', 'tier'),
            ('region', 'partner_type'),
            ('tier', 'partner_type')
        ]

        for dim1, dim2 in dimension_pairs:
            if dim1 not in df.columns or dim2 not in df.columns:
                continue

            grouped = df.groupby([dim1, dim2]).agg({
                probability_col: ['mean', 'count'],
                'partner_id': list
            }).reset_index()

            grouped.columns = [dim1, dim2, 'avg_prob', 'size', 'partner_ids']

            for _, row in grouped.iterrows():
                if row['size'] < self.min_cohort_size:
                    continue

                lift = row['avg_prob'] / baseline_prob if baseline_prob > 0 else 1

                if lift >= self.lift_threshold * 1.2:  # Higher threshold for combos
                    severity = self._calculate_severity(lift, row['size'], row['avg_prob'])

                    at_risk = df[(df[dim1] == row[dim1]) &
                                 (df[dim2] == row[dim2]) &
                                 (df[probability_col] >= threshold)]

                    alert = CohortAlert(
                        cohort_id=f"{dim1}_{row[dim1]}_{dim2}_{row[dim2]}",
                        dimension=f"{dim1} + {dim2}",
                        value=f"{row[dim1]} / {row[dim2]}",
                        size=int(row['size']),
                        avg_churn_probability=float(row['avg_prob']),
                        baseline_probability=float(baseline_prob),
                        lift=float(lift),
                        severity=severity,
                        affected_partners=at_risk['partner_id'].tolist()[:20],
                        description=f"Partners in {row[dim1]} {dim1} with {row[dim2]} tier "
                                    f"show {lift:.1f}x higher churn risk"
                    )
                    alerts.append(alert)

        return alerts

    def _calculate_severity(self, lift: float, size: int, avg_prob: float) -> str:
        """Calculate alert severity based on lift, size, and probability."""
        # Score based on multiple factors
        lift_score = min(lift / 2, 3)  # Cap at 3
        size_score = min(size / 10, 2)  # Cap at 2
        prob_score = avg_prob * 2

        total = lift_score + size_score + prob_score

        if total >= 5 or avg_prob >= 0.7:
            return 'critical'
        elif total >= 3.5 or avg_prob >= 0.5:
            return 'high'
        elif total >= 2:
            return 'medium'
        return 'low'

    def _generate_description(
            self,
            dimension: str,
            row: pd.Series,
            lift: float,
            baseline: float
    ) -> str:
        """Generate human-readable description of cohort pattern."""
        dimension_labels = {
            'region': 'region',
            'tier': 'tier level',
            'partner_type': 'partner type',
            'tenure_bucket': 'tenure group'
        }

        dim_label = dimension_labels.get(dimension, dimension)

        return (
            f"Partners in the '{row[dimension]}' {dim_label} show {lift:.1f}x higher "
            f"churn risk than baseline ({row['avg_prob']:.1%} vs {baseline:.1%}). "
            f"This affects {row['size']} partners."
        )

    def get_cohort_summary(self, alerts: List[CohortAlert]) -> pd.DataFrame:
        """
        Create summary DataFrame of detected cohorts.

        Args:
            alerts: List of CohortAlert objects

        Returns:
            DataFrame with cohort summary
        """
        records = []
        for alert in alerts:
            records.append({
                'cohort_id': alert.cohort_id,
                'dimension': alert.dimension,
                'value': alert.value,
                'size': alert.size,
                'avg_probability': alert.avg_churn_probability,
                'lift': alert.lift,
                'severity': alert.severity,
                'affected_count': len(alert.affected_partners)
            })

        return pd.DataFrame(records)

    def identify_systemic_issues(
            self,
            alerts: List[CohortAlert]
    ) -> Dict:
        """
        Analyze cohort alerts to identify systemic issues.

        Args:
            alerts: List of detected cohort alerts

        Returns:
            Dict with systemic issue analysis
        """
        if not alerts:
            return {'has_systemic_issues': False, 'issues': []}

        # Group by dimension to find patterns
        dimension_counts = {}
        dimension_severity = {}

        for alert in alerts:
            dim = alert.dimension
            if dim not in dimension_counts:
                dimension_counts[dim] = 0
                dimension_severity[dim] = []
            dimension_counts[dim] += 1
            dimension_severity[dim].append(alert.severity)

        # Identify systemic issues
        issues = []
        for dim, count in dimension_counts.items():
            severities = dimension_severity[dim]
            critical_count = severities.count('critical') + severities.count('high')

            if count >= 2 and critical_count > 0:
                issues.append({
                    'dimension': dim,
                    'cohort_count': count,
                    'critical_count': critical_count,
                    'recommendation': self._get_systemic_recommendation(dim)
                })

        return {
            'has_systemic_issues': len(issues) > 0,
            'issues': issues,
            'total_cohorts_detected': len(alerts),
            'critical_cohorts': sum(1 for a in alerts if a.severity in ['critical', 'high'])
        }

    def _get_systemic_recommendation(self, dimension: str) -> str:
        """Get recommendation for systemic issue by dimension."""
        recommendations = {
            'region': 'Consider regional review of competitive landscape and local support quality.',
            'tier': 'Review tier-specific value proposition and benefits alignment.',
            'partner_type': 'Evaluate program structure and support for this partner type.',
            'tenure_bucket': 'Assess onboarding or long-term engagement processes.'
        }
        return recommendations.get(dimension, 'Conduct root cause analysis for this cohort.')


if __name__ == "__main__":
    # Test cohort detection
    np.random.seed(42)

    # Create sample data with some cohort patterns
    n = 500
    regions = ['asia_pacific', 'europe', 'latin_america', 'africa', 'middle_east']
    tiers = ['platinum', 'gold', 'silver', 'bronze']

    df = pd.DataFrame({
        'partner_id': [f'P{i:05d}' for i in range(n)],
        'region': np.random.choice(regions, n),
        'tier': np.random.choice(tiers, n),
        'partner_type': np.random.choice(['affiliate', 'ib', 'master_affiliate'], n),
        'churn_probability': np.random.beta(2, 5, n)
    })

    # Inject cohort pattern: asia_pacific bronze partners have higher churn
    mask = (df['region'] == 'asia_pacific') & (df['tier'] == 'bronze')
    df.loc[mask, 'churn_probability'] = np.clip(
        df.loc[mask, 'churn_probability'] + 0.35, 0, 1
    )

    # Another pattern: europe IBs have higher churn
    mask2 = (df['region'] == 'europe') & (df['partner_type'] == 'ib')
    df.loc[mask2, 'churn_probability'] = np.clip(
        df.loc[mask2, 'churn_probability'] + 0.25, 0, 1
    )

    # Detect cohorts
    detector = CohortDetector()
    alerts = detector.detect_cohorts(df)

    print("Cohort Detection Test")
    print("=" * 60)
    print(f"\nDetected {len(alerts)} cohort patterns\n")

    for alert in alerts[:5]:
        print(f"Cohort: {alert.cohort_id}")
        print(f"  Severity: {alert.severity}")
        print(f"  Size: {alert.size} partners")
        print(f"  Avg Probability: {alert.avg_churn_probability:.1%}")
        print(f"  Lift: {alert.lift:.2f}x")
        print(f"  Description: {alert.description}")
        print()

    # Systemic analysis
    systemic = detector.identify_systemic_issues(alerts)
    print("\nSystemic Issue Analysis:")
    print(f"  Has systemic issues: {systemic['has_systemic_issues']}")
    if systemic['issues']:
        for issue in systemic['issues']:
            print(f"  - {issue['dimension']}: {issue['cohort_count']} cohorts, "
                  f"{issue['critical_count']} critical")
            print(f"    Recommendation: {issue['recommendation']}")
