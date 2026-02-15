"""
Intervention window calculator for optimal timing of retention actions.
Determines the best window for intervention based on churn trajectory.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


@dataclass
class InterventionWindow:
    """Represents an optimal intervention window."""
    partner_id: str
    current_probability: float
    predicted_churn_date: Optional[datetime]
    days_until_churn: int
    intervention_deadline: datetime
    urgency: str
    recommended_action: str
    confidence: str


class InterventionWindowCalculator:
    """
    Calculate optimal intervention windows for at-risk partners.
    Uses churn probability trajectory to estimate timing.
    """

    def __init__(self):
        """Initialize calculator with default parameters."""
        self.cusum_alert_days = Config.CUSUM_CONFIG['alert_window_days']

    def calculate_window(
            self,
            partner_id: str,
            current_probability: float,
            probability_trend: float = 0,
            cusum_alert: bool = False,
            last_intervention_days: Optional[int] = None
    ) -> InterventionWindow:
        """
        Calculate intervention window for a single partner.

        Args:
            partner_id: Partner identifier
            current_probability: Current churn probability
            probability_trend: Daily probability change rate
            cusum_alert: Whether CUSUM has triggered
            last_intervention_days: Days since last intervention

        Returns:
            InterventionWindow object
        """
        # Estimate days until churn based on probability level
        if current_probability >= 0.8:
            estimated_days = 14
        elif current_probability >= 0.6:
            estimated_days = 30
        elif current_probability >= 0.4:
            estimated_days = 60
        else:
            estimated_days = 90

        # Adjust for trend
        if probability_trend > 0.01:  # Worsening
            estimated_days = int(estimated_days * 0.7)
        elif probability_trend < -0.01:  # Improving
            estimated_days = int(estimated_days * 1.3)

        # CUSUM alerts suggest earlier action needed
        if cusum_alert:
            estimated_days = min(estimated_days, self.cusum_alert_days)

        # Calculate dates
        today = datetime.now()
        predicted_churn = today + timedelta(days=estimated_days)
        intervention_deadline = today + timedelta(days=max(1, estimated_days - 7))

        # Determine urgency
        urgency = self._calculate_urgency(
            estimated_days, current_probability, cusum_alert
        )

        # Get recommended action based on urgency
        action = self._get_recommended_action(urgency, last_intervention_days)

        # Confidence level
        confidence = self._calculate_confidence(current_probability, cusum_alert)

        return InterventionWindow(
            partner_id=partner_id,
            current_probability=current_probability,
            predicted_churn_date=predicted_churn,
            days_until_churn=estimated_days,
            intervention_deadline=intervention_deadline,
            urgency=urgency,
            recommended_action=action,
            confidence=confidence
        )

    def _calculate_urgency(
            self,
            days_until_churn: int,
            probability: float,
            cusum_alert: bool
    ) -> str:
        """Calculate urgency level."""
        if days_until_churn <= 7 or probability >= 0.85:
            return 'immediate'
        elif days_until_churn <= 14 or probability >= 0.7 or cusum_alert:
            return 'urgent'
        elif days_until_churn <= 30 or probability >= 0.5:
            return 'high'
        elif days_until_churn <= 60:
            return 'moderate'
        return 'standard'

    def _get_recommended_action(
            self,
            urgency: str,
            last_intervention_days: Optional[int]
    ) -> str:
        """Get recommended action based on urgency."""
        # If recent intervention, adjust recommendation
        if last_intervention_days and last_intervention_days < 14:
            return "Follow up on previous intervention"

        actions = {
            'immediate': 'Emergency call from senior RM or executive',
            'urgent': 'Personal call within 24-48 hours',
            'high': 'Schedule meeting within this week',
            'moderate': 'Send personalized outreach email',
            'standard': 'Include in regular check-in cycle'
        }
        return actions.get(urgency, 'Review case')

    def _calculate_confidence(
            self,
            probability: float,
            cusum_alert: bool
    ) -> str:
        """Calculate confidence in the prediction."""
        if cusum_alert and probability >= 0.6:
            return 'high'
        elif probability >= 0.7:
            return 'high'
        elif probability >= 0.5:
            return 'medium'
        return 'low'

    def calculate_batch_windows(
            self,
            df: pd.DataFrame,
            probability_col: str = 'churn_probability',
            trend_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate intervention windows for all partners.

        Args:
            df: DataFrame with partner data
            probability_col: Column with churn probability
            trend_col: Optional column with probability trend

        Returns:
            DataFrame with intervention window details
        """
        windows = []

        for _, row in df.iterrows():
            trend = row.get(trend_col, 0) if trend_col else 0
            cusum = row.get('cusum_alert', False)

            window = self.calculate_window(
                partner_id=row.get('partner_id', 'Unknown'),
                current_probability=row[probability_col],
                probability_trend=trend,
                cusum_alert=cusum
            )

            windows.append({
                'partner_id': window.partner_id,
                'current_probability': window.current_probability,
                'days_until_churn': window.days_until_churn,
                'intervention_deadline': window.intervention_deadline.strftime('%Y-%m-%d'),
                'urgency': window.urgency,
                'recommended_action': window.recommended_action,
                'confidence': window.confidence
            })

        result = pd.DataFrame(windows)

        # Sort by urgency
        urgency_order = {'immediate': 0, 'urgent': 1, 'high': 2, 'moderate': 3, 'standard': 4}
        result['urgency_order'] = result['urgency'].map(urgency_order)
        result = result.sort_values('urgency_order').drop('urgency_order', axis=1)

        return result

    def get_intervention_calendar(
            self,
            windows_df: pd.DataFrame,
            days_ahead: int = 30
    ) -> Dict[str, List[Dict]]:
        """
        Create intervention calendar grouped by week.

        Args:
            windows_df: DataFrame with intervention windows
            days_ahead: Days to look ahead

        Returns:
            Dict with weekly intervention schedule
        """
        today = datetime.now()
        calendar = {}

        for i in range(0, days_ahead, 7):
            week_start = today + timedelta(days=i)
            week_end = week_start + timedelta(days=7)
            week_key = f"Week {i // 7 + 1}: {week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}"

            # Find partners with deadlines in this week
            week_partners = []
            for _, row in windows_df.iterrows():
                deadline = datetime.strptime(row['intervention_deadline'], '%Y-%m-%d')
                if week_start <= deadline < week_end:
                    week_partners.append({
                        'partner_id': row['partner_id'],
                        'urgency': row['urgency'],
                        'action': row['recommended_action'],
                        'probability': row['current_probability']
                    })

            calendar[week_key] = sorted(
                week_partners,
                key=lambda x: {'immediate': 0, 'urgent': 1, 'high': 2}.get(x['urgency'], 3)
            )

        return calendar

    def summarize_workload(
            self,
            windows_df: pd.DataFrame
    ) -> Dict:
        """
        Summarize intervention workload.

        Args:
            windows_df: DataFrame with intervention windows

        Returns:
            Dict with workload summary
        """
        urgency_counts = windows_df['urgency'].value_counts().to_dict()

        # Estimate hours needed
        hours_per_urgency = {
            'immediate': 4,
            'urgent': 2,
            'high': 1.5,
            'moderate': 1,
            'standard': 0.5
        }

        total_hours = sum(
            urgency_counts.get(u, 0) * hours
            for u, hours in hours_per_urgency.items()
        )

        return {
            'total_partners': len(windows_df),
            'urgency_breakdown': urgency_counts,
            'estimated_hours': total_hours,
            'immediate_count': urgency_counts.get('immediate', 0),
            'this_week_count': urgency_counts.get('immediate', 0) + urgency_counts.get('urgent', 0),
            'avg_days_until_churn': windows_df['days_until_churn'].mean()
        }


if __name__ == "__main__":
    # Test intervention window calculator
    calculator = InterventionWindowCalculator()

    print("Intervention Window Calculator Test")
    print("=" * 60)

    # Test individual calculation
    test_cases = [
        {'id': 'P001', 'prob': 0.85, 'trend': 0.02, 'cusum': True},
        {'id': 'P002', 'prob': 0.65, 'trend': 0.00, 'cusum': False},
        {'id': 'P003', 'prob': 0.45, 'trend': -0.01, 'cusum': False},
        {'id': 'P004', 'prob': 0.30, 'trend': 0.01, 'cusum': True},
    ]

    for case in test_cases:
        window = calculator.calculate_window(
            partner_id=case['id'],
            current_probability=case['prob'],
            probability_trend=case['trend'],
            cusum_alert=case['cusum']
        )

        print(f"\nPartner: {window.partner_id}")
        print(f"  Probability: {window.current_probability:.0%}")
        print(f"  Days until churn: {window.days_until_churn}")
        print(f"  Deadline: {window.intervention_deadline.strftime('%Y-%m-%d')}")
        print(f"  Urgency: {window.urgency}")
        print(f"  Action: {window.recommended_action}")
        print(f"  Confidence: {window.confidence}")

    # Test batch processing
    print("\n" + "=" * 60)
    print("\nBatch Processing Test")

    df = pd.DataFrame({
        'partner_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'churn_probability': [0.9, 0.7, 0.5, 0.3, 0.15],
        'cusum_alert': [True, False, True, False, False]
    })

    windows = calculator.calculate_batch_windows(df)
    print(windows[['partner_id', 'days_until_churn', 'urgency', 'recommended_action']])

    # Workload summary
    print("\nWorkload Summary:")
    summary = calculator.summarize_workload(windows)
    for key, value in summary.items():
        print(f"  {key}: {value}")
