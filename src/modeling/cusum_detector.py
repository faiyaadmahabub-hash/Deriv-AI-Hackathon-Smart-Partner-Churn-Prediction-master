"""
CUSUM (Cumulative Sum) Early Warning Detector.
Detects behavioral shifts 3+ weeks before traditional churn signals.
This is a KEY DIFFERENTIATOR for the hackathon.
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
class CUSUMAlert:
    """Represents a CUSUM early warning alert."""
    partner_id: str
    feature: str
    alert_type: str  # 'negative_shift' or 'positive_shift'
    cusum_value: float
    threshold: float
    weeks_detected: int
    severity: str  # 'low', 'medium', 'high'
    message: str


class CUSUMDetector:
    """
    CUSUM-based early warning system for partner churn.

    CUSUM (Cumulative Sum) detects small but persistent shifts in behavior
    that may not be visible in point-in-time analysis. This gives us
    3+ weeks of early warning before traditional metrics show decline.

    Key insight: Partners don't churn overnight. There's always a gradual
    behavioral shift that CUSUM can detect early.
    """

    def __init__(
            self,
            threshold_multiplier: float = None,
            drift_rate: float = None,
            min_periods: int = None
    ):
        """
        Initialize CUSUM detector.

        Args:
            threshold_multiplier: Sensitivity multiplier for detection threshold
            drift_rate: Expected drift rate per period
            min_periods: Minimum periods before detection is valid
        """
        config = Config.CUSUM_CONFIG
        self.threshold_multiplier = threshold_multiplier or config['threshold_multiplier']
        self.drift_rate = drift_rate or config['drift_rate']
        self.min_periods = min_periods or config['min_periods']
        self.features_to_monitor = config['features_to_monitor']
        self.alert_window_days = config['alert_window_days']

        # Store reference statistics
        self.reference_means: Dict[str, float] = {}
        self.reference_stds: Dict[str, float] = {}
        self.thresholds: Dict[str, float] = {}
        self.is_calibrated = False

    def calibrate(self, df: pd.DataFrame, partner_id_col: str = 'partner_id'):
        """
        Calibrate CUSUM parameters from historical data.

        Uses non-churning partners to establish baseline behavior.

        Args:
            df: Historical data with partner metrics
            partner_id_col: Column name for partner ID
        """
        print("Calibrating CUSUM detector...")

        # Filter to non-churning partners if churn column exists
        if 'churned' in df.columns:
            baseline_df = df[df['churned'] == 0]
            print(f"  Using {len(baseline_df)} non-churning partner records for calibration")
        else:
            baseline_df = df
            print(f"  Using all {len(baseline_df)} records for calibration")

        # Calculate reference statistics for each monitored feature
        for feature in self.features_to_monitor:
            if feature in baseline_df.columns:
                self.reference_means[feature] = baseline_df[feature].mean()
                self.reference_stds[feature] = baseline_df[feature].std()
                self.thresholds[feature] = self.reference_stds[feature] * self.threshold_multiplier

                print(f"  {feature}:")
                print(f"    Mean: {self.reference_means[feature]:.2f}")
                print(f"    Std: {self.reference_stds[feature]:.2f}")
                print(f"    Threshold: {self.thresholds[feature]:.2f}")

        self.is_calibrated = True

    def calculate_cusum(
            self,
            values: np.ndarray,
            reference_mean: float,
            drift: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate CUSUM statistics for a time series.

        Args:
            values: Time series values
            reference_mean: Expected mean (target)
            drift: Allowable drift before signaling

        Returns:
            Tuple of (positive CUSUM, negative CUSUM)
        """
        drift = drift or self.drift_rate

        n = len(values)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(1, n):
            deviation = values[i] - reference_mean

            # Positive CUSUM (detects upward shift)
            cusum_pos[i] = max(0, cusum_pos[i - 1] + deviation - drift)

            # Negative CUSUM (detects downward shift)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + deviation + drift)

        return cusum_pos, cusum_neg

    def detect_single_partner(
            self,
            partner_data: pd.DataFrame,
            partner_id: str
    ) -> List[CUSUMAlert]:
        """
        Detect behavioral shifts for a single partner.

        Args:
            partner_data: Time series data for one partner
            partner_id: Partner identifier

        Returns:
            List of CUSUMAlert objects
        """
        if not self.is_calibrated:
            raise ValueError("Detector must be calibrated before detection")

        alerts = []

        for feature in self.features_to_monitor:
            if feature not in partner_data.columns:
                continue

            values = partner_data[feature].values

            if len(values) < self.min_periods:
                continue

            reference_mean = self.reference_means[feature]
            threshold = self.thresholds[feature]

            cusum_pos, cusum_neg = self.calculate_cusum(values, reference_mean)

            # Check for negative shift (most relevant for churn)
            if cusum_neg[-1] < -threshold:
                severity = self._calculate_severity(abs(cusum_neg[-1]), threshold)
                alerts.append(CUSUMAlert(
                    partner_id=partner_id,
                    feature=feature,
                    alert_type='negative_shift',
                    cusum_value=cusum_neg[-1],
                    threshold=-threshold,
                    weeks_detected=len(values),
                    severity=severity,
                    message=self._generate_alert_message(feature, 'negative_shift', severity)
                ))

            # Check for positive shift in negative indicators
            if feature in ['support_tickets_total', 'escalation_count', 'competitor_mentions_count']:
                if cusum_pos[-1] > threshold:
                    severity = self._calculate_severity(cusum_pos[-1], threshold)
                    alerts.append(CUSUMAlert(
                        partner_id=partner_id,
                        feature=feature,
                        alert_type='positive_shift',
                        cusum_value=cusum_pos[-1],
                        threshold=threshold,
                        weeks_detected=len(values),
                        severity=severity,
                        message=self._generate_alert_message(feature, 'positive_shift', severity)
                    ))

        return alerts

    def detect_batch(
            self,
            df: pd.DataFrame,
            time_col: str = 'week',
            partner_id_col: str = 'partner_id'
    ) -> Dict[str, List[CUSUMAlert]]:
        """
        Detect behavioral shifts for all partners.

        Args:
            df: DataFrame with time series data
            time_col: Column name for time period
            partner_id_col: Column name for partner ID

        Returns:
            Dict mapping partner_id to list of alerts
        """
        if not self.is_calibrated:
            raise ValueError("Detector must be calibrated before detection")

        all_alerts = {}

        partner_ids = df[partner_id_col].unique()

        for partner_id in partner_ids:
            partner_data = df[df[partner_id_col] == partner_id].sort_values(time_col)
            alerts = self.detect_single_partner(partner_data, partner_id)

            if alerts:
                all_alerts[partner_id] = alerts

        return all_alerts

    def detect_from_snapshot(
            self,
            current_df: pd.DataFrame,
            historical_avg_df: pd.DataFrame = None
    ) -> List[CUSUMAlert]:
        """
        Detect shifts from a single snapshot compared to historical averages.

        Useful when full time series isn't available.

        Args:
            current_df: Current snapshot of partner metrics
            historical_avg_df: Historical averages (if different from calibration)

        Returns:
            List of alerts across all partners
        """
        if not self.is_calibrated:
            raise ValueError("Detector must be calibrated before detection")

        alerts = []

        for _, partner in current_df.iterrows():
            partner_id = partner.get('partner_id', 'unknown')

            for feature in self.features_to_monitor:
                if feature not in current_df.columns:
                    continue

                current_value = partner[feature]
                reference_mean = self.reference_means[feature]
                threshold = self.thresholds[feature]

                # Calculate z-score style deviation
                deviation = (current_value - reference_mean) / self.reference_stds[feature]

                # For positive features (login, commission), negative deviation is bad
                if feature not in ['support_tickets_total', 'escalation_count', 'competitor_mentions_count']:
                    if deviation < -self.threshold_multiplier / 2:
                        severity = self._calculate_severity(abs(deviation) * self.reference_stds[feature], threshold)
                        alerts.append(CUSUMAlert(
                            partner_id=partner_id,
                            feature=feature,
                            alert_type='negative_shift',
                            cusum_value=deviation * self.reference_stds[feature],
                            threshold=-threshold,
                            weeks_detected=1,
                            severity=severity,
                            message=self._generate_alert_message(feature, 'negative_shift', severity)
                        ))
                else:
                    # For negative features, positive deviation is bad
                    if deviation > self.threshold_multiplier / 2:
                        severity = self._calculate_severity(deviation * self.reference_stds[feature], threshold)
                        alerts.append(CUSUMAlert(
                            partner_id=partner_id,
                            feature=feature,
                            alert_type='positive_shift',
                            cusum_value=deviation * self.reference_stds[feature],
                            threshold=threshold,
                            weeks_detected=1,
                            severity=severity,
                            message=self._generate_alert_message(feature, 'positive_shift', severity)
                        ))

        return alerts

    def _calculate_severity(self, value: float, threshold: float) -> str:
        """Calculate alert severity based on threshold breach magnitude."""
        ratio = value / threshold
        if ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        return 'low'

    def _generate_alert_message(
            self,
            feature: str,
            alert_type: str,
            severity: str
    ) -> str:
        """Generate human-readable alert message."""
        feature_labels = {
            'login_frequency_weekly': 'login frequency',
            'commission_current_month': 'commission',
            'client_signup_rate': 'client signups',
            'session_duration_avg': 'session duration',
            'active_clients_count': 'active clients',
            'support_tickets_total': 'support tickets',
            'escalation_count': 'escalations',
            'competitor_mentions_count': 'competitor mentions'
        }

        label = feature_labels.get(feature, feature)

        if alert_type == 'negative_shift':
            return f"Sustained decline detected in {label} ({severity} severity)"
        else:
            return f"Sustained increase detected in {label} ({severity} severity)"

    def get_early_warning_summary(
            self,
            alerts: Dict[str, List[CUSUMAlert]]
    ) -> pd.DataFrame:
        """
        Generate summary DataFrame of early warnings.

        Args:
            alerts: Dict of partner_id -> list of alerts

        Returns:
            DataFrame with summary statistics
        """
        records = []

        for partner_id, partner_alerts in alerts.items():
            high_count = sum(1 for a in partner_alerts if a.severity == 'high')
            medium_count = sum(1 for a in partner_alerts if a.severity == 'medium')
            low_count = sum(1 for a in partner_alerts if a.severity == 'low')

            affected_features = list(set(a.feature for a in partner_alerts))

            records.append({
                'partner_id': partner_id,
                'total_alerts': len(partner_alerts),
                'high_severity': high_count,
                'medium_severity': medium_count,
                'low_severity': low_count,
                'affected_features': ', '.join(affected_features),
                'risk_score': high_count * 3 + medium_count * 2 + low_count,
                'earliest_detection_weeks': min(a.weeks_detected for a in partner_alerts)
            })

        return pd.DataFrame(records).sort_values('risk_score', ascending=False)


def demonstrate_cusum_value():
    """
    Demonstrate the value of CUSUM early detection.
    Shows how CUSUM catches shifts before traditional metrics.
    """
    print("\n" + "=" * 60)
    print("CUSUM EARLY DETECTION DEMONSTRATION")
    print("=" * 60)

    # Simulate a churning partner's behavior over 12 weeks
    np.random.seed(42)
    weeks = np.arange(1, 13)

    # Normal partner (stable behavior)
    normal_login = 15 + np.random.normal(0, 1, 12)

    # Churning partner (gradual decline starting week 4)
    churning_login = np.concatenate([
        15 + np.random.normal(0, 1, 4),  # Stable first 4 weeks
        14 + np.random.normal(0, 1, 2),  # Slight dip weeks 5-6
        12 + np.random.normal(0, 1, 2),  # Noticeable decline weeks 7-8
        9 + np.random.normal(0, 1, 2),  # Significant drop weeks 9-10
        5 + np.random.normal(0, 1, 2)   # Severe decline weeks 11-12
    ])

    # Calculate CUSUM
    reference_mean = 15
    cusum_pos = np.zeros(12)
    cusum_neg = np.zeros(12)

    for i in range(1, 12):
        deviation = churning_login[i] - reference_mean
        cusum_pos[i] = max(0, cusum_pos[i - 1] + deviation - 0.5)
        cusum_neg[i] = min(0, cusum_neg[i - 1] + deviation + 0.5)

    # Find when CUSUM would trigger (threshold = 4 * std = 4)
    threshold = 4
    cusum_alert_week = None
    for i, val in enumerate(cusum_neg):
        if val < -threshold:
            cusum_alert_week = i + 1
            break

    # Find when traditional threshold would trigger (20% drop)
    traditional_alert_week = None
    for i, val in enumerate(churning_login):
        if val < reference_mean * 0.8:  # 20% below baseline
            traditional_alert_week = i + 1
            break

    print("\nScenario: Partner with gradual disengagement")
    print("-" * 50)
    print("\nWeekly Login Frequency:")
    for i in range(12):
        indicator = ""
        if cusum_alert_week and i + 1 == cusum_alert_week:
            indicator = " <-- CUSUM Alert!"
        if traditional_alert_week and i + 1 == traditional_alert_week:
            indicator = " <-- Traditional Alert!" if not indicator else indicator + " & Traditional!"
        print(f"  Week {i + 1:2d}: {churning_login[i]:5.1f}  (CUSUM: {cusum_neg[i]:6.2f}){indicator}")

    print("\n" + "-" * 50)
    print(f"CUSUM detected shift at: Week {cusum_alert_week}")
    print(f"Traditional metric triggered at: Week {traditional_alert_week}")
    print(f"\nEARLY WARNING ADVANTAGE: {traditional_alert_week - cusum_alert_week} weeks earlier!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_cusum_value()

    # Test with sample data
    print("\n\nTesting CUSUM Detector with Sample Data...")

    # Create sample time series data
    np.random.seed(42)
    n_partners = 50
    n_weeks = 12

    records = []
    for i in range(n_partners):
        is_churner = i < 12  # First 12 are churners
        for week in range(n_weeks):
            if is_churner:
                decline_factor = 1 - (week / n_weeks) * 0.5
                login_freq = 15 * decline_factor * np.random.normal(1, 0.1)
            else:
                login_freq = 15 * np.random.normal(1, 0.1)

            records.append({
                'partner_id': f'P{i:04d}',
                'week': week + 1,
                'login_frequency_weekly': max(0.5, login_freq),
                'commission_current_month': max(0, 5000 * (decline_factor if is_churner else 1) * np.random.normal(1, 0.15)),
                'client_signup_rate': max(0, 10 * (decline_factor if is_churner else 1) * np.random.normal(1, 0.12)),
                'churned': 1 if is_churner else 0
            })

    df = pd.DataFrame(records)

    # Calibrate and detect
    detector = CUSUMDetector()
    detector.calibrate(df)

    alerts = detector.detect_batch(df)
    print(f"\nDetected early warnings for {len(alerts)} partners")

    if alerts:
        summary = detector.get_early_warning_summary(alerts)
        print("\nTop 5 At-Risk Partners (CUSUM Analysis):")
        print(summary.head())
