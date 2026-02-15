"""
Apply decline cascade patterns to simulate realistic churn behavior.
Partners who churn exhibit predictable decline patterns across multiple features.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


def calculate_churn_probability(row: pd.Series) -> float:
    """
    Calculate base churn probability based on partner characteristics.
    Higher risk partners have higher base probability.
    """
    base_prob = Config.CHURN_RATE

    # Tier adjustment (lower tiers more likely to churn)
    tier_adjustments = {
        'platinum': -0.15,
        'gold': -0.08,
        'silver': 0.02,
        'bronze': 0.08
    }
    base_prob += tier_adjustments.get(row['tier'], 0)

    # Tenure adjustment (newer partners more likely to churn)
    if row['tenure_months'] < 6:
        base_prob += 0.10
    elif row['tenure_months'] < 12:
        base_prob += 0.05
    elif row['tenure_months'] > 36:
        base_prob -= 0.08

    # Commission trend adjustment
    if row['commission_trend'] < -0.2:
        base_prob += 0.12
    elif row['commission_trend'] > 0.2:
        base_prob -= 0.08

    # Support issues adjustment
    if row['escalation_count'] > 3:
        base_prob += 0.08
    if row['complaint_count'] > 2:
        base_prob += 0.10

    # Competitive signals adjustment
    if row['competitor_mentions_count'] > 2:
        base_prob += 0.12
    if row['pricing_complaints'] > 2:
        base_prob += 0.08

    # Engagement adjustment
    if row['login_frequency_weekly'] < 2:
        base_prob += 0.10
    elif row['login_frequency_weekly'] > 10:
        base_prob -= 0.05

    return np.clip(base_prob, 0.05, 0.70)


def assign_churn_labels(
        df: pd.DataFrame,
        target_churn_rate: float = None,
        seed: int = None
) -> pd.DataFrame:
    """
    Assign churn labels based on probability-weighted selection.

    Args:
        df: DataFrame with partner features
        target_churn_rate: Desired overall churn rate
        seed: Random seed

    Returns:
        DataFrame with 'churned' column and 'churn_probability' column
    """
    target_churn_rate = target_churn_rate or Config.CHURN_RATE
    seed = seed or Config.RANDOM_SEED
    np.random.seed(seed)

    df = df.copy()

    # Calculate churn probability for each partner
    df['churn_probability'] = df.apply(calculate_churn_probability, axis=1)

    # Normalize probabilities to achieve target churn rate
    current_avg_prob = df['churn_probability'].mean()
    scaling_factor = target_churn_rate / current_avg_prob
    df['churn_probability'] = np.clip(df['churn_probability'] * scaling_factor, 0.02, 0.95)

    # Assign churn labels based on probability
    df['churned'] = (np.random.random(len(df)) < df['churn_probability']).astype(int)

    # Verify and adjust to hit target rate
    actual_rate = df['churned'].mean()
    if abs(actual_rate - target_churn_rate) > 0.02:
        # Adjust by flipping some labels
        n_target = int(target_churn_rate * len(df))
        n_current = df['churned'].sum()

        if n_current < n_target:
            # Need more churners - flip highest probability non-churners
            non_churned = df[df['churned'] == 0].nlargest(n_target - n_current, 'churn_probability').index
            df.loc[non_churned, 'churned'] = 1
        else:
            # Need fewer churners - flip lowest probability churners
            churned = df[df['churned'] == 1].nsmallest(n_current - n_target, 'churn_probability').index
            df.loc[churned, 'churned'] = 0

    return df


def apply_phase_decline(
        value: float,
        drop_range: Tuple[float, float],
        is_increase: bool = False
) -> float:
    """Apply a decline (or increase) to a value based on range."""
    drop_pct = np.random.uniform(drop_range[0], drop_range[1])
    if is_increase:
        return value * (1 + drop_pct)
    return value * (1 - drop_pct)


def apply_decline_cascade(
        df: pd.DataFrame,
        seed: int = None
) -> pd.DataFrame:
    """
    Apply realistic decline cascade to churning partners.

    Churning partners exhibit gradual decline across multiple phases:
    - Phase 1: Early warning signals (engagement drops)
    - Phase 2: Engagement decline deepens
    - Phase 3: Performance impact (clients, revenue)
    - Phase 4: Imminent churn (severe decline)

    Args:
        df: DataFrame with partner features and churn labels
        seed: Random seed

    Returns:
        DataFrame with modified features for churning partners
    """
    seed = seed or Config.RANDOM_SEED
    np.random.seed(seed)

    df = df.copy()
    churned_mask = df['churned'] == 1
    n_churned = churned_mask.sum()

    if n_churned == 0:
        return df

    cascade = Config.DECLINE_CASCADE

    # Determine which phase each churning partner is in (for variety)
    # Most are in phase 3-4 (closer to churn), some in earlier phases
    phase_probs = [0.10, 0.20, 0.35, 0.35]  # Phase 1, 2, 3, 4
    phases = np.random.choice([1, 2, 3, 4], size=n_churned, p=phase_probs)

    churned_indices = df[churned_mask].index.tolist()

    for idx, phase in zip(churned_indices, phases):
        # Apply cumulative decline based on phase
        # Phase 4 partners have gone through all phases, Phase 1 only phase 1

        if phase >= 1:
            # Phase 1 declines
            p1 = cascade['phase_1']
            df.loc[idx, 'login_frequency_weekly'] = apply_phase_decline(
                df.loc[idx, 'login_frequency_weekly'],
                p1['login_frequency_drop']
            )
            df.loc[idx, 'session_duration_avg'] = apply_phase_decline(
                df.loc[idx, 'session_duration_avg'],
                p1['session_duration_drop']
            )
            df.loc[idx, 'dashboard_clicks_daily'] = apply_phase_decline(
                df.loc[idx, 'dashboard_clicks_daily'],
                p1['dashboard_clicks_drop']
            )
            # Update trends to negative
            df.loc[idx, 'login_frequency_trend'] = np.clip(
                df.loc[idx, 'login_frequency_trend'] - np.random.uniform(0.1, 0.25),
                -1, 1
            )

        if phase >= 2:
            # Phase 2 declines (cumulative)
            p2 = cascade['phase_2']
            df.loc[idx, 'login_frequency_weekly'] = apply_phase_decline(
                df.loc[idx, 'login_frequency_weekly'],
                p2['login_frequency_drop']
            )
            df.loc[idx, 'support_tickets_total'] = int(apply_phase_decline(
                df.loc[idx, 'support_tickets_total'],
                p2['support_tickets_increase'],
                is_increase=True
            ))
            df.loc[idx, 'event_attendance_rate'] = apply_phase_decline(
                df.loc[idx, 'event_attendance_rate'],
                p2['event_attendance_drop']
            )
            df.loc[idx, 'training_completion_rate'] = apply_phase_decline(
                df.loc[idx, 'training_completion_rate'],
                p2['training_completion_drop']
            )
            # Increase escalations
            df.loc[idx, 'escalation_count'] = min(
                int(df.loc[idx, 'escalation_count'] + np.random.randint(1, 4)),
                20
            )
            # Update support trend
            df.loc[idx, 'support_tickets_trend'] = np.clip(
                df.loc[idx, 'support_tickets_trend'] + np.random.uniform(0.2, 0.5),
                -1, 1
            )

        if phase >= 3:
            # Phase 3 declines (performance impact)
            p3 = cascade['phase_3']
            df.loc[idx, 'client_signup_rate'] = apply_phase_decline(
                df.loc[idx, 'client_signup_rate'],
                p3['client_signup_drop']
            )
            df.loc[idx, 'client_retention_rate'] = apply_phase_decline(
                df.loc[idx, 'client_retention_rate'],
                p3['client_retention_drop']
            )
            df.loc[idx, 'commission_current_month'] = apply_phase_decline(
                df.loc[idx, 'commission_current_month'],
                p3['commission_drop']
            )
            # Update trends
            df.loc[idx, 'commission_trend'] = np.clip(
                df.loc[idx, 'commission_trend'] - np.random.uniform(0.15, 0.35),
                -1, 1
            )
            df.loc[idx, 'client_signup_trend'] = np.clip(
                df.loc[idx, 'client_signup_trend'] - np.random.uniform(0.15, 0.30),
                -1, 1
            )
            df.loc[idx, 'active_clients_trend'] = np.clip(
                df.loc[idx, 'active_clients_trend'] - np.random.uniform(0.10, 0.25),
                -1, 1
            )

        if phase >= 4:
            # Phase 4 declines (imminent churn)
            p4 = cascade['phase_4']
            df.loc[idx, 'login_frequency_weekly'] = apply_phase_decline(
                df.loc[idx, 'login_frequency_weekly'],
                p4['login_frequency_drop']
            )
            df.loc[idx, 'commission_current_month'] = apply_phase_decline(
                df.loc[idx, 'commission_current_month'],
                p4['commission_drop']
            )
            df.loc[idx, 'active_clients_count'] = int(apply_phase_decline(
                df.loc[idx, 'active_clients_count'],
                p4['active_clients_drop']
            ))
            df.loc[idx, 'competitor_mentions_count'] = int(apply_phase_decline(
                df.loc[idx, 'competitor_mentions_count'],
                p4['competitor_mentions_increase'],
                is_increase=True
            ))
            # Severe trend deterioration
            df.loc[idx, 'login_frequency_trend'] = np.clip(
                df.loc[idx, 'login_frequency_trend'] - np.random.uniform(0.3, 0.5),
                -1, 1
            )
            # Increase competitive threat score
            df.loc[idx, 'competitive_threat_score'] = min(
                df.loc[idx, 'competitive_threat_score'] + np.random.uniform(15, 35),
                100
            )
            # Sentiment deterioration
            df.loc[idx, 'sentiment_score_avg'] = np.clip(
                df.loc[idx, 'sentiment_score_avg'] - np.random.uniform(0.2, 0.5),
                -1, 1
            )
            df.loc[idx, 'sentiment_trend'] = np.clip(
                df.loc[idx, 'sentiment_trend'] - np.random.uniform(0.2, 0.4),
                -1, 1
            )
            # Increase frustration and churn signals
            df.loc[idx, 'frustration_keyword_count'] = int(min(
                df.loc[idx, 'frustration_keyword_count'] + np.random.randint(3, 10),
                50
            ))
            df.loc[idx, 'churn_intent_signals'] = int(min(
                df.loc[idx, 'churn_intent_signals'] + np.random.randint(1, 4),
                10
            ))
            # Update days since login
            df.loc[idx, 'last_login_days_ago'] = int(min(
                df.loc[idx, 'last_login_days_ago'] + np.random.randint(5, 20),
                365
            ))
            # NPS drops
            df.loc[idx, 'nps_score'] = max(
                int(df.loc[idx, 'nps_score'] - np.random.randint(2, 5)),
                0
            )

    # Add some noise to non-churning partners too (natural variation)
    non_churned_mask = df['churned'] == 0
    n_non_churned = non_churned_mask.sum()

    # Apply small random variations to non-churners
    noise_factor = 0.05
    for col in ['login_frequency_weekly', 'session_duration_avg', 'commission_current_month']:
        df.loc[non_churned_mask, col] = df.loc[non_churned_mask, col] * (
                1 + np.random.normal(0, noise_factor, n_non_churned)
        )

    # Ensure all values are within valid ranges
    df = clip_to_valid_ranges(df)

    return df


def clip_to_valid_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Clip all numeric features to their valid ranges."""
    from config.feature_schema import FeatureSchema

    ranges = FeatureSchema.get_feature_ranges()

    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            df[col] = np.clip(df[col], min_val, max_val)

    # Ensure integer columns stay as integers
    int_cols = ['active_clients_count', 'marketing_materials_downloaded',
                'last_login_days_ago', 'tenure_months', 'webinar_attendance',
                'nps_score', 'referral_count', 'support_tickets_total',
                'escalation_count', 'complaint_count', 'repeated_issues_count',
                'competitor_mentions_count', 'pricing_complaints',
                'feature_requests_unmet', 'platform_comparison_views',
                'external_link_clicks', 'market_share_concern_flag',
                'frustration_keyword_count', 'churn_intent_signals',
                'topic_shift_detected', 'payout_disputes_count',
                'payout_delay_incidents', 'payment_method_changes']

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features for time-series analysis."""
    df = df.copy()

    # Days until predicted churn (for churners, simulate discovery timing)
    df['days_until_churn'] = None
    churned_mask = df['churned'] == 1

    # Churners are at various stages
    n_churned = churned_mask.sum()
    df.loc[churned_mask, 'days_until_churn'] = np.random.choice(
        [7, 14, 21, 30, 45, 60],
        size=n_churned,
        p=[0.15, 0.20, 0.25, 0.20, 0.12, 0.08]
    )

    return df


if __name__ == "__main__":
    # Test the cascade
    from generate_base_features import generate_base_features

    # Generate base data
    df = generate_base_features(n_partners=100)
    print(f"Generated {len(df)} partners")

    # Assign churn labels
    df = assign_churn_labels(df)
    print(f"Churn rate: {df['churned'].mean():.2%}")

    # Apply decline cascade
    df = apply_decline_cascade(df)
    print(f"\nAfter decline cascade:")
    print(f"Churned partners avg login freq: {df[df['churned'] == 1]['login_frequency_weekly'].mean():.2f}")
    print(f"Non-churned partners avg login freq: {df[df['churned'] == 0]['login_frequency_weekly'].mean():.2f}")

    # Add temporal features
    df = add_temporal_features(df)
    print(f"\nDays until churn distribution for churners:")
    print(df[df['churned'] == 1]['days_until_churn'].value_counts())
