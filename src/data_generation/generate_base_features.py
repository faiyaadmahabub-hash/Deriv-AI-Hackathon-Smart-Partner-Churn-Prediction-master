"""
Generate base features for synthetic partner data.
Creates realistic partner profiles before applying churn patterns.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.feature_schema import FeatureSchema, FeatureCategory


def generate_partner_ids(n_partners: int) -> np.ndarray:
    """Generate unique partner IDs."""
    return np.array([f"P{str(i).zfill(6)}" for i in range(1, n_partners + 1)])


def generate_categorical_features(n_partners: int, seed: int) -> Dict[str, np.ndarray]:
    """Generate categorical features based on distributions."""
    np.random.seed(seed)

    # Tier distribution
    tiers = np.random.choice(
        list(Config.TIER_DISTRIBUTION.keys()),
        size=n_partners,
        p=list(Config.TIER_DISTRIBUTION.values())
    )

    # Partner type distribution
    partner_types = np.random.choice(
        list(Config.TYPE_DISTRIBUTION.keys()),
        size=n_partners,
        p=list(Config.TYPE_DISTRIBUTION.values())
    )

    # Region distribution
    regions = np.random.choice(
        list(Config.REGION_DISTRIBUTION.keys()),
        size=n_partners,
        p=list(Config.REGION_DISTRIBUTION.values())
    )

    return {
        'tier': tiers,
        'partner_type': partner_types,
        'region': regions
    }


def get_tier_multiplier(tier: str) -> float:
    """Get performance multiplier based on tier."""
    multipliers = {
        'platinum': 3.0,
        'gold': 2.0,
        'silver': 1.2,
        'bronze': 1.0
    }
    return multipliers.get(tier, 1.0)


def generate_engagement_features(n_partners: int, tiers: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category B: Engagement features."""
    np.random.seed(seed)
    features = {}

    # Base engagement varies by tier
    tier_multipliers = np.array([get_tier_multiplier(t) for t in tiers])

    # Login frequency (weekly)
    features['login_frequency_weekly'] = np.clip(
        np.random.gamma(4, 2, n_partners) * tier_multipliers * 0.8,
        0, 30
    )

    # Login frequency trend (mostly stable or slight decline)
    features['login_frequency_trend'] = np.clip(
        np.random.normal(0.02, 0.15, n_partners),
        -1, 1
    )

    # Session duration (minutes)
    features['session_duration_avg'] = np.clip(
        np.random.gamma(5, 4, n_partners) * tier_multipliers * 0.7,
        1, 120
    )

    # Session duration trend
    features['session_duration_trend'] = np.clip(
        np.random.normal(0.01, 0.12, n_partners),
        -1, 1
    )

    # Dashboard clicks
    features['dashboard_clicks_daily'] = np.clip(
        np.random.gamma(3, 4, n_partners) * tier_multipliers,
        0, 100
    )

    # Reports viewed weekly
    features['reports_viewed_weekly'] = np.clip(
        np.random.poisson(5, n_partners) * tier_multipliers,
        0, 50
    )

    # Marketing materials downloaded
    features['marketing_materials_downloaded'] = np.clip(
        np.random.poisson(3, n_partners) * tier_multipliers * 0.5,
        0, 30
    ).astype(int)

    # API calls daily
    features['api_calls_daily'] = np.clip(
        np.random.exponential(50, n_partners) * tier_multipliers * 1.5,
        0, 1000
    )

    # Mobile app usage percentage
    features['mobile_app_usage_pct'] = np.clip(
        np.random.beta(2, 3, n_partners) * 100,
        0, 100
    )

    # Days since last login (lower is better)
    features['last_login_days_ago'] = np.clip(
        np.random.exponential(3, n_partners) / tier_multipliers,
        0, 365
    ).astype(int)

    # Feature adoption score
    features['feature_adoption_score'] = np.clip(
        np.random.beta(3, 2, n_partners) * 100 * (tier_multipliers / 2),
        0, 100
    )

    # Engagement consistency
    features['engagement_consistency_score'] = np.clip(
        np.random.beta(4, 2, n_partners) * 100,
        0, 100
    )

    return features


def generate_client_health_features(n_partners: int, tiers: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category C: Client Health features."""
    np.random.seed(seed + 1)
    features = {}

    tier_multipliers = np.array([get_tier_multiplier(t) for t in tiers])

    # Active clients count
    base_clients = np.random.gamma(10, 20, n_partners)
    features['active_clients_count'] = np.clip(
        base_clients * tier_multipliers * 5,
        0, 5000
    ).astype(int)

    # Active clients trend
    features['active_clients_trend'] = np.clip(
        np.random.normal(0.03, 0.15, n_partners),
        -1, 1
    )

    # Client signup rate
    features['client_signup_rate'] = np.clip(
        np.random.gamma(3, 2, n_partners) * tier_multipliers,
        0, 100
    )

    # Client signup trend
    features['client_signup_trend'] = np.clip(
        np.random.normal(0.01, 0.18, n_partners),
        -1, 1
    )

    # Client retention rate
    features['client_retention_rate'] = np.clip(
        np.random.beta(6, 2, n_partners) * 100,
        0, 100
    )

    # Client lifetime value
    features['client_lifetime_value_avg'] = np.clip(
        np.random.gamma(3, 300, n_partners) * tier_multipliers,
        0, 10000
    )

    # Client trading volume
    features['client_trading_volume'] = np.clip(
        np.random.gamma(2, 50000, n_partners) * tier_multipliers * 3,
        0, 10000000
    )

    # Client deposits
    features['client_deposit_total'] = np.clip(
        np.random.gamma(3, 10000, n_partners) * tier_multipliers * 2,
        0, 1000000
    )

    # Dormant clients percentage
    features['dormant_clients_pct'] = np.clip(
        np.random.beta(2, 5, n_partners) * 100,
        0, 100
    )

    # Client satisfaction
    features['client_satisfaction_score'] = np.clip(
        np.random.normal(3.8, 0.6, n_partners),
        1, 5
    )

    return features


def generate_financial_features(n_partners: int, tiers: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category D: Financial features."""
    np.random.seed(seed + 2)
    features = {}

    tier_multipliers = np.array([get_tier_multiplier(t) for t in tiers])

    # Base commission by tier
    base_commission = np.array([Config.AVG_MONTHLY_COMMISSION.get(t, 3000) for t in tiers])

    # Current month commission
    features['commission_current_month'] = np.clip(
        base_commission * np.random.normal(1.0, 0.25, n_partners),
        0, 100000
    )

    # Previous month commission
    features['commission_prev_month'] = np.clip(
        base_commission * np.random.normal(1.0, 0.22, n_partners),
        0, 100000
    )

    # 3-month average
    features['commission_3month_avg'] = np.clip(
        base_commission * np.random.normal(1.0, 0.18, n_partners),
        0, 100000
    )

    # Commission trend
    features['commission_trend'] = np.clip(
        np.random.normal(0.02, 0.15, n_partners),
        -1, 1
    )

    # Commission volatility
    features['commission_volatility'] = np.clip(
        np.random.exponential(0.2, n_partners),
        0, 2
    )

    # Payout disputes
    features['payout_disputes_count'] = np.clip(
        np.random.poisson(0.5, n_partners),
        0, 20
    ).astype(int)

    # Payout delays
    features['payout_delay_incidents'] = np.clip(
        np.random.poisson(0.3, n_partners),
        0, 15
    ).astype(int)

    # Revenue share percentage
    features['revenue_share_pct'] = np.clip(
        30 + tier_multipliers * 8 + np.random.normal(0, 5, n_partners),
        0, 70
    )

    # Revenue share vs peers
    features['revenue_share_vs_peers'] = np.clip(
        np.random.normal(0, 10, n_partners),
        -50, 50
    )

    # Bonus eligibility
    features['bonus_eligibility_score'] = np.clip(
        np.random.beta(3, 2, n_partners) * 100 * (tier_multipliers / 2),
        0, 100
    )

    # Incentive utilization
    features['incentive_utilization'] = np.clip(
        np.random.beta(2, 2, n_partners) * 100,
        0, 100
    )

    # Payment method changes
    features['payment_method_changes'] = np.clip(
        np.random.poisson(0.3, n_partners),
        0, 5
    ).astype(int)

    # Commission vs target
    features['commission_vs_target'] = np.clip(
        np.random.normal(10, 30, n_partners),
        -100, 200
    )

    # Lifetime earnings
    tenure_months = np.random.randint(1, 80, n_partners)
    features['lifetime_earnings'] = np.clip(
        base_commission * tenure_months * np.random.normal(0.9, 0.2, n_partners),
        0, 5000000
    )

    return features


def generate_relationship_features(n_partners: int, tiers: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category E: Relationship features."""
    np.random.seed(seed + 3)
    features = {}

    tier_multipliers = np.array([get_tier_multiplier(t) for t in tiers])

    # Tenure months
    features['tenure_months'] = np.clip(
        np.random.gamma(3, 12, n_partners),
        1, 120
    ).astype(int)

    # Event attendance rate
    features['event_attendance_rate'] = np.clip(
        np.random.beta(2, 2, n_partners) * 100 * (tier_multipliers / 2.5),
        0, 100
    )

    # Training completion rate
    features['training_completion_rate'] = np.clip(
        np.random.beta(3, 2, n_partners) * 100,
        0, 100
    )

    # Webinar attendance
    features['webinar_attendance'] = np.clip(
        np.random.poisson(2, n_partners) * tier_multipliers / 1.5,
        0, 20
    ).astype(int)

    # RM meeting frequency
    features['rm_meeting_frequency'] = np.clip(
        np.random.gamma(2, 0.8, n_partners) * tier_multipliers / 2,
        0, 8
    )

    # RM responsiveness
    features['rm_responsiveness_score'] = np.clip(
        np.random.beta(4, 2, n_partners) * 100,
        0, 100
    )

    # NPS Score
    features['nps_score'] = np.clip(
        np.random.normal(7, 2, n_partners),
        0, 10
    ).astype(int)

    # Referral count
    features['referral_count'] = np.clip(
        np.random.poisson(1.5, n_partners) * tier_multipliers / 2,
        0, 50
    ).astype(int)

    return features


def generate_support_features(n_partners: int, tiers: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category F: Support features."""
    np.random.seed(seed + 4)
    features = {}

    # Support tickets
    features['support_tickets_total'] = np.clip(
        np.random.poisson(5, n_partners),
        0, 50
    ).astype(int)

    # Tickets trend
    features['support_tickets_trend'] = np.clip(
        np.random.normal(0, 0.2, n_partners),
        -1, 1
    )

    # Resolution time
    features['avg_resolution_time_hours'] = np.clip(
        np.random.gamma(2, 8, n_partners),
        0, 168
    )

    # Escalations
    features['escalation_count'] = np.clip(
        np.random.poisson(0.8, n_partners),
        0, 20
    ).astype(int)

    # Escalation rate
    total_tickets = features['support_tickets_total']
    escalations = features['escalation_count']
    features['escalation_rate'] = np.where(
        total_tickets > 0,
        np.clip(escalations / total_tickets * 100, 0, 100),
        0
    )

    # Complaints
    features['complaint_count'] = np.clip(
        np.random.poisson(0.3, n_partners),
        0, 10
    ).astype(int)

    # First response satisfaction
    features['first_response_satisfaction'] = np.clip(
        np.random.beta(4, 2, n_partners) * 100,
        0, 100
    )

    # Self-service usage
    features['self_service_usage'] = np.clip(
        np.random.beta(2, 3, n_partners) * 100,
        0, 100
    )

    # Repeated issues
    features['repeated_issues_count'] = np.clip(
        np.random.poisson(0.5, n_partners),
        0, 15
    ).astype(int)

    # Support satisfaction
    features['support_satisfaction_avg'] = np.clip(
        np.random.normal(3.7, 0.7, n_partners),
        1, 5
    )

    return features


def generate_competitive_features(n_partners: int, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category G: Competitive features."""
    np.random.seed(seed + 5)
    features = {}

    # Competitor mentions (mostly 0, occasional mentions)
    features['competitor_mentions_count'] = np.clip(
        np.random.poisson(0.5, n_partners),
        0, 20
    ).astype(int)

    # Pricing complaints
    features['pricing_complaints'] = np.clip(
        np.random.poisson(0.3, n_partners),
        0, 15
    ).astype(int)

    # Unmet feature requests
    features['feature_requests_unmet'] = np.clip(
        np.random.poisson(1, n_partners),
        0, 20
    ).astype(int)

    # Platform comparison views
    features['platform_comparison_views'] = np.clip(
        np.random.poisson(0.5, n_partners),
        0, 30
    ).astype(int)

    # External link clicks
    features['external_link_clicks'] = np.clip(
        np.random.poisson(1, n_partners),
        0, 50
    ).astype(int)

    # Market share concern flag
    features['market_share_concern_flag'] = np.random.binomial(1, 0.08, n_partners)

    # Competitive threat score (calculated)
    features['competitive_threat_score'] = np.clip(
        (features['competitor_mentions_count'] * 10 +
         features['pricing_complaints'] * 8 +
         features['platform_comparison_views'] * 5 +
         features['market_share_concern_flag'] * 20 +
         np.random.normal(0, 5, n_partners)),
        0, 100
    )

    return features


def generate_nlp_features(n_partners: int, seed: int) -> Dict[str, np.ndarray]:
    """Generate Category H: NLP-derived features."""
    np.random.seed(seed + 6)
    features = {}

    # Sentiment score (mostly positive)
    features['sentiment_score_avg'] = np.clip(
        np.random.normal(0.3, 0.3, n_partners),
        -1, 1
    )

    # Sentiment trend
    features['sentiment_trend'] = np.clip(
        np.random.normal(0, 0.15, n_partners),
        -1, 1
    )

    # Frustration keywords
    features['frustration_keyword_count'] = np.clip(
        np.random.poisson(1, n_partners),
        0, 50
    ).astype(int)

    # Urgency indicator
    features['urgency_indicator'] = np.clip(
        np.random.exponential(15, n_partners),
        0, 100
    )

    # Churn intent signals
    features['churn_intent_signals'] = np.clip(
        np.random.poisson(0.2, n_partners),
        0, 10
    ).astype(int)

    # Communication frequency change
    features['communication_frequency_change'] = np.clip(
        np.random.normal(0, 0.2, n_partners),
        -1, 1
    )

    # Topic shift detected
    features['topic_shift_detected'] = np.random.binomial(1, 0.05, n_partners)

    return features


def generate_base_features(
        n_partners: int = None,
        seed: int = None
) -> pd.DataFrame:
    """
    Generate base features for all partners.

    Args:
        n_partners: Number of partners to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all base features
    """
    n_partners = n_partners or Config.N_PARTNERS
    seed = seed or Config.RANDOM_SEED

    # Generate partner IDs
    partner_ids = generate_partner_ids(n_partners)

    # Generate categorical features
    categorical = generate_categorical_features(n_partners, seed)
    tiers = categorical['tier']

    # Generate all feature categories
    engagement = generate_engagement_features(n_partners, tiers, seed)
    client_health = generate_client_health_features(n_partners, tiers, seed)
    financial = generate_financial_features(n_partners, tiers, seed)
    relationship = generate_relationship_features(n_partners, tiers, seed)
    support = generate_support_features(n_partners, tiers, seed)
    competitive = generate_competitive_features(n_partners, seed)
    nlp = generate_nlp_features(n_partners, seed)

    # Combine all features into DataFrame
    data = {'partner_id': partner_ids}
    data.update(categorical)
    data.update(engagement)
    data.update(client_health)
    data.update(financial)
    data.update(relationship)
    data.update(support)
    data.update(competitive)
    data.update(nlp)

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":
    # Test generation
    df = generate_base_features(n_partners=100)
    print(f"Generated {len(df)} partners with {len(df.columns)} features")
    print(f"\nFeature columns: {list(df.columns)}")
    print(f"\nTier distribution:\n{df['tier'].value_counts(normalize=True)}")
    print(f"\nSample data:\n{df.head()}")
