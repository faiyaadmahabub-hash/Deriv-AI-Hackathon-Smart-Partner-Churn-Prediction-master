"""
Feature Schema for Partner Churn Prediction System.
Defines 72 features across 8 categories (A-H).
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class FeatureCategory(Enum):
    """Feature categories for organization and SHAP analysis."""
    A_CORE_ID = "Core Identity"
    B_ENGAGEMENT = "Engagement"
    C_CLIENT_HEALTH = "Client Health"
    D_FINANCIAL = "Financial"
    E_RELATIONSHIP = "Relationship"
    F_SUPPORT = "Support"
    G_COMPETITIVE = "Competitive"
    H_NLP_DERIVED = "NLP-Derived"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    category: FeatureCategory
    dtype: str
    description: str
    range: Optional[Tuple] = None
    categorical_values: Optional[List[str]] = None
    is_target: bool = False
    is_cusum_monitored: bool = False


class FeatureSchema:
    """
    Complete feature schema with 72 features across 8 categories.
    Used for data generation, preprocessing, and explainability.
    """

    # ========== CATEGORY A: CORE IDENTITY (4 features) ==========
    CATEGORY_A = [
        FeatureDefinition(
            name="partner_id",
            category=FeatureCategory.A_CORE_ID,
            dtype="string",
            description="Unique partner identifier"
        ),
        FeatureDefinition(
            name="tier",
            category=FeatureCategory.A_CORE_ID,
            dtype="categorical",
            description="Partner tier level",
            categorical_values=["platinum", "gold", "silver", "bronze"]
        ),
        FeatureDefinition(
            name="partner_type",
            category=FeatureCategory.A_CORE_ID,
            dtype="categorical",
            description="Type of partner",
            categorical_values=["affiliate", "ib", "master_affiliate", "payment_agent"]
        ),
        FeatureDefinition(
            name="region",
            category=FeatureCategory.A_CORE_ID,
            dtype="categorical",
            description="Geographic region",
            categorical_values=["asia_pacific", "europe", "latin_america", "africa", "middle_east"]
        ),
    ]

    # ========== CATEGORY B: ENGAGEMENT (12 features) ==========
    CATEGORY_B = [
        FeatureDefinition(
            name="login_frequency_weekly",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Average logins per week",
            range=(0, 30),
            is_cusum_monitored=True
        ),
        FeatureDefinition(
            name="login_frequency_trend",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Week-over-week login change rate",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="session_duration_avg",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Average session duration in minutes",
            range=(1, 120),
            is_cusum_monitored=True
        ),
        FeatureDefinition(
            name="session_duration_trend",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Session duration trend over 4 weeks",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="dashboard_clicks_daily",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Average daily dashboard interactions",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="reports_viewed_weekly",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Number of reports viewed per week",
            range=(0, 50)
        ),
        FeatureDefinition(
            name="marketing_materials_downloaded",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="int",
            description="Marketing materials downloaded last 30 days",
            range=(0, 30)
        ),
        FeatureDefinition(
            name="api_calls_daily",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Average daily API calls",
            range=(0, 1000)
        ),
        FeatureDefinition(
            name="mobile_app_usage_pct",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Percentage of sessions via mobile app",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="last_login_days_ago",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="int",
            description="Days since last login",
            range=(0, 365)
        ),
        FeatureDefinition(
            name="feature_adoption_score",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Score of platform features used (0-100)",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="engagement_consistency_score",
            category=FeatureCategory.B_ENGAGEMENT,
            dtype="float",
            description="Consistency of engagement over time",
            range=(0, 100)
        ),
    ]

    # ========== CATEGORY C: CLIENT HEALTH (10 features) ==========
    CATEGORY_C = [
        FeatureDefinition(
            name="active_clients_count",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="int",
            description="Number of active referred clients",
            range=(0, 5000),
            is_cusum_monitored=True
        ),
        FeatureDefinition(
            name="active_clients_trend",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Month-over-month active client change",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="client_signup_rate",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="New client signups per week",
            range=(0, 100),
            is_cusum_monitored=True
        ),
        FeatureDefinition(
            name="client_signup_trend",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Signup rate trend over 4 weeks",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="client_retention_rate",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="30-day client retention rate",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="client_lifetime_value_avg",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Average client lifetime value (USD)",
            range=(0, 10000)
        ),
        FeatureDefinition(
            name="client_trading_volume",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Total client trading volume (USD)",
            range=(0, 10000000)
        ),
        FeatureDefinition(
            name="client_deposit_total",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Total client deposits (USD)",
            range=(0, 1000000)
        ),
        FeatureDefinition(
            name="dormant_clients_pct",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Percentage of dormant clients",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="client_satisfaction_score",
            category=FeatureCategory.C_CLIENT_HEALTH,
            dtype="float",
            description="Average client satisfaction (1-5)",
            range=(1, 5)
        ),
    ]

    # ========== CATEGORY D: FINANCIAL (14 features) ==========
    CATEGORY_D = [
        FeatureDefinition(
            name="commission_current_month",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Current month commission (USD)",
            range=(0, 100000),
            is_cusum_monitored=True
        ),
        FeatureDefinition(
            name="commission_prev_month",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Previous month commission (USD)",
            range=(0, 100000)
        ),
        FeatureDefinition(
            name="commission_3month_avg",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="3-month average commission (USD)",
            range=(0, 100000)
        ),
        FeatureDefinition(
            name="commission_trend",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Commission trend (3-month slope)",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="commission_volatility",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Commission volatility (std/mean)",
            range=(0, 2)
        ),
        FeatureDefinition(
            name="payout_disputes_count",
            category=FeatureCategory.D_FINANCIAL,
            dtype="int",
            description="Number of payout disputes last 90 days",
            range=(0, 20)
        ),
        FeatureDefinition(
            name="payout_delay_incidents",
            category=FeatureCategory.D_FINANCIAL,
            dtype="int",
            description="Payout delay incidents last 90 days",
            range=(0, 15)
        ),
        FeatureDefinition(
            name="revenue_share_pct",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Current revenue share percentage",
            range=(0, 70)
        ),
        FeatureDefinition(
            name="revenue_share_vs_peers",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Revenue share vs. tier average",
            range=(-50, 50)
        ),
        FeatureDefinition(
            name="bonus_eligibility_score",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Bonus program eligibility score",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="incentive_utilization",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Percentage of available incentives used",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="payment_method_changes",
            category=FeatureCategory.D_FINANCIAL,
            dtype="int",
            description="Payment method changes last 90 days",
            range=(0, 5)
        ),
        FeatureDefinition(
            name="commission_vs_target",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Commission vs. tier target (%)",
            range=(-100, 200)
        ),
        FeatureDefinition(
            name="lifetime_earnings",
            category=FeatureCategory.D_FINANCIAL,
            dtype="float",
            description="Total lifetime earnings (USD)",
            range=(0, 5000000)
        ),
    ]

    # ========== CATEGORY E: RELATIONSHIP (8 features) ==========
    CATEGORY_E = [
        FeatureDefinition(
            name="tenure_months",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="int",
            description="Months as partner",
            range=(1, 120)
        ),
        FeatureDefinition(
            name="event_attendance_rate",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="float",
            description="Percentage of events attended",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="training_completion_rate",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="float",
            description="Percentage of trainings completed",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="webinar_attendance",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="int",
            description="Webinars attended last 90 days",
            range=(0, 20)
        ),
        FeatureDefinition(
            name="rm_meeting_frequency",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="float",
            description="RM meetings per month",
            range=(0, 8)
        ),
        FeatureDefinition(
            name="rm_responsiveness_score",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="float",
            description="Partner responsiveness to RM",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="nps_score",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="int",
            description="Net Promoter Score (0-10)",
            range=(0, 10)
        ),
        FeatureDefinition(
            name="referral_count",
            category=FeatureCategory.E_RELATIONSHIP,
            dtype="int",
            description="Other partners referred",
            range=(0, 50)
        ),
    ]

    # ========== CATEGORY F: SUPPORT (10 features) ==========
    CATEGORY_F = [
        FeatureDefinition(
            name="support_tickets_total",
            category=FeatureCategory.F_SUPPORT,
            dtype="int",
            description="Total support tickets last 90 days",
            range=(0, 50)
        ),
        FeatureDefinition(
            name="support_tickets_trend",
            category=FeatureCategory.F_SUPPORT,
            dtype="float",
            description="Ticket volume trend",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="avg_resolution_time_hours",
            category=FeatureCategory.F_SUPPORT,
            dtype="float",
            description="Average ticket resolution time (hours)",
            range=(0, 168)
        ),
        FeatureDefinition(
            name="escalation_count",
            category=FeatureCategory.F_SUPPORT,
            dtype="int",
            description="Escalated tickets last 90 days",
            range=(0, 20)
        ),
        FeatureDefinition(
            name="escalation_rate",
            category=FeatureCategory.F_SUPPORT,
            dtype="float",
            description="Percentage of tickets escalated",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="complaint_count",
            category=FeatureCategory.F_SUPPORT,
            dtype="int",
            description="Formal complaints last 90 days",
            range=(0, 10)
        ),
        FeatureDefinition(
            name="first_response_satisfaction",
            category=FeatureCategory.F_SUPPORT,
            dtype="float",
            description="First response satisfaction rate",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="self_service_usage",
            category=FeatureCategory.F_SUPPORT,
            dtype="float",
            description="Self-service resolution rate",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="repeated_issues_count",
            category=FeatureCategory.F_SUPPORT,
            dtype="int",
            description="Same issue reported multiple times",
            range=(0, 15)
        ),
        FeatureDefinition(
            name="support_satisfaction_avg",
            category=FeatureCategory.F_SUPPORT,
            dtype="float",
            description="Average support satisfaction score",
            range=(1, 5)
        ),
    ]

    # ========== CATEGORY G: COMPETITIVE (7 features) ==========
    CATEGORY_G = [
        FeatureDefinition(
            name="competitor_mentions_count",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="int",
            description="Competitor mentions in communications",
            range=(0, 20)
        ),
        FeatureDefinition(
            name="pricing_complaints",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="int",
            description="Pricing/rate complaints last 90 days",
            range=(0, 15)
        ),
        FeatureDefinition(
            name="feature_requests_unmet",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="int",
            description="Unmet feature requests",
            range=(0, 20)
        ),
        FeatureDefinition(
            name="platform_comparison_views",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="int",
            description="Competitor comparison page views",
            range=(0, 30)
        ),
        FeatureDefinition(
            name="external_link_clicks",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="int",
            description="Clicks on external/competitor links",
            range=(0, 50)
        ),
        FeatureDefinition(
            name="market_share_concern_flag",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="int",
            description="Expressed market share concerns (0/1)",
            range=(0, 1)
        ),
        FeatureDefinition(
            name="competitive_threat_score",
            category=FeatureCategory.G_COMPETITIVE,
            dtype="float",
            description="Calculated competitive threat level",
            range=(0, 100)
        ),
    ]

    # ========== CATEGORY H: NLP-DERIVED (7 features) ==========
    CATEGORY_H = [
        FeatureDefinition(
            name="sentiment_score_avg",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="float",
            description="Average communication sentiment (-1 to 1)",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="sentiment_trend",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="float",
            description="Sentiment change over 30 days",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="frustration_keyword_count",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="int",
            description="Frustration keywords detected",
            range=(0, 50)
        ),
        FeatureDefinition(
            name="urgency_indicator",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="float",
            description="Communication urgency level",
            range=(0, 100)
        ),
        FeatureDefinition(
            name="churn_intent_signals",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="int",
            description="Explicit churn signals detected",
            range=(0, 10)
        ),
        FeatureDefinition(
            name="communication_frequency_change",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="float",
            description="Change in communication frequency",
            range=(-1, 1)
        ),
        FeatureDefinition(
            name="topic_shift_detected",
            category=FeatureCategory.H_NLP_DERIVED,
            dtype="int",
            description="Topic shift towards exit themes (0/1)",
            range=(0, 1)
        ),
    ]

    # Target variable
    TARGET = FeatureDefinition(
        name="churned",
        category=FeatureCategory.A_CORE_ID,
        dtype="int",
        description="Whether partner churned (0/1)",
        range=(0, 1),
        is_target=True
    )

    @classmethod
    def get_all_features(cls) -> List[FeatureDefinition]:
        """Get all features as a flat list."""
        return (cls.CATEGORY_A + cls.CATEGORY_B + cls.CATEGORY_C +
                cls.CATEGORY_D + cls.CATEGORY_E + cls.CATEGORY_F +
                cls.CATEGORY_G + cls.CATEGORY_H)

    @classmethod
    def get_feature_names(cls, include_target: bool = False) -> List[str]:
        """Get list of all feature names."""
        features = cls.get_all_features()
        names = [f.name for f in features if not f.is_target]
        if include_target:
            names.append(cls.TARGET.name)
        return names

    @classmethod
    def get_numeric_features(cls) -> List[str]:
        """Get names of numeric features (float/int)."""
        features = cls.get_all_features()
        return [f.name for f in features
                if f.dtype in ['float', 'int'] and not f.is_target and f.name != 'partner_id']

    @classmethod
    def get_categorical_features(cls) -> List[str]:
        """Get names of categorical features."""
        features = cls.get_all_features()
        return [f.name for f in features if f.dtype == 'categorical']

    @classmethod
    def get_cusum_features(cls) -> List[str]:
        """Get features monitored by CUSUM detector."""
        features = cls.get_all_features()
        return [f.name for f in features if f.is_cusum_monitored]

    @classmethod
    def get_features_by_category(cls, category: FeatureCategory) -> List[FeatureDefinition]:
        """Get features for a specific category."""
        features = cls.get_all_features()
        return [f for f in features if f.category == category]

    @classmethod
    def get_category_mapping(cls) -> Dict[str, str]:
        """Get mapping of feature names to category names (for SHAP)."""
        features = cls.get_all_features()
        return {f.name: f.category.value for f in features}

    @classmethod
    def get_feature_ranges(cls) -> Dict[str, Tuple]:
        """Get valid ranges for all numeric features."""
        features = cls.get_all_features()
        return {f.name: f.range for f in features if f.range is not None}

    @classmethod
    def get_feature_count(cls) -> int:
        """Get total number of features (excluding target)."""
        return len([f for f in cls.get_all_features() if not f.is_target])

    @classmethod
    def get_model_features(cls) -> List[str]:
        """Get feature names used for model training (excludes ID, target, and categorical)."""
        exclude = {'partner_id', 'churned', 'churn_probability', 'days_until_churn'}
        all_features = cls.get_all_features()
        return [f.name for f in all_features
                if f.name not in exclude and not f.is_target and f.dtype != 'string']

    @classmethod
    def validate_dataframe(cls, df) -> Dict[str, List[str]]:
        """Validate a dataframe against the schema."""
        issues = {'missing_features': [], 'extra_features': [], 'type_mismatches': []}
        expected_features = set(cls.get_feature_names(include_target=True))
        actual_features = set(df.columns)

        issues['missing_features'] = list(expected_features - actual_features)
        issues['extra_features'] = list(actual_features - expected_features - {'partner_id'})

        return issues
