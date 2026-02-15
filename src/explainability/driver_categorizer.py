"""
Driver categorization for SHAP explanations.
Maps features to business-friendly categories for interpretation.
"""
import pandas as pd
from typing import Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.feature_schema import FeatureSchema, FeatureCategory


class DriverCategorizer:
    """
    Categorize SHAP drivers into business-friendly categories.
    Helps relationship managers understand what's driving churn risk.
    """

    # Human-readable category labels
    CATEGORY_LABELS = {
        FeatureCategory.A_CORE_ID.value: "Profile",
        FeatureCategory.B_ENGAGEMENT.value: "Engagement",
        FeatureCategory.C_CLIENT_HEALTH.value: "Client Health",
        FeatureCategory.D_FINANCIAL.value: "Financial",
        FeatureCategory.E_RELATIONSHIP.value: "Relationship",
        FeatureCategory.F_SUPPORT.value: "Support",
        FeatureCategory.G_COMPETITIVE.value: "Competitive",
        FeatureCategory.H_NLP_DERIVED.value: "Sentiment"
    }

    # Category icons for dashboard
    CATEGORY_ICONS = {
        "Profile": "[P]",
        "Engagement": "[E]",
        "Client Health": "[C]",
        "Financial": "[F]",
        "Relationship": "[R]",
        "Support": "[S]",
        "Competitive": "[X]",
        "Sentiment": "[N]"
    }

    # Category colors for visualization
    CATEGORY_COLORS = {
        "Profile": "#9E9E9E",
        "Engagement": "#2196F3",
        "Client Health": "#4CAF50",
        "Financial": "#FF9800",
        "Relationship": "#9C27B0",
        "Support": "#F44336",
        "Competitive": "#E91E63",
        "Sentiment": "#00BCD4"
    }

    # Human-readable feature labels
    FEATURE_LABELS = {
        # Engagement
        'login_frequency_weekly': 'Weekly Logins',
        'login_frequency_trend': 'Login Trend',
        'session_duration_avg': 'Avg Session Duration',
        'session_duration_trend': 'Session Duration Trend',
        'dashboard_clicks_daily': 'Daily Dashboard Clicks',
        'reports_viewed_weekly': 'Weekly Reports Viewed',
        'marketing_materials_downloaded': 'Marketing Downloads',
        'api_calls_daily': 'Daily API Calls',
        'mobile_app_usage_pct': 'Mobile App Usage',
        'last_login_days_ago': 'Days Since Last Login',
        'feature_adoption_score': 'Feature Adoption',
        'engagement_consistency_score': 'Engagement Consistency',

        # Client Health
        'active_clients_count': 'Active Clients',
        'active_clients_trend': 'Client Growth Trend',
        'client_signup_rate': 'Client Signup Rate',
        'client_signup_trend': 'Signup Trend',
        'client_retention_rate': 'Client Retention Rate',
        'client_lifetime_value_avg': 'Avg Client LTV',
        'client_trading_volume': 'Client Trading Volume',
        'client_deposit_total': 'Total Client Deposits',
        'dormant_clients_pct': 'Dormant Clients %',
        'client_satisfaction_score': 'Client Satisfaction',

        # Financial
        'commission_current_month': 'Current Month Commission',
        'commission_prev_month': 'Previous Month Commission',
        'commission_3month_avg': '3-Month Avg Commission',
        'commission_trend': 'Commission Trend',
        'commission_volatility': 'Commission Volatility',
        'payout_disputes_count': 'Payout Disputes',
        'payout_delay_incidents': 'Payout Delays',
        'revenue_share_pct': 'Revenue Share %',
        'revenue_share_vs_peers': 'Revenue Share vs Peers',
        'bonus_eligibility_score': 'Bonus Eligibility',
        'incentive_utilization': 'Incentive Utilization',
        'payment_method_changes': 'Payment Method Changes',
        'commission_vs_target': 'Commission vs Target',
        'lifetime_earnings': 'Lifetime Earnings',

        # Relationship
        'tenure_months': 'Partner Tenure',
        'event_attendance_rate': 'Event Attendance',
        'training_completion_rate': 'Training Completion',
        'webinar_attendance': 'Webinar Attendance',
        'rm_meeting_frequency': 'RM Meeting Frequency',
        'rm_responsiveness_score': 'Responsiveness to RM',
        'nps_score': 'NPS Score',
        'referral_count': 'Partner Referrals',

        # Support
        'support_tickets_total': 'Support Tickets',
        'support_tickets_trend': 'Ticket Trend',
        'avg_resolution_time_hours': 'Avg Resolution Time',
        'escalation_count': 'Escalations',
        'escalation_rate': 'Escalation Rate',
        'complaint_count': 'Formal Complaints',
        'first_response_satisfaction': 'First Response Satisfaction',
        'self_service_usage': 'Self-Service Usage',
        'repeated_issues_count': 'Repeated Issues',
        'support_satisfaction_avg': 'Support Satisfaction',

        # Competitive
        'competitor_mentions_count': 'Competitor Mentions',
        'pricing_complaints': 'Pricing Complaints',
        'feature_requests_unmet': 'Unmet Feature Requests',
        'platform_comparison_views': 'Competitor Comparisons',
        'external_link_clicks': 'External Link Clicks',
        'market_share_concern_flag': 'Market Share Concerns',
        'competitive_threat_score': 'Competitive Threat Score',

        # Sentiment
        'sentiment_score_avg': 'Avg Sentiment Score',
        'sentiment_trend': 'Sentiment Trend',
        'frustration_keyword_count': 'Frustration Keywords',
        'urgency_indicator': 'Urgency Level',
        'churn_intent_signals': 'Churn Intent Signals',
        'communication_frequency_change': 'Communication Change',
        'topic_shift_detected': 'Topic Shift Detected',

        # Engineered features
        'engagement_score': 'Overall Engagement Score',
        'revenue_health_score': 'Revenue Health Score',
        'client_health_score': 'Client Health Score',
        'risk_indicator': 'Risk Indicator Score',

        # Categorical
        'tier': 'Partner Tier',
        'partner_type': 'Partner Type',
        'region': 'Region'
    }

    def __init__(self):
        """Initialize the categorizer with feature-to-category mapping."""
        self.feature_to_category = FeatureSchema.get_category_mapping()

    def categorize_drivers(self, drivers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add category information to drivers DataFrame.

        Args:
            drivers_df: DataFrame with 'feature' column

        Returns:
            DataFrame with added category columns
        """
        df = drivers_df.copy()

        # Add category
        df['category'] = df['feature'].map(
            lambda f: self.feature_to_category.get(f, 'Other')
        )

        # Add human-readable labels
        df['category_label'] = df['category'].map(
            lambda c: self.CATEGORY_LABELS.get(c, c)
        )
        df['feature_label'] = df['feature'].map(
            lambda f: self.FEATURE_LABELS.get(f, f.replace('_', ' ').title())
        )

        # Add icon and color
        df['category_icon'] = df['category_label'].map(
            lambda c: self.CATEGORY_ICONS.get(c, '[?]')
        )
        df['category_color'] = df['category_label'].map(
            lambda c: self.CATEGORY_COLORS.get(c, '#757575')
        )

        return df

    def get_category_summary(self, drivers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of drivers grouped by category.

        Args:
            drivers_df: DataFrame with categorized drivers

        Returns:
            DataFrame with category-level aggregations
        """
        if 'category_label' not in drivers_df.columns:
            drivers_df = self.categorize_drivers(drivers_df)

        summary = drivers_df.groupby('category_label').agg({
            'shap_value': ['sum', 'mean', 'count'],
            'feature': 'first'  # Just to get count
        }).reset_index()

        summary.columns = ['category', 'total_impact', 'avg_impact', 'feature_count', '_']
        summary = summary.drop('_', axis=1)

        # Add absolute total for sorting
        summary['abs_total_impact'] = summary['total_impact'].abs()
        summary = summary.sort_values('abs_total_impact', ascending=False)

        # Add direction
        summary['overall_direction'] = summary['total_impact'].apply(
            lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
        )

        # Add icons
        summary['icon'] = summary['category'].map(
            lambda c: self.CATEGORY_ICONS.get(c, '[?]')
        )

        return summary[['category', 'icon', 'total_impact', 'avg_impact',
                        'feature_count', 'overall_direction']]

    def generate_driver_narrative(
            self,
            drivers_df: pd.DataFrame,
            probability: float
    ) -> str:
        """
        Generate a narrative explanation of the top drivers.

        Args:
            drivers_df: DataFrame with categorized drivers
            probability: Churn probability

        Returns:
            Human-readable narrative
        """
        if 'category_label' not in drivers_df.columns:
            drivers_df = self.categorize_drivers(drivers_df)

        # Get category summary
        category_summary = self.get_category_summary(drivers_df)

        # Build narrative
        parts = []

        # Risk assessment
        if probability >= 0.7:
            parts.append("This partner shows strong signs of disengagement.")
        elif probability >= 0.4:
            parts.append("This partner shows moderate warning signs.")
        else:
            parts.append("This partner appears relatively stable.")

        # Top risk categories
        risk_categories = category_summary[
            category_summary['overall_direction'] == 'Increases Risk'
            ].head(2)

        if len(risk_categories) > 0:
            cat_names = ', '.join(risk_categories['category'].tolist())
            parts.append(f"Key risk areas: {cat_names}.")

        # Top driver details
        top_risk = drivers_df[drivers_df['direction'] == 'Increases Risk'].head(2)
        if len(top_risk) > 0:
            driver_details = []
            for _, driver in top_risk.iterrows():
                driver_details.append(
                    f"{driver['feature_label']} ({driver['category_label']})"
                )
            parts.append(f"Primary risk factors: {', '.join(driver_details)}.")

        # Protective factors
        protective = drivers_df[drivers_df['direction'] == 'Decreases Risk'].head(1)
        if len(protective) > 0:
            prot = protective.iloc[0]
            parts.append(
                f"Positive factor: {prot['feature_label']} provides some stability."
            )

        return " ".join(parts)

    def get_intervention_hints(
            self,
            drivers_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Generate intervention hints based on top drivers.

        Args:
            drivers_df: DataFrame with categorized drivers

        Returns:
            List of intervention hint dictionaries
        """
        if 'category_label' not in drivers_df.columns:
            drivers_df = self.categorize_drivers(drivers_df)

        hints = []

        # Map categories to interventions
        category_interventions = {
            'Engagement': {
                'action': 'Schedule personalized platform walkthrough',
                'urgency': 'medium',
                'cost': 'low'
            },
            'Financial': {
                'action': 'Review commission structure and propose incentive',
                'urgency': 'high',
                'cost': 'medium'
            },
            'Client Health': {
                'action': 'Offer client acquisition support and co-marketing',
                'urgency': 'medium',
                'cost': 'medium'
            },
            'Competitive': {
                'action': 'Conduct competitive analysis call, address concerns',
                'urgency': 'high',
                'cost': 'low'
            },
            'Support': {
                'action': 'Escalate to support manager, ensure priority handling',
                'urgency': 'high',
                'cost': 'low'
            },
            'Relationship': {
                'action': 'Schedule relationship review meeting with RM',
                'urgency': 'medium',
                'cost': 'low'
            },
            'Sentiment': {
                'action': 'Personal outreach from senior management',
                'urgency': 'high',
                'cost': 'medium'
            }
        }

        # Get risk-increasing drivers by category
        risk_drivers = drivers_df[drivers_df['direction'] == 'Increases Risk']

        for category in risk_drivers['category_label'].unique():
            if category in category_interventions:
                intervention = category_interventions[category].copy()
                intervention['category'] = category
                intervention['icon'] = self.CATEGORY_ICONS.get(category, '[?]')

                # Add specific driver context
                cat_drivers = risk_drivers[risk_drivers['category_label'] == category]
                intervention['related_factors'] = cat_drivers['feature_label'].tolist()

                hints.append(intervention)

        return hints


if __name__ == "__main__":
    # Test driver categorization
    import numpy as np

    # Create sample driver data
    sample_drivers = pd.DataFrame({
        'feature': ['commission_current_month', 'login_frequency_weekly',
                    'competitor_mentions_count', 'client_retention_rate',
                    'sentiment_score_avg'],
        'value': [5000, 3.5, 5, 65, -0.2],
        'shap_value': [0.15, 0.12, 0.08, -0.05, 0.06],
        'direction': ['Increases Risk', 'Increases Risk', 'Increases Risk',
                      'Decreases Risk', 'Increases Risk']
    })

    categorizer = DriverCategorizer()

    # Categorize drivers
    categorized = categorizer.categorize_drivers(sample_drivers)
    print("Categorized Drivers:")
    print(categorized[['feature_label', 'category_label', 'category_icon', 'shap_value', 'direction']])

    # Get category summary
    print("\n\nCategory Summary:")
    summary = categorizer.get_category_summary(categorized)
    print(summary)

    # Generate narrative
    print("\n\nDriver Narrative:")
    narrative = categorizer.generate_driver_narrative(categorized, 0.72)
    print(narrative)

    # Get intervention hints
    print("\n\nIntervention Hints:")
    hints = categorizer.get_intervention_hints(categorized)
    for hint in hints:
        print(f"\n{hint['icon']} {hint['category']}:")
        print(f"   Action: {hint['action']}")
        print(f"   Urgency: {hint['urgency']}")
        print(f"   Related: {', '.join(hint['related_factors'])}")
