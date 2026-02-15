"""
Prompt templates for the Gemini recommendation engine.
"""
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Represents a prompt template with placeholders."""
    name: str
    system_context: str
    user_template: str
    required_fields: List[str]


class PromptTemplates:
    """
    Collection of prompt templates for different recommendation scenarios.
    """

    # System context that establishes Gemini's role
    BASE_SYSTEM_CONTEXT = """You are a retention strategy advisor for Deriv partnership
management. Analyze partner data and churn risk factors to produce specific,
actionable retention recommendations.

Key principles:
1. Be specific and actionable - avoid generic advice
2. Prioritize recommendations by impact and urgency
3. Consider the partner's tier and type when suggesting interventions
4. Reference actual data points when making recommendations
5. Keep responses concise and professional
6. Focus on retention strategies that have proven effective
7. Consider cost-effectiveness of interventions"""

    # Main recommendation template
    RETENTION_RECOMMENDATION = PromptTemplate(
        name="retention_recommendation",
        system_context=BASE_SYSTEM_CONTEXT,
        user_template="""Analyze this at-risk partner and provide retention recommendations.

PARTNER PROFILE:
- Partner ID: {partner_id}
- Tier: {tier}
- Type: {partner_type}
- Region: {region}
- Tenure: {tenure_months} months

CHURN RISK ASSESSMENT:
- Churn Probability: {churn_probability:.1%}
- Risk Level: {risk_level}

TOP RISK DRIVERS:
{risk_drivers}

RECENT METRICS:
- Monthly Commission: ${commission:,.0f}
- Login Frequency: {login_frequency:.1f}/week
- Active Clients: {active_clients:,}
- Commission Trend: {commission_trend:+.1%}

Please provide:
1. A brief diagnosis (2-3 sentences) of why this partner is at risk
2. Top 3 specific retention actions ranked by priority
3. Suggested talking points for the relationship manager
4. Recommended intervention timeline""",
        required_fields=['partner_id', 'tier', 'partner_type', 'region', 'tenure_months',
                         'churn_probability', 'risk_level', 'risk_drivers', 'commission',
                         'login_frequency', 'active_clients', 'commission_trend']
    )

    # Competitive response template
    COMPETITIVE_RESPONSE = PromptTemplate(
        name="competitive_response",
        system_context=BASE_SYSTEM_CONTEXT + """

Additionally, you have expertise in competitive positioning for forex/trading partnerships.
Focus on differentiating Deriv's value proposition and addressing competitor concerns.""",
        user_template="""A partner is showing competitive concerns. Help craft a response.

PARTNER: {partner_id} ({tier} {partner_type})
CHURN PROBABILITY: {churn_probability:.1%}

COMPETITIVE SIGNALS DETECTED:
{competitive_signals}

COMPETITORS MENTIONED: {competitors_mentioned}

PARTNER'S CURRENT DEAL:
- Revenue Share: {revenue_share}%
- Commission Structure: {commission_structure}
- Support Level: {support_level}

Please provide:
1. Analysis of what's driving the competitive interest
2. Deriv's key differentiators to highlight
3. Specific counter-offer suggestions if appropriate
4. Talking points addressing competitor comparison
5. Risk of losing this partner and recommended urgency""",
        required_fields=['partner_id', 'tier', 'partner_type', 'churn_probability',
                         'competitive_signals', 'competitors_mentioned', 'revenue_share',
                         'commission_structure', 'support_level']
    )

    # Financial intervention template
    FINANCIAL_INTERVENTION = PromptTemplate(
        name="financial_intervention",
        system_context=BASE_SYSTEM_CONTEXT + """

You also understand financial incentive structures and ROI calculations for retention.
Consider the partner's value and recommend cost-effective financial interventions.""",
        user_template="""Recommend financial interventions for this at-risk partner.

PARTNER PROFILE:
- ID: {partner_id}
- Tier: {tier}
- Monthly Value: ${monthly_value:,.0f}
- Annual Value: ${annual_value:,.0f}
- Lifetime Earnings: ${lifetime_earnings:,.0f}

CHURN RISK:
- Probability: {churn_probability:.1%}
- Expected Loss: ${expected_loss:,.0f}/year

FINANCIAL RISK FACTORS:
{financial_risk_factors}

CURRENT DEAL:
- Revenue Share: {revenue_share}%
- Commission vs Target: {commission_vs_target:+.0f}%
- Payout Disputes: {payout_disputes}

AVAILABLE INTERVENTION BUDGET: ${intervention_budget:,.0f}

Recommend:
1. Specific financial incentive with estimated cost
2. Expected ROI of the intervention
3. Alternative non-financial interventions
4. Implementation timeline
5. Success metrics to track""",
        required_fields=['partner_id', 'tier', 'monthly_value', 'annual_value',
                         'lifetime_earnings', 'churn_probability', 'expected_loss',
                         'financial_risk_factors', 'revenue_share', 'commission_vs_target',
                         'payout_disputes', 'intervention_budget']
    )

    # Support escalation template
    SUPPORT_ESCALATION = PromptTemplate(
        name="support_escalation",
        system_context=BASE_SYSTEM_CONTEXT + """

You are also skilled at analyzing support issues and recommending escalation strategies.
Focus on resolving root causes and improving the partner experience.""",
        user_template="""Analyze support issues for this at-risk partner.

PARTNER: {partner_id} ({tier} {partner_type})

SUPPORT METRICS (Last 90 Days):
- Total Tickets: {total_tickets}
- Avg Resolution Time: {resolution_time} hours
- Escalations: {escalations}
- Complaints: {complaints}
- Repeated Issues: {repeated_issues}
- Satisfaction Score: {satisfaction_score}/5

COMMON ISSUE CATEGORIES:
{issue_categories}

SENTIMENT INDICATORS:
- Avg Sentiment: {sentiment_score}
- Frustration Keywords: {frustration_count}

Please provide:
1. Root cause analysis of support dissatisfaction
2. Immediate actions to take (within 48 hours)
3. Systemic improvements to recommend
4. Escalation path if needed
5. Communication template for outreach""",
        required_fields=['partner_id', 'tier', 'partner_type', 'total_tickets',
                         'resolution_time', 'escalations', 'complaints', 'repeated_issues',
                         'satisfaction_score', 'issue_categories', 'sentiment_score',
                         'frustration_count']
    )

    # Cohort analysis template
    COHORT_ANALYSIS = PromptTemplate(
        name="cohort_analysis",
        system_context=BASE_SYSTEM_CONTEXT + """

You can also analyze patterns across groups of partners to identify systemic issues.
Look for common factors and recommend organization-level responses.""",
        user_template="""Analyze this cohort of at-risk partners for common patterns.

COHORT DEFINITION:
- Size: {cohort_size} partners
- Grouping: {grouping_factor}
- Average Churn Probability: {avg_churn_probability:.1%}

COHORT COMPOSITION:
{cohort_composition}

COMMON RISK FACTORS:
{common_risk_factors}

TREND ANALYSIS:
- Partners at risk trend: {risk_trend}
- Commission trend: {commission_trend}
- Engagement trend: {engagement_trend}

Please analyze:
1. Is this a systemic issue or individual partner problems?
2. What's the likely root cause affecting this cohort?
3. Recommended organization-level response
4. Individual interventions still needed
5. Urgency assessment and timeline""",
        required_fields=['cohort_size', 'grouping_factor', 'avg_churn_probability',
                         'cohort_composition', 'common_risk_factors', 'risk_trend',
                         'commission_trend', 'engagement_trend']
    )

    @classmethod
    def get_template(cls, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        templates = {
            'retention_recommendation': cls.RETENTION_RECOMMENDATION,
            'competitive_response': cls.COMPETITIVE_RESPONSE,
            'financial_intervention': cls.FINANCIAL_INTERVENTION,
            'support_escalation': cls.SUPPORT_ESCALATION,
            'cohort_analysis': cls.COHORT_ANALYSIS
        }
        if name not in templates:
            raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")
        return templates[name]

    @classmethod
    def format_template(cls, template_name: str, **kwargs) -> Dict[str, str]:
        """
        Format a template with provided values.

        Args:
            template_name: Name of the template to use
            **kwargs: Values to fill in the template

        Returns:
            Dict with 'system' and 'user' prompts
        """
        template = cls.get_template(template_name)

        # Check required fields
        missing = [f for f in template.required_fields if f not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return {
            'system': template.system_context,
            'user': template.user_template.format(**kwargs)
        }

    @classmethod
    def format_risk_drivers(cls, drivers: List[Dict]) -> str:
        """Format risk drivers for prompt insertion."""
        lines = []
        for i, driver in enumerate(drivers, 1):
            direction = "^" if driver.get('direction') == 'Increases Risk' else "v"
            lines.append(
                f"{i}. {driver['feature_label']}: {driver['value']:.2f} "
                f"({direction} {abs(driver['shap_value']):.3f} impact)"
            )
        return '\n'.join(lines)

    @classmethod
    def format_competitive_signals(cls, signals: List[Dict]) -> str:
        """Format competitive signals for prompt insertion."""
        lines = []
        for signal in signals:
            lines.append(f"- {signal['type']}: {signal['detail']}")
        return '\n'.join(lines) if lines else "No specific signals detected"


# Quick access function
def get_recommendation_prompt(partner_data: Dict, risk_drivers: List[Dict]) -> Dict[str, str]:
    """
    Quick function to generate a retention recommendation prompt.

    Args:
        partner_data: Dict with partner information
        risk_drivers: List of risk driver dicts

    Returns:
        Dict with formatted system and user prompts
    """
    return PromptTemplates.format_template(
        'retention_recommendation',
        partner_id=partner_data.get('partner_id', 'Unknown'),
        tier=partner_data.get('tier', 'Unknown'),
        partner_type=partner_data.get('partner_type', 'Unknown'),
        region=partner_data.get('region', 'Unknown'),
        tenure_months=partner_data.get('tenure_months', 0),
        churn_probability=partner_data.get('churn_probability', 0),
        risk_level=partner_data.get('risk_level', 'Unknown'),
        risk_drivers=PromptTemplates.format_risk_drivers(risk_drivers),
        commission=partner_data.get('commission_current_month', 0),
        login_frequency=partner_data.get('login_frequency_weekly', 0),
        active_clients=partner_data.get('active_clients_count', 0),
        commission_trend=partner_data.get('commission_trend', 0)
    )


if __name__ == "__main__":
    # Test template formatting
    sample_drivers = [
        {'feature_label': 'Commission Trend', 'value': -0.35, 'shap_value': 0.15, 'direction': 'Increases Risk'},
        {'feature_label': 'Login Frequency', 'value': 2.5, 'shap_value': 0.12, 'direction': 'Increases Risk'},
        {'feature_label': 'Client Retention', 'value': 72, 'shap_value': -0.05, 'direction': 'Decreases Risk'}
    ]

    sample_partner = {
        'partner_id': 'P001234',
        'tier': 'gold',
        'partner_type': 'ib',
        'region': 'asia_pacific',
        'tenure_months': 24,
        'churn_probability': 0.72,
        'risk_level': 'high',
        'commission_current_month': 15000,
        'login_frequency_weekly': 2.5,
        'active_clients_count': 150,
        'commission_trend': -0.35
    }

    prompt = get_recommendation_prompt(sample_partner, sample_drivers)
    print("System Prompt:")
    print("-" * 40)
    print(prompt['system'][:200] + "...")
    print("\nUser Prompt:")
    print("-" * 40)
    print(prompt['user'])
