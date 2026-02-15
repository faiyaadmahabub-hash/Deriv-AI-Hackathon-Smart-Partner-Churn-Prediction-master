"""
Gemini API integration for generating retention recommendations.
"""
import google.generativeai as genai
from typing import Dict, List, Optional
from pathlib import Path
import sys
import json
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from src.recommendations.prompt_templates import PromptTemplates, get_recommendation_prompt


class GeminiRecommender:
    """
    Gemini-powered recommendation engine for partner retention.
    Generates contextual, actionable recommendations based on risk analysis.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini recommender.

        Args:
            api_key: Gemini API key (defaults to Config value)
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = Config.GEMINI_MODEL
        self.max_tokens = Config.GEMINI_MAX_TOKENS
        self.temperature = Config.GEMINI_TEMPERATURE

        self._configure_api()
        self.model = None
        self._initialize_model()

    def _configure_api(self):
        """Configure the Gemini API."""
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            print("Warning: No Gemini API key configured. Using mock responses.")

    def _initialize_model(self):
        """Initialize the Gemini model."""
        if self.api_key:
            try:
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini model: {e}")
                self.model = None

    def generate_recommendation(
            self,
            partner_data: Dict,
            risk_drivers: List[Dict],
            template_name: str = 'retention_recommendation'
    ) -> Dict:
        """
        Generate a retention recommendation for a partner.

        Args:
            partner_data: Dict with partner information
            risk_drivers: List of risk driver dicts from SHAP analysis
            template_name: Which prompt template to use

        Returns:
            Dict with recommendation details
        """
        # Format the prompt
        try:
            if template_name == 'retention_recommendation':
                prompt = get_recommendation_prompt(partner_data, risk_drivers)
            else:
                prompt = self._build_prompt(template_name, partner_data, risk_drivers)
        except Exception as e:
            return self._error_response(f"Prompt formatting error: {e}")

        # Generate response
        try:
            response = self._call_gemini(prompt)
        except Exception as e:
            return self._error_response(f"API error: {e}")

        # Parse and structure response
        structured = self._parse_response(response, partner_data)

        return structured

    def _build_prompt(
            self,
            template_name: str,
            partner_data: Dict,
            risk_drivers: List[Dict]
    ) -> Dict[str, str]:
        """Build a prompt from template and data."""
        # Add formatted risk drivers to partner data
        partner_data['risk_drivers'] = PromptTemplates.format_risk_drivers(risk_drivers)

        return PromptTemplates.format_template(template_name, **partner_data)

    def _call_gemini(self, prompt: Dict[str, str]) -> str:
        """Call Gemini API with the prompt."""
        if not self.model:
            return self._generate_mock_response(prompt)

        try:
            # Combine system and user prompts
            full_prompt = f"{prompt['system']}\n\n---\n\n{prompt['user']}"

            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'max_output_tokens': self.max_tokens,
                    'temperature': self.temperature
                }
            )

            return response.text

        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: Dict[str, str]) -> str:
        """Generate a mock response when API is unavailable."""
        return """## Diagnosis
This partner shows declining engagement and financial performance, with decreasing login frequency and commission trends indicating potential disengagement.

## Priority Retention Actions

### 1. Immediate Outreach (High Priority)
Schedule a personal call within 48 hours to discuss their business challenges and understand pain points.

### 2. Performance Review (Medium Priority)
Conduct a comprehensive review of their account performance and identify opportunities for growth.

### 3. Incentive Program (Medium Priority)
Offer a customized incentive package based on their tier and historical performance.

## Talking Points for RM
- Acknowledge their recent challenges with empathy
- Highlight new platform features that could benefit their business
- Discuss available support and training resources
- Explore any unaddressed concerns or issues

## Timeline
- Day 1-2: Initial outreach call
- Day 3-7: Performance review completion
- Day 7-14: Present retention package
- Day 30: Follow-up assessment"""

    def _parse_response(self, response: str, partner_data: Dict) -> Dict:
        """Parse the Gemini response into structured format."""
        # Extract sections using simple parsing
        sections = {
            'diagnosis': '',
            'priority_actions': [],
            'talking_points': [],
            'timeline': '',
            'raw_response': response
        }

        # Try to extract diagnosis
        diagnosis_match = re.search(r'##?\s*Diagnosis\s*\n(.*?)(?=##|$)', response, re.DOTALL | re.IGNORECASE)
        if diagnosis_match:
            sections['diagnosis'] = diagnosis_match.group(1).strip()

        # Extract priority actions
        actions_match = re.search(r'##?\s*(?:Priority\s*)?(?:Retention\s*)?Actions?\s*\n(.*?)(?=##|$)', response, re.DOTALL | re.IGNORECASE)
        if actions_match:
            action_text = actions_match.group(1)
            # Parse numbered items
            action_items = re.findall(r'###?\s*\d+\.\s*(.*?)(?=###?\s*\d+\.|$)', action_text, re.DOTALL)
            sections['priority_actions'] = [a.strip() for a in action_items if a.strip()]

        # Extract talking points
        talking_match = re.search(r'##?\s*Talking Points.*?\n(.*?)(?=##|$)', response, re.DOTALL | re.IGNORECASE)
        if talking_match:
            points = re.findall(r'[-*]\s*(.*?)(?=\n[-*]|\n\n|$)', talking_match.group(1), re.DOTALL)
            sections['talking_points'] = [p.strip() for p in points if p.strip()]

        # Extract timeline
        timeline_match = re.search(r'##?\s*Timeline\s*\n(.*?)(?=##|$)', response, re.DOTALL | re.IGNORECASE)
        if timeline_match:
            sections['timeline'] = timeline_match.group(1).strip()

        # Add metadata
        sections['partner_id'] = partner_data.get('partner_id', 'Unknown')
        sections['risk_level'] = partner_data.get('risk_level', 'Unknown')
        sections['churn_probability'] = partner_data.get('churn_probability', 0)

        return sections

    def _error_response(self, error_message: str) -> Dict:
        """Generate an error response."""
        return {
            'error': True,
            'message': error_message,
            'diagnosis': 'Unable to generate recommendation',
            'priority_actions': ['Contact technical support'],
            'talking_points': [],
            'timeline': 'N/A'
        }

    def batch_recommend(
            self,
            partners: List[Dict],
            risk_drivers_map: Dict[str, List[Dict]],
            top_n: int = 10
    ) -> List[Dict]:
        """
        Generate recommendations for multiple partners.

        Args:
            partners: List of partner data dicts
            risk_drivers_map: Dict mapping partner_id to risk drivers
            top_n: Maximum number of recommendations to generate

        Returns:
            List of recommendation dicts
        """
        recommendations = []

        # Sort by churn probability and take top N
        sorted_partners = sorted(
            partners,
            key=lambda p: p.get('churn_probability', 0),
            reverse=True
        )[:top_n]

        for partner in sorted_partners:
            partner_id = partner.get('partner_id', 'Unknown')
            risk_drivers = risk_drivers_map.get(partner_id, [])

            rec = self.generate_recommendation(partner, risk_drivers)
            recommendations.append(rec)

        return recommendations

    def generate_competitive_response(
            self,
            partner_data: Dict,
            competitive_signals: List[Dict]
    ) -> Dict:
        """
        Generate a response to competitive threats.

        Args:
            partner_data: Dict with partner information
            competitive_signals: List of competitive signal dicts

        Returns:
            Dict with competitive response recommendations
        """
        # Format competitive signals
        partner_data['competitive_signals'] = PromptTemplates.format_competitive_signals(competitive_signals)
        partner_data['competitors_mentioned'] = ', '.join(
            [s.get('competitor', 'Unknown') for s in competitive_signals]
        )

        # Fill in defaults for missing fields
        defaults = {
            'revenue_share': 40,
            'commission_structure': 'Standard revenue share',
            'support_level': 'Standard'
        }
        for key, value in defaults.items():
            if key not in partner_data:
                partner_data[key] = value

        try:
            prompt = PromptTemplates.format_template('competitive_response', **partner_data)
            response = self._call_gemini(prompt)
            return self._parse_response(response, partner_data)
        except Exception as e:
            return self._error_response(str(e))

    def generate_cohort_recommendations(
            self,
            cohort_data: Dict,
            common_risk_factors: List[Dict]
    ) -> Dict:
        """
        Generate recommendations for a cohort of at-risk partners.

        Args:
            cohort_data: Dict with cohort summary
            common_risk_factors: Common risk factors across cohort

        Returns:
            Dict with cohort-level recommendations
        """
        cohort_data['common_risk_factors'] = '\n'.join(
            [f"- {f['factor']}: {f['description']}" for f in common_risk_factors]
        )

        try:
            prompt = PromptTemplates.format_template('cohort_analysis', **cohort_data)
            response = self._call_gemini(prompt)
            return self._parse_response(response, cohort_data)
        except Exception as e:
            return self._error_response(str(e))


def get_recommendation_for_partner(
        partner_id: str,
        partner_data: Dict,
        risk_drivers: List[Dict]
) -> Dict:
    """
    Convenience function to get recommendation for a single partner.

    Args:
        partner_id: Partner identifier
        partner_data: Partner data dict
        risk_drivers: SHAP risk drivers

    Returns:
        Recommendation dict
    """
    recommender = GeminiRecommender()
    partner_data['partner_id'] = partner_id
    return recommender.generate_recommendation(partner_data, risk_drivers)


if __name__ == "__main__":
    # Test the recommender
    print("Testing Gemini Recommender")
    print("=" * 50)

    # Sample data
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

    sample_drivers = [
        {'feature_label': 'Commission Trend', 'value': -0.35, 'shap_value': 0.15, 'direction': 'Increases Risk'},
        {'feature_label': 'Login Frequency', 'value': 2.5, 'shap_value': 0.12, 'direction': 'Increases Risk'},
        {'feature_label': 'Competitor Mentions', 'value': 3, 'shap_value': 0.08, 'direction': 'Increases Risk'}
    ]

    # Initialize recommender
    recommender = GeminiRecommender()

    # Generate recommendation
    print("\nGenerating recommendation...")
    rec = recommender.generate_recommendation(sample_partner, sample_drivers)

    print(f"\nPartner: {rec.get('partner_id')}")
    print(f"Risk Level: {rec.get('risk_level')}")
    print(f"\nDiagnosis:\n{rec.get('diagnosis', 'N/A')}")

    print("\nPriority Actions:")
    for i, action in enumerate(rec.get('priority_actions', []), 1):
        print(f"  {i}. {action[:100]}...")

    print("\nTalking Points:")
    for point in rec.get('talking_points', []):
        print(f"  - {point}")

    print(f"\nTimeline:\n{rec.get('timeline', 'N/A')}")
