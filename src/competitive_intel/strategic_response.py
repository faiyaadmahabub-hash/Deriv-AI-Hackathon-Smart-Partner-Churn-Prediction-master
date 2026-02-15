"""
Strategic response generator for competitive threats.
Uses Gemini to generate counter-move recommendations.
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config

# Try to import Gemini, but provide fallback for offline testing
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class StrategicResponder:
    """
    Generate strategic recommendations for competitive threats.
    Uses Gemini for AI-powered counter-move suggestions.
    """

    def __init__(self):
        """Initialize strategic responder."""
        self.api_key = Config.GEMINI_API_KEY
        self.model = None

        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")

    def generate_strategic_recommendations(
            self,
            regional_threats: Dict[str, Dict],
            segment_vulnerability: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate strategic recommendations for high-threat regions.

        Args:
            regional_threats: Output from GeographicMapper
            segment_vulnerability: Optional segment analysis data

        Returns:
            List of strategic recommendations
        """
        recommendations = []

        # Focus on HIGH and CRITICAL threat regions
        priority_regions = [
            (region, data) for region, data in regional_threats.items()
            if data.get('threat_level') in ['critical', 'high']
        ]

        for region, threat_data in priority_regions:
            if self.model:
                rec = self._generate_gemini_recommendation(region, threat_data)
            else:
                rec = self._generate_fallback_recommendation(region, threat_data)

            recommendations.append(rec)

        return recommendations

    def _generate_gemini_recommendation(
            self,
            region: str,
            threat_data: Dict
    ) -> Dict:
        """Generate recommendation using Gemini."""
        competitor = threat_data['top_competitor']
        affected_count = threat_data['affected_partners']
        penetration = threat_data['penetration_rate']
        avg_risk = threat_data.get('avg_churn_risk', 0)

        prompt = f"""You are a strategic analyst for Deriv, a trading platform with a partner program.

Context:
- Region: {region}
- Primary Competitor Threat: {competitor}
- Affected partners: {affected_count}
- Competitor penetration rate: {penetration:.0%}
- Average churn risk of affected partners: {avg_risk:.0%}
- Threat level: {threat_data['threat_level'].upper()}

Based on common trading platform competitive dynamics:

1. What 2-3 specific strategic counter-moves should Deriv implement to retain partners in {region}?
2. For each counter-move, provide:
   - Specific action description
   - Implementation cost estimate (Low: <$20K, Medium: $20-100K, High: >$100K)
   - Expected retention impact (Low: <10%, Medium: 10-25%, High: >25%)
   - Implementation timeline (Immediate: <1 week, Short-term: 1-4 weeks, Long-term: >1 month)

Focus on: commission rates, platform features, support quality, payment terms, partner benefits.

Be specific and actionable. Format as:

**Counter-move 1: [Title]**
Action: [specific steps]
Cost: [Low/Medium/High]
Impact: [Low/Medium/High]
Timeline: [Immediate/Short-term/Long-term]

**Counter-move 2: [Title]**
..."""

        try:
            response = self.model.generate_content(prompt)
            recommendations_text = response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_fallback_recommendation(region, threat_data)

        return {
            'region': region,
            'threat_competitor': competitor,
            'threat_level': threat_data['threat_level'],
            'affected_partners': affected_count,
            'penetration_rate': penetration,
            'recommendations_text': recommendations_text,
            'priority': 'URGENT' if threat_data['threat_level'] == 'critical' else 'HIGH',
            'generated_at': datetime.now().isoformat(),
            'source': 'gemini'
        }

    def _generate_fallback_recommendation(
            self,
            region: str,
            threat_data: Dict
    ) -> Dict:
        """Generate rule-based recommendation when Gemini unavailable."""
        competitor = threat_data['top_competitor']
        affected_count = threat_data['affected_partners']

        # Rule-based recommendations by competitor
        competitor_strategies = {
            'Exness': {
                'title': 'Rate Competitiveness Review',
                'action': 'Conduct rate comparison and offer 5-10% commission boost for retention',
                'cost': 'Medium',
                'impact': 'High'
            },
            'XM': {
                'title': 'Platform Feature Enhancement',
                'action': 'Accelerate platform improvements and offer exclusive beta access',
                'cost': 'High',
                'impact': 'Medium'
            },
            'IC Markets': {
                'title': 'Professional Support Upgrade',
                'action': 'Assign dedicated account managers for at-risk partners',
                'cost': 'Medium',
                'impact': 'High'
            },
            'Pepperstone': {
                'title': 'Payment Term Optimization',
                'action': 'Offer faster payout cycles and improved payment methods',
                'cost': 'Low',
                'impact': 'Medium'
            },
            'FXTM': {
                'title': 'Emerging Market Focus',
                'action': 'Enhance local currency support and regional promotions',
                'cost': 'Medium',
                'impact': 'High'
            }
        }

        strategy = competitor_strategies.get(competitor, {
            'title': 'Partner Retention Campaign',
            'action': 'Proactive outreach with personalized retention offers',
            'cost': 'Low',
            'impact': 'Medium'
        })

        recommendations_text = f"""**Counter-move 1: {strategy['title']}**
Action: {strategy['action']}
Cost: {strategy['cost']}
Impact: {strategy['impact']}
Timeline: Short-term

**Counter-move 2: Personal Outreach Campaign**
Action: Schedule calls with all {affected_count} affected partners to understand concerns
Cost: Low
Impact: Medium
Timeline: Immediate

**Counter-move 3: Competitive Intelligence Monitoring**
Action: Set up systematic tracking of {competitor} activities in {region}
Cost: Low
Impact: Low (preventive)
Timeline: Immediate"""

        return {
            'region': region,
            'threat_competitor': competitor,
            'threat_level': threat_data['threat_level'],
            'affected_partners': affected_count,
            'penetration_rate': threat_data['penetration_rate'],
            'recommendations_text': recommendations_text,
            'priority': 'URGENT' if threat_data['threat_level'] == 'critical' else 'HIGH',
            'generated_at': datetime.now().isoformat(),
            'source': 'rule_based'
        }

    def get_recommendation_summary(
            self,
            recommendations: List[Dict]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame for dashboard display.

        Args:
            recommendations: List of recommendation dicts

        Returns:
            Summary DataFrame
        """
        records = []

        for rec in recommendations:
            records.append({
                'Region': rec['region'],
                'Competitor': rec['threat_competitor'],
                'Threat Level': rec['threat_level'].upper(),
                'Partners': rec['affected_partners'],
                'Priority': rec['priority'],
                'Source': rec['source'].title()
            })

        df = pd.DataFrame(records)
        if not df.empty:
            priority_order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2}
            df['_sort'] = df['Priority'].map(priority_order)
            df = df.sort_values('_sort').drop('_sort', axis=1)

        return df


if __name__ == "__main__":
    # Test strategic responder
    responder = StrategicResponder()

    print("Strategic Responder Test")
    print("=" * 60)

    # Sample regional threats
    sample_threats = {
        'SEA': {
            'top_competitor': 'Exness',
            'affected_partners': 8,
            'penetration_rate': 0.35,
            'avg_churn_risk': 0.72,
            'threat_level': 'critical'
        },
        'MENA': {
            'top_competitor': 'XM',
            'affected_partners': 4,
            'penetration_rate': 0.25,
            'avg_churn_risk': 0.55,
            'threat_level': 'high'
        }
    }

    recommendations = responder.generate_strategic_recommendations(sample_threats)

    print(f"\nGenerated {len(recommendations)} strategic recommendations:\n")

    for rec in recommendations:
        print(f"\n{'='*50}")
        print(f"Region: {rec['region']}")
        print(f"Threat: {rec['threat_competitor']} ({rec['threat_level'].upper()})")
        print(f"Priority: {rec['priority']}")
        print(f"Source: {rec['source']}")
        print(f"\nRecommendations:\n{rec['recommendations_text']}")
