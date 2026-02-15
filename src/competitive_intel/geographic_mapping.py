"""
Geographic mapping for competitive threats.
Identifies which competitors threaten which regions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class GeographicMapper:
    """
    Map competitive threats by geographic region.
    Identify regional patterns and dominant competitors.
    """

    def __init__(self):
        """Initialize geographic mapper."""
        self.regions = Config.REGIONS
        self.threat_levels = {
            'critical': {'min_partners': 5, 'min_penetration': 0.4},
            'high': {'min_partners': 3, 'min_penetration': 0.25},
            'medium': {'min_partners': 2, 'min_penetration': 0.15},
            'low': {'min_partners': 1, 'min_penetration': 0.0}
        }

    def map_regional_threats(
            self,
            partners: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Map competitive threats by region.

        Args:
            partners: List of partners with competitor data

        Returns:
            Regional threat intelligence
        """
        region_competitor_map = defaultdict(lambda: defaultdict(int))
        region_partner_count = defaultdict(int)
        region_risk_sum = defaultdict(float)

        for partner in partners:
            region = partner.get('region', partner.get('partner_region', 'Unknown'))
            competitor = partner.get('primary_competitor', partner.get('competitor_mentioned'))
            churn_prob = partner.get('churn_probability', 0)

            region_partner_count[region] += 1
            region_risk_sum[region] += churn_prob

            if competitor and competitor != 'None' and competitor != 'Unknown':
                region_competitor_map[region][competitor] += 1

        # Build regional threat profiles
        regional_threats = {}

        for region in set(region_partner_count.keys()):
            competitors = region_competitor_map.get(region, {})
            total_in_region = region_partner_count[region]
            affected_count = sum(competitors.values())

            if not competitors:
                continue

            # Find dominant competitor
            dominant = max(competitors.items(), key=lambda x: x[1])

            # Calculate threat level
            threat_level = self._calculate_threat_level(
                dominant[1],
                total_in_region,
                region_risk_sum[region] / total_in_region if total_in_region > 0 else 0
            )

            regional_threats[region] = {
                'top_competitor': dominant[0],
                'top_competitor_mentions': dominant[1],
                'affected_partners': affected_count,
                'total_partners_in_region': total_in_region,
                'penetration_rate': round(affected_count / total_in_region, 2) if total_in_region > 0 else 0,
                'avg_churn_risk': round(region_risk_sum[region] / total_in_region, 3) if total_in_region > 0 else 0,
                'threat_level': threat_level,
                'competitor_breakdown': dict(competitors),
                'requires_action': threat_level in ['critical', 'high']
            }

        return regional_threats

    def _calculate_threat_level(
            self,
            affected_count: int,
            total_count: int,
            avg_risk: float
    ) -> str:
        """Calculate threat level based on metrics."""
        penetration = affected_count / total_count if total_count > 0 else 0

        # Check thresholds
        for level in ['critical', 'high', 'medium', 'low']:
            thresholds = self.threat_levels[level]
            if (affected_count >= thresholds['min_partners'] and
                penetration >= thresholds['min_penetration']):
                # Elevate if average risk is high
                if avg_risk > 0.6 and level != 'critical':
                    level_order = ['low', 'medium', 'high', 'critical']
                    idx = level_order.index(level)
                    return level_order[min(idx + 1, 3)]
                return level

        return 'low'

    def get_threat_heatmap_data(
            self,
            regional_threats: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Prepare data for threat heatmap visualization.

        Args:
            regional_threats: Output from map_regional_threats

        Returns:
            DataFrame suitable for heatmap
        """
        records = []

        for region, data in regional_threats.items():
            for competitor, count in data['competitor_breakdown'].items():
                records.append({
                    'region': region,
                    'competitor': competitor,
                    'mentions': count,
                    'threat_level': data['threat_level'],
                    'penetration': data['penetration_rate']
                })

        if not records:
            return pd.DataFrame(columns=['region', 'competitor', 'mentions', 'threat_level', 'penetration'])

        df = pd.DataFrame(records)
        return df.sort_values(['region', 'mentions'], ascending=[True, False])

    def get_priority_regions(
            self,
            regional_threats: Dict[str, Dict],
            limit: int = 5
    ) -> List[Dict]:
        """
        Get regions requiring immediate attention.

        Args:
            regional_threats: Regional threat data
            limit: Maximum regions to return

        Returns:
            List of priority regions with action items
        """
        priority_regions = []

        for region, data in regional_threats.items():
            if not data['requires_action']:
                continue

            priority_regions.append({
                'region': region,
                'threat_level': data['threat_level'],
                'top_competitor': data['top_competitor'],
                'affected_partners': data['affected_partners'],
                'avg_risk': data['avg_churn_risk'],
                'recommended_action': self._get_recommended_action(data)
            })

        # Sort by severity
        level_priority = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        priority_regions.sort(key=lambda x: (level_priority.get(x['threat_level'], 4), -x['affected_partners']))

        return priority_regions[:limit]

    def _get_recommended_action(self, region_data: Dict) -> str:
        """Generate recommended action for region."""
        threat_level = region_data['threat_level']
        competitor = region_data['top_competitor']

        actions = {
            'critical': f"Immediate rate review and retention campaign needed to counter {competitor}",
            'high': f"Schedule partner calls to understand {competitor} appeal; consider regional incentives",
            'medium': f"Monitor {competitor} activity; prepare counter-offers",
            'low': f"Track competitive mentions; no immediate action required"
        }

        return actions.get(threat_level, "Continue monitoring")

    def get_regional_summary_df(
            self,
            regional_threats: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame for dashboard.

        Args:
            regional_threats: Regional threat data

        Returns:
            Summary DataFrame
        """
        records = []

        for region, data in regional_threats.items():
            records.append({
                'Region': region,
                'Top Threat': data['top_competitor'],
                'Partners Affected': data['affected_partners'],
                'Total Partners': data['total_partners_in_region'],
                'Penetration': f"{data['penetration_rate']:.0%}",
                'Avg Risk': f"{data['avg_churn_risk']:.0%}",
                'Threat Level': data['threat_level'].upper(),
                'Action Needed': 'YES' if data['requires_action'] else 'NO'
            })

        df = pd.DataFrame(records)
        if not df.empty:
            # Sort by threat level
            level_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            df['_sort'] = df['Threat Level'].map(level_order)
            df = df.sort_values('_sort').drop('_sort', axis=1)

        return df


if __name__ == "__main__":
    # Test geographic mapper
    mapper = GeographicMapper()

    print("Geographic Mapper Test")
    print("=" * 60)

    # Sample partners
    sample_partners = [
        {'partner_id': 'P001', 'region': 'SEA', 'primary_competitor': 'Exness', 'churn_probability': 0.75},
        {'partner_id': 'P002', 'region': 'SEA', 'primary_competitor': 'Exness', 'churn_probability': 0.68},
        {'partner_id': 'P003', 'region': 'SEA', 'primary_competitor': 'Exness', 'churn_probability': 0.72},
        {'partner_id': 'P004', 'region': 'SEA', 'primary_competitor': 'XM', 'churn_probability': 0.55},
        {'partner_id': 'P005', 'region': 'SEA', 'primary_competitor': None, 'churn_probability': 0.30},
        {'partner_id': 'P006', 'region': 'MENA', 'primary_competitor': 'XM', 'churn_probability': 0.60},
        {'partner_id': 'P007', 'region': 'MENA', 'primary_competitor': 'FXTM', 'churn_probability': 0.45},
        {'partner_id': 'P008', 'region': 'Europe', 'primary_competitor': 'IC Markets', 'churn_probability': 0.50},
        {'partner_id': 'P009', 'region': 'Europe', 'primary_competitor': 'Pepperstone', 'churn_probability': 0.42},
        {'partner_id': 'P010', 'region': 'Africa', 'primary_competitor': None, 'churn_probability': 0.35},
    ]

    regional_threats = mapper.map_regional_threats(sample_partners)

    print("\nRegional Threat Analysis:")
    for region, data in regional_threats.items():
        print(f"\n{region}:")
        print(f"  Top Competitor: {data['top_competitor']}")
        print(f"  Affected: {data['affected_partners']}/{data['total_partners_in_region']}")
        print(f"  Threat Level: {data['threat_level'].upper()}")
        print(f"  Action: {data['requires_action']}")

    print("\n\nPriority Regions:")
    for region in mapper.get_priority_regions(regional_threats):
        print(f"  {region['region']}: {region['recommended_action']}")

    print("\n\nSummary DataFrame:")
    summary = mapper.get_regional_summary_df(regional_threats)
    print(summary.to_string(index=False))
