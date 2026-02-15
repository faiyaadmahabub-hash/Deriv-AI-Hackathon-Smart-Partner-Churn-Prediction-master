"""
Threat clustering for competitive intelligence.
Groups partners by type of competitive threat.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class ThreatClusterer:
    """
    Cluster partners by type of competitive threat.
    Identifies patterns: pricing, features, support, rates.
    """

    def __init__(self):
        """Initialize threat clusterer."""
        self.threat_keywords = Config.THREAT_TYPE_KEYWORDS
        self.threat_types = list(self.threat_keywords.keys())

    def cluster_by_threat(
            self,
            partners: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Cluster partners by primary competitive threat type.

        Args:
            partners: List of partners with competitive signal data

        Returns:
            Dict with threat type as key, list of partners as value
        """
        clusters = {threat: [] for threat in self.threat_types}
        clusters['unknown'] = []

        for partner in partners:
            threat_type = self._identify_primary_threat(partner)
            
            cluster_entry = {
                'partner_id': partner.get('partner_id'),
                'tier': partner.get('tier', partner.get('partner_tier')),
                'region': partner.get('region', partner.get('partner_region')),
                'churn_probability': partner.get('churn_probability', 0),
                'competitor_mentioned': partner.get('primary_competitor', 'Unknown'),
                'threat_indicators': partner.get('threat_types', [])
            }

            clusters[threat_type].append(cluster_entry)

        return clusters

    def _identify_primary_threat(self, partner: Dict) -> str:
        """Identify the primary threat type for a partner."""
        threat_types = partner.get('threat_types', [])

        if not threat_types:
            # Try to infer from feedback or features
            feedback = partner.get('feedback_text', '').lower()
            for threat_type, keywords in self.threat_keywords.items():
                if any(kw in feedback for kw in keywords):
                    return threat_type
            return 'unknown'

        # Return most mentioned threat type
        return threat_types[0]

    def get_cluster_statistics(
            self,
            clusters: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Calculate statistics for each threat cluster.

        Args:
            clusters: Output from cluster_by_threat

        Returns:
            Statistics per cluster
        """
        stats = {}

        for threat_type, partners in clusters.items():
            if not partners:
                continue

            df = pd.DataFrame(partners)

            stats[threat_type] = {
                'partner_count': len(partners),
                'avg_churn_probability': round(df['churn_probability'].mean(), 3) if 'churn_probability' in df else 0,
                'max_churn_probability': round(df['churn_probability'].max(), 3) if 'churn_probability' in df else 0,
                'affected_regions': df['region'].dropna().unique().tolist() if 'region' in df else [],
                'affected_tiers': df['tier'].dropna().unique().tolist() if 'tier' in df else [],
                'top_competitors': self._get_top_competitors(df),
                'urgency': self._calculate_urgency(df)
            }

        return stats

    def _get_top_competitors(self, df: pd.DataFrame) -> List[str]:
        """Get most mentioned competitors in cluster."""
        if 'competitor_mentioned' not in df.columns:
            return []

        competitor_counts = df['competitor_mentioned'].value_counts()
        return competitor_counts.head(3).index.tolist()

    def _calculate_urgency(self, df: pd.DataFrame) -> str:
        """
        Calculate urgency level for cluster.
        
        Returns: 'critical', 'high', 'medium', 'low'
        """
        if df.empty:
            return 'low'

        avg_prob = df['churn_probability'].mean() if 'churn_probability' in df else 0
        count = len(df)

        if avg_prob >= 0.7 and count >= 5:
            return 'critical'
        elif avg_prob >= 0.5 or count >= 10:
            return 'high'
        elif avg_prob >= 0.3 or count >= 5:
            return 'medium'
        return 'low'

    def get_cross_threat_analysis(
            self,
            partners: List[Dict]
    ) -> Dict:
        """
        Analyze partners with multiple threat types.

        Args:
            partners: List of partner data

        Returns:
            Analysis of threat overlaps
        """
        multi_threat_partners = []
        threat_combinations = defaultdict(int)

        for partner in partners:
            threats = partner.get('threat_types', [])
            if len(threats) > 1:
                multi_threat_partners.append({
                    'partner_id': partner.get('partner_id'),
                    'threats': threats,
                    'churn_probability': partner.get('churn_probability', 0)
                })

                # Count combinations
                combo = tuple(sorted(threats[:2]))  # Top 2 threats
                threat_combinations[combo] += 1

        return {
            'multi_threat_partners': len(multi_threat_partners),
            'common_combinations': dict(sorted(
                threat_combinations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'partners': multi_threat_partners[:10]  # Top 10 examples
        }

    def get_threat_summary_dataframe(
            self,
            clusters: Dict[str, List[Dict]]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame for dashboard display.

        Args:
            clusters: Clustered partners

        Returns:
            Summary DataFrame
        """
        stats = self.get_cluster_statistics(clusters)

        records = []
        for threat_type, data in stats.items():
            records.append({
                'Threat Type': threat_type.title(),
                'Partners': data['partner_count'],
                'Avg Risk': f"{data['avg_churn_probability']:.0%}",
                'Urgency': data['urgency'].upper(),
                'Top Competitor': data['top_competitors'][0] if data['top_competitors'] else 'N/A',
                'Regions': ', '.join(data['affected_regions'][:3])
            })

        df = pd.DataFrame(records)
        # Sort by urgency
        urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        df['_sort'] = df['Urgency'].map(urgency_order)
        df = df.sort_values('_sort').drop('_sort', axis=1)

        return df


if __name__ == "__main__":
    # Test threat clusterer
    clusterer = ThreatClusterer()

    print("Threat Clusterer Test")
    print("=" * 60)

    # Sample partners with threat data
    sample_partners = [
        {'partner_id': 'P001', 'tier': 'gold', 'region': 'SEA', 
         'churn_probability': 0.75, 'threat_types': ['pricing'], 
         'primary_competitor': 'Exness'},
        {'partner_id': 'P002', 'tier': 'silver', 'region': 'SEA', 
         'churn_probability': 0.65, 'threat_types': ['pricing', 'rates'],
         'primary_competitor': 'Exness'},
        {'partner_id': 'P003', 'tier': 'gold', 'region': 'MENA', 
         'churn_probability': 0.55, 'threat_types': ['features'],
         'primary_competitor': 'XM'},
        {'partner_id': 'P004', 'tier': 'platinum', 'region': 'Europe', 
         'churn_probability': 0.80, 'threat_types': ['support'],
         'primary_competitor': 'IC Markets'},
        {'partner_id': 'P005', 'tier': 'bronze', 'region': 'Africa', 
         'churn_probability': 0.45, 'threat_types': ['rates'],
         'primary_competitor': 'FXTM'},
    ]

    clusters = clusterer.cluster_by_threat(sample_partners)
    stats = clusterer.get_cluster_statistics(clusters)

    print("\nCluster Statistics:")
    for threat_type, data in stats.items():
        print(f"\n{threat_type.upper()}:")
        print(f"  Partners: {data['partner_count']}")
        print(f"  Avg Risk: {data['avg_churn_probability']:.0%}")
        print(f"  Urgency: {data['urgency']}")

    print("\nSummary DataFrame:")
    summary_df = clusterer.get_threat_summary_dataframe(clusters)
    print(summary_df.to_string(index=False))
