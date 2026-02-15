"""
Intervention tracker for logging and monitoring partner interventions.
Tracks interventions and their outcomes for feedback learning.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class InterventionTracker:
    """
    Track interventions and outcomes to build historical success rate data.
    Core component of the feedback loop system.
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize intervention tracker.

        Args:
            filepath: Path to CSV file for persistence
        """
        if filepath is None:
            project_root = Path(__file__).parent.parent.parent
            filepath = str(project_root / 'data' / 'intervention_outcomes.csv')

        self.filepath = filepath
        self.outcomes = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        """Load existing outcomes or create new DataFrame."""
        try:
            return pd.read_csv(self.filepath)
        except FileNotFoundError:
            return pd.DataFrame(columns=[
                'intervention_id',
                'partner_id',
                'intervention_date',
                'risk_score_at_trigger',
                'risk_tier_at_trigger',
                'primary_shap_driver',
                'driver_category',
                'intervention_type',
                'intervention_cost',
                'outcome_30d',
                'outcome_90d',
                'risk_score_30d',
                'risk_score_90d',
                'commission_change_30d_pct',
                'active_clients_change_30d_pct',
                'notes',
                'created_at',
                'updated_at'
            ])

    def log_intervention(self, intervention_data: Dict) -> str:
        """
        Log a new intervention.

        Args:
            intervention_data: Dict with intervention details

        Returns:
            Intervention ID
        """
        # Generate intervention ID
        intervention_id = f"INT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{intervention_data.get('partner_id', 'UNK')}"

        # Create record
        record = {
            'intervention_id': intervention_id,
            'partner_id': intervention_data.get('partner_id'),
            'intervention_date': intervention_data.get('date', datetime.now().isoformat()),
            'risk_score_at_trigger': intervention_data.get('risk_score'),
            'risk_tier_at_trigger': intervention_data.get('risk_tier'),
            'primary_shap_driver': intervention_data.get('primary_driver'),
            'driver_category': intervention_data.get('driver_category'),
            'intervention_type': intervention_data.get('intervention_type'),
            'intervention_cost': intervention_data.get('cost', 0),
            'outcome_30d': None,
            'outcome_90d': None,
            'risk_score_30d': None,
            'risk_score_90d': None,
            'commission_change_30d_pct': None,
            'active_clients_change_30d_pct': None,
            'notes': intervention_data.get('notes'),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        self.outcomes = pd.concat([
            self.outcomes,
            pd.DataFrame([record])
        ], ignore_index=True)

        self.save()
        return intervention_id

    def update_outcome(
            self,
            intervention_id: str,
            outcome_data: Dict,
            window: str = '30d'
    ) -> bool:
        """
        Update outcome after observation window.

        Args:
            intervention_id: ID of the intervention
            outcome_data: Dict with outcome metrics
            window: '30d' or '90d'

        Returns:
            True if update successful
        """
        mask = self.outcomes['intervention_id'] == intervention_id

        if not mask.any():
            return False

        # Update outcome fields
        outcome_col = f'outcome_{window}'
        risk_col = f'risk_score_{window}'

        self.outcomes.loc[mask, outcome_col] = outcome_data.get('outcome')
        self.outcomes.loc[mask, risk_col] = outcome_data.get('risk_score')

        if window == '30d':
            self.outcomes.loc[mask, 'commission_change_30d_pct'] = outcome_data.get('commission_change_pct')
            self.outcomes.loc[mask, 'active_clients_change_30d_pct'] = outcome_data.get('clients_change_pct')

        self.outcomes.loc[mask, 'updated_at'] = datetime.now().isoformat()

        self.save()
        return True

    def get_partner_history(self, partner_id: str) -> pd.DataFrame:
        """Get all interventions for a partner."""
        return self.outcomes[self.outcomes['partner_id'] == partner_id].copy()

    def get_interventions_by_type(self, intervention_type: str) -> pd.DataFrame:
        """Get all interventions of a specific type."""
        return self.outcomes[self.outcomes['intervention_type'] == intervention_type].copy()

    def get_pending_outcomes(self, window: str = '30d') -> pd.DataFrame:
        """Get interventions awaiting outcome updates."""
        outcome_col = f'outcome_{window}'
        return self.outcomes[self.outcomes[outcome_col].isna()].copy()

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all interventions."""
        if self.outcomes.empty:
            return self._empty_summary()

        completed = self.outcomes[self.outcomes['outcome_30d'].notna()]

        # Calculate success rates
        if len(completed) > 0:
            success_count = completed[completed['outcome_30d'].isin(['Retained', 'Stabilized'])].shape[0]
            success_rate = success_count / len(completed)
        else:
            success_rate = 0

        # Cost analysis
        total_cost = self.outcomes['intervention_cost'].sum()
        avg_cost = self.outcomes['intervention_cost'].mean()

        # By type breakdown
        type_breakdown = {}
        for itype in self.outcomes['intervention_type'].dropna().unique():
            type_data = self.outcomes[self.outcomes['intervention_type'] == itype]
            type_completed = type_data[type_data['outcome_30d'].notna()]
            type_success = type_completed[type_completed['outcome_30d'].isin(['Retained', 'Stabilized'])].shape[0]

            type_breakdown[itype] = {
                'count': len(type_data),
                'completed': len(type_completed),
                'success_rate': type_success / len(type_completed) if len(type_completed) > 0 else 0,
                'total_cost': type_data['intervention_cost'].sum()
            }

        return {
            'total_interventions': len(self.outcomes),
            'completed_interventions': len(completed),
            'pending_outcomes': len(self.outcomes) - len(completed),
            'overall_success_rate': round(success_rate, 3),
            'total_cost': round(total_cost, 2),
            'avg_cost_per_intervention': round(avg_cost, 2),
            'type_breakdown': type_breakdown
        }

    def _empty_summary(self) -> Dict:
        """Return empty summary structure."""
        return {
            'total_interventions': 0,
            'completed_interventions': 0,
            'pending_outcomes': 0,
            'overall_success_rate': 0,
            'total_cost': 0,
            'avg_cost_per_intervention': 0,
            'type_breakdown': {}
        }

    def save(self):
        """Persist outcomes to disk."""
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        self.outcomes.to_csv(self.filepath, index=False)

    def get_dashboard_data(self) -> pd.DataFrame:
        """Get formatted data for dashboard display."""
        if self.outcomes.empty:
            return pd.DataFrame()

        display_cols = [
            'partner_id', 'intervention_date', 'intervention_type',
            'risk_tier_at_trigger', 'outcome_30d', 'intervention_cost'
        ]

        available_cols = [c for c in display_cols if c in self.outcomes.columns]
        return self.outcomes[available_cols].copy()


if __name__ == "__main__":
    # Test intervention tracker
    tracker = InterventionTracker(filepath='temp_interventions.csv')

    print("Intervention Tracker Test")
    print("=" * 60)

    # Log some test interventions
    test_interventions = [
        {
            'partner_id': 'P001',
            'risk_score': 0.78,
            'risk_tier': 'high',
            'primary_driver': 'commission_decline_30d',
            'driver_category': 'Financial',
            'intervention_type': 'dispute_resolution',
            'cost': 150
        },
        {
            'partner_id': 'P002',
            'risk_score': 0.65,
            'risk_tier': 'medium',
            'primary_driver': 'days_since_last_login',
            'driver_category': 'Engagement',
            'intervention_type': 'personal_outreach',
            'cost': 100
        }
    ]

    for intervention in test_interventions:
        int_id = tracker.log_intervention(intervention)
        print(f"Logged intervention: {int_id}")

    # Update an outcome
    tracker.update_outcome(
        tracker.outcomes.iloc[0]['intervention_id'],
        {'outcome': 'Retained', 'risk_score': 0.35, 'commission_change_pct': 15},
        window='30d'
    )

    print("\nSummary Statistics:")
    stats = tracker.get_summary_statistics()
    print(f"  Total: {stats['total_interventions']}")
    print(f"  Completed: {stats['completed_interventions']}")
    print(f"  Success Rate: {stats['overall_success_rate']:.0%}")

    # Cleanup test file
    import os
    os.remove('temp_interventions.csv')
