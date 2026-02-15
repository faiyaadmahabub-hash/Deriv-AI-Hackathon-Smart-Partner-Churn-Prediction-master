"""
Scenario modeling for intervention planning.
Models "What if we save X partners?" scenarios.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


class ScenarioModeler:
    """
    Model intervention scenarios and optimize partner selection by ROI.
    Answers "If we have $X budget, which partners should we prioritize?"
    """

    def __init__(self):
        """Initialize scenario modeler."""
        self.intervention_costs = Config.INTERVENTION_COSTS
        self.success_rates = Config.INTERVENTION_SUCCESS_RATES

    def model_scenarios(
            self,
            partner_data: List[Dict],
            intervention_budget: float,
            default_intervention_cost: float = 500
    ) -> List[Dict]:
        """
        Model scenarios with different intervention counts.

        Args:
            partner_data: List with exposure + ROI data per partner
            intervention_budget: Available budget for interventions
            default_intervention_cost: Default cost if not specified

        Returns:
            List of scenarios with different partner counts
        """
        if not partner_data:
            return []

        # Calculate priority score for each partner
        scored_partners = self._score_partners(partner_data, default_intervention_cost)

        # Sort by priority (highest value / cost ratio first)
        sorted_partners = sorted(
            scored_partners,
            key=lambda p: p['priority_score'],
            reverse=True
        )

        scenarios = []
        cumulative_cost = 0
        cumulative_value_saved = 0
        cumulative_expected_value = 0
        partners_interventioned = 0

        # Track which partners are included
        included_partners = []

        for partner in sorted_partners:
            intervention_cost = partner.get('intervention_cost', default_intervention_cost)

            # Stop if budget exceeded
            if cumulative_cost + intervention_cost > intervention_budget:
                break

            cumulative_cost += intervention_cost
            cumulative_value_saved += partner.get('value_at_risk', 0)
            cumulative_expected_value += partner.get('expected_value_saved', 0)
            partners_interventioned += 1
            included_partners.append(partner['partner_id'])

            # Record scenario at milestones
            if partners_interventioned in [1, 3, 5, 8, 10, 15, 20] or \
               partners_interventioned == len(sorted_partners):
                net_benefit = cumulative_expected_value - cumulative_cost
                roi = net_benefit / cumulative_cost if cumulative_cost > 0 else 0

                scenarios.append({
                    'partners_saved': partners_interventioned,
                    'total_cost': round(cumulative_cost, 2),
                    'total_value_at_risk': round(cumulative_value_saved, 2),
                    'expected_value_saved': round(cumulative_expected_value, 2),
                    'net_benefit': round(net_benefit, 2),
                    'roi': round(roi, 2),
                    'remaining_budget': round(intervention_budget - cumulative_cost, 2),
                    'budget_utilization': round(cumulative_cost / intervention_budget * 100, 1),
                    'partners_list': included_partners.copy()
                })

        return scenarios

    def _score_partners(
            self,
            partner_data: List[Dict],
            default_cost: float
    ) -> List[Dict]:
        """
        Calculate priority score for each partner.

        Priority = (value_at_risk * success_probability) / intervention_cost
        """
        scored = []

        for partner in partner_data:
            value_at_risk = partner.get('value_at_risk', 0)
            success_rate = partner.get('success_rate', 0.5)
            intervention_cost = partner.get('intervention_cost', default_cost)

            # Expected value saved = value * probability of successful intervention
            expected_value = value_at_risk * success_rate

            # Priority = expected value per dollar spent
            priority = expected_value / intervention_cost if intervention_cost > 0 else 0

            scored.append({
                **partner,
                'expected_value_saved': expected_value,
                'intervention_cost': intervention_cost,
                'priority_score': priority
            })

        return scored

    def get_optimal_budget_allocation(
            self,
            partner_data: List[Dict],
            total_budget: float,
            budget_increments: int = 10
    ) -> List[Dict]:
        """
        Find optimal budget allocation across different budget levels.

        Args:
            partner_data: Partner data with exposure info
            total_budget: Maximum available budget
            budget_increments: Number of budget levels to analyze

        Returns:
            List of optimal allocations at different budget levels
        """
        allocations = []
        increment = total_budget / budget_increments

        for i in range(1, budget_increments + 1):
            budget = increment * i
            scenarios = self.model_scenarios(partner_data, budget)

            if scenarios:
                # Get the last (best) scenario for this budget
                best = scenarios[-1]
                allocations.append({
                    'budget_level': round(budget, 2),
                    'partners_covered': best['partners_saved'],
                    'expected_return': best['expected_value_saved'],
                    'roi': best['roi'],
                    'efficiency': round(best['expected_value_saved'] / budget, 2) if budget > 0 else 0
                })

        return allocations

    def compare_intervention_strategies(
            self,
            partner_data: List[Dict],
            budget: float
    ) -> Dict:
        """
        Compare different intervention strategies.

        Strategies:
        1. ROI-optimized (default)
        2. Highest value first
        3. Highest probability first
        4. Equal distribution

        Args:
            partner_data: Partner data
            budget: Available budget

        Returns:
            Comparison of strategies
        """
        strategies = {}

        # Strategy 1: ROI-optimized (our default)
        roi_scenarios = self.model_scenarios(partner_data, budget)
        if roi_scenarios:
            strategies['roi_optimized'] = roi_scenarios[-1]

        # Strategy 2: Highest value first
        value_sorted = sorted(partner_data, key=lambda p: p.get('value_at_risk', 0), reverse=True)
        value_scenario = self._run_strategy(value_sorted, budget)
        strategies['highest_value_first'] = value_scenario

        # Strategy 3: Highest probability first
        prob_sorted = sorted(partner_data, key=lambda p: p.get('churn_probability', 0), reverse=True)
        prob_scenario = self._run_strategy(prob_sorted, budget)
        strategies['highest_risk_first'] = prob_scenario

        # Determine winner
        best_strategy = max(strategies.items(), key=lambda s: s[1].get('net_benefit', 0))

        return {
            'strategies': strategies,
            'recommended': best_strategy[0],
            'recommended_net_benefit': best_strategy[1].get('net_benefit', 0)
        }

    def _run_strategy(
            self,
            sorted_partners: List[Dict],
            budget: float,
            default_cost: float = 500
    ) -> Dict:
        """Run a strategy and return the final scenario."""
        cumulative_cost = 0
        cumulative_value = 0
        cumulative_expected = 0
        count = 0

        for partner in sorted_partners:
            cost = partner.get('intervention_cost', default_cost)
            if cumulative_cost + cost > budget:
                break

            cumulative_cost += cost
            value_at_risk = partner.get('value_at_risk', 0)
            success_rate = partner.get('success_rate', 0.5)

            cumulative_value += value_at_risk
            cumulative_expected += value_at_risk * success_rate
            count += 1

        net_benefit = cumulative_expected - cumulative_cost

        return {
            'partners_saved': count,
            'total_cost': round(cumulative_cost, 2),
            'total_value_at_risk': round(cumulative_value, 2),
            'expected_value_saved': round(cumulative_expected, 2),
            'net_benefit': round(net_benefit, 2),
            'roi': round(net_benefit / cumulative_cost, 2) if cumulative_cost > 0 else 0
        }


if __name__ == "__main__":
    # Test scenario modeler
    modeler = ScenarioModeler()

    print("Scenario Modeler Test")
    print("=" * 60)

    # Create sample partner data
    np.random.seed(42)
    sample_partners = []
    for i in range(30):
        value = np.random.gamma(3, 10000)
        prob = np.random.beta(3, 2)
        sample_partners.append({
            'partner_id': f'P{i:04d}',
            'tier': np.random.choice(['platinum', 'gold', 'silver', 'bronze']),
            'value_at_risk': value,
            'churn_probability': prob,
            'success_rate': np.random.uniform(0.4, 0.8),
            'intervention_cost': np.random.choice([300, 500, 800, 1200])
        })

    # Test scenarios
    budget = 10000
    scenarios = modeler.model_scenarios(sample_partners, budget)

    print(f"\nScenarios with ${budget:,} budget:\n")
    print(f"{'Partners':<10} {'Cost':<12} {'Expected Value':<15} {'ROI':<8} {'Budget Used'}")
    print("-" * 60)
    for s in scenarios:
        print(f"{s['partners_saved']:<10} ${s['total_cost']:<10,.0f} "
              f"${s['expected_value_saved']:<13,.0f} {s['roi']:<8.2f} "
              f"{s['budget_utilization']}%")

    # Test strategy comparison
    print("\n\nStrategy Comparison:")
    print("-" * 40)
    comparison = modeler.compare_intervention_strategies(sample_partners, budget)
    for strategy, result in comparison['strategies'].items():
        print(f"{strategy}: ROI={result['roi']:.2f}, Net=${result['net_benefit']:,.0f}")
    print(f"\nRecommended: {comparison['recommended']}")
