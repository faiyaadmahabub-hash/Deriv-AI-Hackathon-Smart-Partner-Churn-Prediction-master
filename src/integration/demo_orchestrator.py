"""
Demo orchestrator for end-to-end prediction pipeline.
Coordinates all system components for demo and production use.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.feature_schema import FeatureSchema


class DemoOrchestrator:
    """
    Orchestrates the complete demo flow for the Partner Churn Prediction System.
    Coordinates all pipelines: prediction, explanation, business impact, competitive intel.
    """

    def __init__(self, model=None, data: Optional[pd.DataFrame] = None):
        """
        Initialize demo orchestrator.

        Args:
            model: Trained XGBoost model (optional, loads from disk if not provided)
            data: Partner data DataFrame (optional, loads from disk if not provided)
        """
        self.model = model
        self.data = data
        self.config = Config

        # Initialize component modules lazily
        self._shap_calculator = None
        self._exposure_calculator = None
        self._roi_calculator = None
        self._portfolio_aggregator = None
        self._competitor_extractor = None
        self._geographic_mapper = None
        self._intervention_tracker = None

    def _load_components(self):
        """Lazy load component modules."""
        try:
            from src.explainability.shap_calculator import SHAPCalculator
            if self.model is not None:
                feature_names = FeatureSchema.get_model_features()
                self._shap_calculator = SHAPCalculator(self.model, feature_names)
        except (ImportError, Exception):
            pass

        try:
            from src.business_impact.financial_exposure import FinancialExposureCalculator
            from src.business_impact.intervention_roi import InterventionROICalculator
            from src.business_impact.portfolio_aggregation import PortfolioAggregator
            self._exposure_calculator = FinancialExposureCalculator()
            self._roi_calculator = InterventionROICalculator()
            self._portfolio_aggregator = PortfolioAggregator()
        except ImportError:
            pass

        try:
            from src.competitive_intel.competitor_extraction import CompetitorExtractor
            from src.competitive_intel.geographic_mapping import GeographicMapper
            self._competitor_extractor = CompetitorExtractor()
            self._geographic_mapper = GeographicMapper()
        except ImportError:
            pass

        try:
            from src.feedback_loop.intervention_tracker import InterventionTracker
            self._intervention_tracker = InterventionTracker()
        except ImportError:
            pass

    def run_prediction_pipeline(
            self,
            partner_id: str
    ) -> Dict:
        """
        Run complete prediction pipeline for a single partner.

        Args:
            partner_id: Partner ID to analyze

        Returns:
            Complete analysis results
        """
        self._load_components()

        if self.data is None:
            return {'error': 'No data loaded'}

        # Get partner data
        partner_mask = self.data['partner_id'] == partner_id
        if not partner_mask.any():
            return {'error': f'Partner {partner_id} not found'}

        partner_data = self.data[partner_mask].iloc[0].to_dict()

        results = {
            'partner_id': partner_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'partner_info': {
                'tier': partner_data.get('tier', partner_data.get('partner_tier')),
                'region': partner_data.get('region', partner_data.get('partner_region')),
                'type': partner_data.get('partner_type', 'Unknown')
            }
        }

        # Step 1: Churn Prediction
        if self.model is not None:
            try:
                features = self._prepare_features(partner_data)
                churn_prob = float(self.model.predict_proba(features)[0][1])
                results['churn_probability'] = round(churn_prob, 3)
                results['risk_tier'] = self._get_risk_tier(churn_prob)
            except Exception as e:
                results['prediction_error'] = str(e)
                results['churn_probability'] = 0.5
                results['risk_tier'] = 'medium'
        else:
            # Demo mode: use synthetic probability
            results['churn_probability'] = partner_data.get('churn_probability', 0.5)
            results['risk_tier'] = self._get_risk_tier(results['churn_probability'])

        # Step 2: SHAP Explanation
        if self._shap_calculator and self.model:
            try:
                shap_result = self._shap_calculator.explain_partner(
                    self.model,
                    self._prepare_features(partner_data)
                )
                results['shap_explanation'] = shap_result
            except Exception as e:
                results['shap_error'] = str(e)

        # Step 3: Business Impact
        if self._exposure_calculator:
            try:
                commission = partner_data.get('commission_current_month',
                            partner_data.get('commission_usd_30d', 5000))
                tier = partner_data.get('tier', partner_data.get('partner_tier', 'silver'))
                tenure = partner_data.get('tenure_months', 12)

                exposure = self._exposure_calculator.calculate_partner_exposure(
                    monthly_commission=commission,
                    churn_probability=results['churn_probability'],
                    tier=tier,
                    tenure_months=tenure
                )
                results['business_impact'] = exposure
            except Exception as e:
                results['impact_error'] = str(e)

        # Step 4: ROI Analysis
        if self._roi_calculator and 'business_impact' in results:
            try:
                roi = self._roi_calculator.calculate_intervention_roi(
                    partner_value=results['business_impact'].get('annual_value', 0),
                    churn_probability=results['churn_probability'],
                    intervention_type='account_review',
                    tier=results['partner_info'].get('tier', 'silver')
                )
                results['roi_analysis'] = roi
            except Exception as e:
                results['roi_error'] = str(e)

        return results

    def run_portfolio_analysis(self) -> Dict:
        """
        Run portfolio-level analysis across all partners.

        Returns:
            Portfolio analysis results
        """
        self._load_components()

        if self.data is None:
            return {'error': 'No data loaded'}

        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_partners': len(self.data)
        }

        # Run predictions for all partners
        if 'churn_probability' not in self.data.columns:
            if self.model is not None:
                try:
                    features = self._prepare_features_batch(self.data)
                    probs = self.model.predict_proba(features)[:, 1]
                    self.data['churn_probability'] = probs
                except Exception as e:
                    results['prediction_error'] = str(e)
                    self.data['churn_probability'] = 0.5

        # Risk distribution
        high_risk = (self.data['churn_probability'] >= 0.7).sum()
        medium_risk = ((self.data['churn_probability'] >= 0.4) & 
                       (self.data['churn_probability'] < 0.7)).sum()
        low_risk = (self.data['churn_probability'] < 0.4).sum()

        results['risk_distribution'] = {
            'high': int(high_risk),
            'medium': int(medium_risk),
            'low': int(low_risk)
        }

        # Portfolio exposure
        if self._exposure_calculator:
            try:
                portfolio = self._exposure_calculator.calculate_portfolio_exposure(
                    self.data
                )
                results['portfolio_exposure'] = portfolio
            except Exception as e:
                results['exposure_error'] = str(e)

        # Get top 10 at-risk partners
        top_risk = self.data.nlargest(10, 'churn_probability')[
            ['partner_id', 'churn_probability']
        ].to_dict('records')
        results['top_risk_partners'] = top_risk

        return results

    def run_competitive_analysis(self) -> Dict:
        """
        Run competitive intelligence analysis.

        Returns:
            Competitive analysis results
        """
        self._load_components()

        if self.data is None:
            return {'error': 'No data loaded'}

        results = {
            'analysis_timestamp': datetime.now().isoformat()
        }

        # Geographic threat mapping
        if self._geographic_mapper:
            try:
                partners_list = self.data.to_dict('records')
                regional_threats = self._geographic_mapper.map_regional_threats(partners_list)
                results['regional_threats'] = regional_threats

                priority_regions = self._geographic_mapper.get_priority_regions(regional_threats)
                results['priority_regions'] = priority_regions
            except Exception as e:
                results['geographic_error'] = str(e)

        return results

    def _prepare_features(self, partner_data: Dict) -> np.ndarray:
        """Prepare features for single partner prediction."""
        feature_cols = FeatureSchema.get_model_features()
        features = []
        for col in feature_cols:
            features.append(partner_data.get(col, 0))
        return np.array(features).reshape(1, -1)

    def _prepare_features_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for batch prediction."""
        feature_cols = FeatureSchema.get_model_features()
        available_cols = [c for c in feature_cols if c in df.columns]
        return df[available_cols].fillna(0).values

    def _get_risk_tier(self, probability: float) -> str:
        """Convert probability to risk tier."""
        for tier_name in ['critical', 'high', 'medium', 'low']:
            tier_info = self.config.RISK_TIERS.get(tier_name, {})
            if probability >= tier_info.get('min_prob', 1.0):
                return tier_name
        return 'minimal'

    def format_demo_output(self, result: Dict) -> str:
        """
        Format analysis result for demo presentation.

        Args:
            result: Analysis result dict

        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"PARTNER RISK ANALYSIS: {result.get('partner_id', 'Unknown')}")
        lines.append("=" * 60)
        lines.append("")

        # Partner Info
        info = result.get('partner_info', {})
        lines.append(f"Tier: {info.get('tier', 'N/A')} | Region: {info.get('region', 'N/A')}")
        lines.append("")

        # Risk Score
        prob = result.get('churn_probability', 0)
        tier = result.get('risk_tier', 'unknown')
        lines.append(f"CHURN RISK: {prob:.0%} ({tier.upper()} Risk)")
        lines.append("")

        # Business Impact
        impact = result.get('business_impact', {})
        if impact:
            lines.append("FINANCIAL IMPACT:")
            lines.append(f"   - Monthly Value: ${impact.get('monthly_value', 0):,.0f}")
            lines.append(f"   - Value at Risk: ${impact.get('value_at_risk', 0):,.0f}")
            lines.append(f"   - Expected Loss: ${impact.get('expected_annual_loss', 0):,.0f}/year")
            lines.append("")

        # ROI Analysis
        roi = result.get('roi_analysis', {})
        if roi:
            lines.append("INTERVENTION ROI:")
            lines.append(f"   - Recommended: {roi.get('intervention_type', 'N/A')}")
            lines.append(f"   - Cost: ${roi.get('cost', 0):,.0f}")
            lines.append(f"   - Expected ROI: {roi.get('roi', 0):.1f}x")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_demo_partners(self, partner_ids: List[str], filepath: str = None):
        """Save demo partner configurations."""
        if filepath is None:
            project_root = Path(__file__).parent.parent.parent
            filepath = str(project_root / 'demo' / 'demo_partners.json')

        demo_data = []
        for pid in partner_ids:
            result = self.run_prediction_pipeline(pid)
            demo_data.append({
                'partner_id': pid,
                'story_type': self._get_story_type(result),
                'key_metrics': {
                    'risk': result.get('churn_probability'),
                    'tier': result.get('partner_info', {}).get('tier'),
                    'value_at_risk': result.get('business_impact', {}).get('value_at_risk', 0)
                }
            })

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(demo_data, f, indent=2)

    def _get_story_type(self, result: Dict) -> str:
        """Determine story type for demo partner."""
        prob = result.get('churn_probability', 0.5)
        if prob >= 0.8:
            return 'critical_save_opportunity'
        elif prob >= 0.6:
            return 'early_warning'
        elif prob >= 0.4:
            return 'watch_list'
        return 'healthy'


if __name__ == "__main__":
    # Test demo orchestrator
    print("Demo Orchestrator Test")
    print("=" * 60)

    # Create with sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'partner_id': [f'P{i:04d}' for i in range(100)],
        'tier': np.random.choice(['platinum', 'gold', 'silver', 'bronze'], 100),
        'region': np.random.choice(['SEA', 'MENA', 'Europe', 'Africa'], 100),
        'partner_type': np.random.choice(['IB', 'Affiliate'], 100),
        'churn_probability': np.random.beta(2, 5, 100),
        'commission_current_month': np.random.gamma(3, 5000, 100),
        'tenure_months': np.random.randint(1, 60, 100)
    })

    orchestrator = DemoOrchestrator(data=sample_data)

    # Test single partner
    result = orchestrator.run_prediction_pipeline('P0001')
    print("\nSingle Partner Analysis:")
    print(orchestrator.format_demo_output(result))

    # Test portfolio
    portfolio = orchestrator.run_portfolio_analysis()
    print("\nPortfolio Summary:")
    print(f"  Total Partners: {portfolio['total_partners']}")
    print(f"  Risk Distribution: {portfolio['risk_distribution']}")
