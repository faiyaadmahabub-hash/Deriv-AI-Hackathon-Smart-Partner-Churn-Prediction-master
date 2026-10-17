"""Small synthetic pipeline and actual dashboard checks; no training or API calls."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = next(ROOT.glob("**/run_demo.py")).parent
sys.path.insert(0, str(SOURCE))

from src.data_generation.generate_data_main import generate_partner_data
from src.preprocessing.preprocess_main import preprocess_data
from src.alerting.risk_tier_assignment import RiskTierAssigner
from src.business_impact.financial_exposure import FinancialExposureCalculator


def main():
    data = generate_partner_data(n_partners=160, seed=42, save_to_file=False)
    assert len(data) == 160 and data.partner_id.is_unique
    assert set(data.churned.unique()) == {0, 1}
    with tempfile.TemporaryDirectory(prefix="churn-check-") as temporary:
        path = Path(temporary) / "synthetic.csv"
        data.to_csv(path, index=False)
        train, test, y_train, y_test, pipeline = preprocess_data(path, save_output=False)
        assert len(train) + len(test) == 160
        assert train.columns.equals(test.columns)
        assert not train.isna().any().any()
        print(f"PASS synthetic preprocessing train={train.shape} test={test.shape}")
    tiers = RiskTierAssigner().assign_tiers([0, .2, .4, .6, .8, 1])
    assert tiers == ["minimal", "low", "medium", "high", "critical", "critical"]
    result = FinancialExposureCalculator().calculate_partner_exposure(1000, .5, "gold", 12)
    assert result["expected_annual_loss"] == 6000
    assert result["value_at_risk"] == 12000
    print("PASS risk thresholds and explicit synthetic exposure arithmetic")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(SOURCE / "dashboard/app.py"), default_timeout=60).run()
    assert not app.exception, [item.message for item in app.exception]
    print(f"PASS dashboard home: {len(app.session_state['data'])} illustrative rows")
    for page in sorted((SOURCE / "dashboard/pages").glob("*.py")):
        app = app.switch_page(f"pages/{page.name}").run()
        assert not app.exception, [(page.name, item.message) for item in app.exception]
        print(f"PASS dashboard page {page.name} (default controls)")
    print("No trained model, SHAP evaluation, Gemini request or real-business validation performed")


if __name__ == "__main__":
    main()
