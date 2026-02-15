"""
Entry point for running the Partner Churn Prediction System demo.
Orchestrates data generation, model training, and dashboard launch.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config


def run_data_generation():
    """Generate synthetic partner data."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Synthetic Partner Data")
    print("=" * 60)

    from src.data_generation.generate_data_main import generate_partner_data
    df = generate_partner_data()
    print(f"\nGenerated {len(df)} partner records.")
    return df


def run_preprocessing():
    """Run preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Preprocessing Pipeline")
    print("=" * 60)

    from src.preprocessing.preprocess_main import preprocess_data
    X_train, X_test, y_train, y_test, pipeline = preprocess_data()
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, pipeline


def run_model_training(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("\n" + "=" * 60)
    print("STEP 3: Training XGBoost Model")
    print("=" * 60)

    from src.modeling.train_xgboost import train_xgboost_model
    predictor, metrics = train_xgboost_model(X_train, y_train, X_test, y_test)
    print(f"\nModel trained. ROC-AUC: {metrics.get('cv_roc_auc_mean', 0):.4f}")
    return predictor, metrics


def run_evaluation(predictor, X_test, y_test):
    """Evaluate model."""
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Model")
    print("=" * 60)

    from src.modeling.evaluate_model import generate_evaluation_report
    report = generate_evaluation_report(predictor, X_test, y_test)
    print(report)
    return {}


def run_dashboard():
    """Launch Streamlit dashboard."""
    print("\n" + "=" * 60)
    print("STEP 5: Launching Dashboard")
    print("=" * 60)

    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    print(f"\nStarting Streamlit dashboard: {dashboard_path}")
    print("Access at: http://localhost:8501")
    print("Press Ctrl+C to stop.\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true"
    ])


def run_demo_analysis():
    """Run demo analysis without dashboard."""
    print("\n" + "=" * 60)
    print("STEP 5: Running Demo Analysis")
    print("=" * 60)

    from src.integration.demo_orchestrator import DemoOrchestrator
    import pandas as pd
    import numpy as np

    # Load data
    data_path = Config.get_data_path('partner_data.csv', processed=False)
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        print("No data found. Generating sample data...")
        np.random.seed(42)
        data = pd.DataFrame({
            'partner_id': [f'P{i:04d}' for i in range(100)],
            'tier': np.random.choice(['platinum', 'gold', 'silver', 'bronze'], 100),
            'region': np.random.choice(list(Config.REGION_DISTRIBUTION.keys()), 100),
            'partner_type': np.random.choice(list(Config.TYPE_DISTRIBUTION.keys()), 100),
            'churn_probability': np.random.beta(2, 5, 100),
            'commission_current_month': np.random.gamma(3, 5000, 100),
            'tenure_months': np.random.randint(1, 60, 100)
        })

    orchestrator = DemoOrchestrator(data=data)

    # Analyze top risk partners
    top_risk = data.nlargest(5, 'churn_probability') if 'churn_probability' in data.columns else data.head(5)

    for _, partner in top_risk.iterrows():
        result = orchestrator.run_prediction_pipeline(partner['partner_id'])
        print(orchestrator.format_demo_output(result))
        print()

    # Portfolio analysis
    print("\n" + "=" * 60)
    print("PORTFOLIO ANALYSIS")
    print("=" * 60)
    portfolio = orchestrator.run_portfolio_analysis()
    print(f"Total Partners: {portfolio.get('total_partners', 0)}")
    print(f"Risk Distribution: {portfolio.get('risk_distribution', {})}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Partner Churn Prediction System Demo")
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['full', 'dashboard', 'train', 'generate', 'analyze'],
        help='Demo mode: full (all steps), dashboard (launch dashboard only), '
             'train (generate + preprocess + train), generate (data only), '
             'analyze (run analysis without dashboard)'
    )
    parser.add_argument('--skip-dashboard', action='store_true', help='Skip dashboard launch in full mode')

    args = parser.parse_args()

    print("=" * 60)
    print("  Partner Churn Prediction System")
    print("  Early Warning and Intervention Platform")
    print("=" * 60)

    Config.ensure_directories()

    if args.mode == 'dashboard':
        run_dashboard()
        return

    if args.mode == 'generate':
        run_data_generation()
        return

    if args.mode in ['full', 'train']:
        # Step 1: Generate data
        run_data_generation()

        # Step 2: Preprocess
        X_train, X_test, y_train, y_test, pipeline = run_preprocessing()

        # Step 3: Train model
        predictor, metrics = run_model_training(X_train, y_train, X_test, y_test)

        # Step 4: Evaluate
        run_evaluation(predictor, X_test, y_test)

        if args.mode == 'train':
            print("\nTraining complete!")
            return

    if args.mode == 'full':
        if args.skip_dashboard:
            run_demo_analysis()
        else:
            run_dashboard()

    if args.mode == 'analyze':
        run_demo_analysis()


if __name__ == "__main__":
    main()
