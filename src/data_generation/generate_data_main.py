"""
Main data generation orchestrator.
Combines base feature generation with decline cascade application.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.feature_schema import FeatureSchema
from src.data_generation.generate_base_features import generate_base_features
from src.data_generation.apply_decline_cascade import (
    assign_churn_labels,
    apply_decline_cascade,
    add_temporal_features
)


def generate_partner_data(
        n_partners: int = None,
        churn_rate: float = None,
        seed: int = None,
        save_to_file: bool = True,
        include_temporal: bool = True
) -> pd.DataFrame:
    """
    Generate complete synthetic partner dataset with churn patterns.

    Args:
        n_partners: Number of partners to generate (default: Config.N_PARTNERS)
        churn_rate: Target churn rate (default: Config.CHURN_RATE)
        seed: Random seed for reproducibility (default: Config.RANDOM_SEED)
        save_to_file: Whether to save to CSV file
        include_temporal: Whether to include temporal features

    Returns:
        DataFrame with all partner features and churn labels
    """
    n_partners = n_partners or Config.N_PARTNERS
    churn_rate = churn_rate or Config.CHURN_RATE
    seed = seed or Config.RANDOM_SEED

    print(f"Generating {n_partners} synthetic partners...")

    # Step 1: Generate base features
    print("  Step 1/4: Generating base features...")
    df = generate_base_features(n_partners=n_partners, seed=seed)

    # Step 2: Assign churn labels
    print("  Step 2/4: Assigning churn labels...")
    df = assign_churn_labels(df, target_churn_rate=churn_rate, seed=seed)

    # Step 3: Apply decline cascade to churning partners
    print("  Step 3/4: Applying decline cascade...")
    df = apply_decline_cascade(df, seed=seed)

    # Step 4: Add temporal features
    if include_temporal:
        print("  Step 4/4: Adding temporal features...")
        df = add_temporal_features(df)

    # Validate the generated data
    print("\nValidating generated data...")
    validation_report = validate_generated_data(df)

    if validation_report['is_valid']:
        print("  Data validation passed!")
    else:
        print(f"  Data validation warnings: {validation_report['warnings']}")

    # Save to file if requested
    if save_to_file:
        Config.ensure_directories()
        output_path = Config.get_data_path('partner_data.csv', processed=False)
        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")

    # Print summary statistics
    print_data_summary(df)

    return df


def validate_generated_data(df: pd.DataFrame) -> dict:
    """Validate the generated dataset against expected criteria."""
    report = {
        'is_valid': True,
        'warnings': [],
        'stats': {}
    }

    # Check churn rate
    actual_churn_rate = df['churned'].mean()
    if abs(actual_churn_rate - Config.CHURN_RATE) > 0.03:
        report['warnings'].append(
            f"Churn rate ({actual_churn_rate:.2%}) differs from target ({Config.CHURN_RATE:.2%})"
        )
    report['stats']['churn_rate'] = actual_churn_rate

    # Check feature count
    expected_features = FeatureSchema.get_feature_count()
    # Account for partner_id, churned, churn_probability, days_until_churn
    actual_features = len(df.columns) - 4
    if actual_features < expected_features - 5:
        report['warnings'].append(
            f"Feature count ({actual_features}) is less than expected ({expected_features})"
        )
    report['stats']['feature_count'] = actual_features

    # Check for null values
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if len(columns_with_nulls) > 1:  # days_until_churn can have nulls
        report['warnings'].append(
            f"Columns with unexpected nulls: {list(columns_with_nulls.index)}"
        )

    # Check tier distribution
    tier_dist = df['tier'].value_counts(normalize=True)
    report['stats']['tier_distribution'] = tier_dist.to_dict()

    # Check separation between churned and non-churned
    churned_login = df[df['churned'] == 1]['login_frequency_weekly'].mean()
    non_churned_login = df[df['churned'] == 0]['login_frequency_weekly'].mean()
    if churned_login >= non_churned_login:
        report['warnings'].append(
            "Churned partners have higher login frequency than non-churned (unexpected)"
        )
    report['stats']['churned_avg_login'] = churned_login
    report['stats']['non_churned_avg_login'] = non_churned_login

    if len(report['warnings']) > 0:
        report['is_valid'] = False

    return report


def print_data_summary(df: pd.DataFrame):
    """Print summary statistics for the generated data."""
    print("\n" + "=" * 60)
    print("DATA GENERATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal Partners: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Churn Rate: {df['churned'].mean():.2%}")

    print("\nTier Distribution:")
    tier_dist = df['tier'].value_counts()
    for tier, count in tier_dist.items():
        print(f"  {tier}: {count:,} ({count / len(df):.1%})")

    print("\nPartner Type Distribution:")
    type_dist = df['partner_type'].value_counts()
    for ptype, count in type_dist.items():
        print(f"  {ptype}: {count:,} ({count / len(df):.1%})")

    print("\nRegion Distribution:")
    region_dist = df['region'].value_counts()
    for region, count in region_dist.items():
        print(f"  {region}: {count:,} ({count / len(df):.1%})")

    print("\nKey Metrics (Churned vs Non-Churned):")
    metrics = ['login_frequency_weekly', 'commission_current_month',
               'client_signup_rate', 'sentiment_score_avg']

    for metric in metrics:
        churned_val = df[df['churned'] == 1][metric].mean()
        non_churned_val = df[df['churned'] == 0][metric].mean()
        diff_pct = ((churned_val - non_churned_val) / non_churned_val * 100) if non_churned_val != 0 else 0
        print(f"  {metric}:")
        print(f"    Churned: {churned_val:.2f}")
        print(f"    Non-Churned: {non_churned_val:.2f}")
        print(f"    Difference: {diff_pct:+.1f}%")

    print("\n" + "=" * 60)


def generate_time_series_data(
        n_partners: int = 100,
        n_weeks: int = 12,
        seed: int = None
) -> pd.DataFrame:
    """
    Generate time-series data for CUSUM analysis.
    Creates weekly snapshots of key metrics.

    Args:
        n_partners: Number of partners
        n_weeks: Number of weeks of history
        seed: Random seed

    Returns:
        DataFrame with weekly time-series data
    """
    seed = seed or Config.RANDOM_SEED
    np.random.seed(seed)

    # Generate base partner data
    base_df = generate_partner_data(n_partners=n_partners, save_to_file=False, include_temporal=False)

    records = []

    for _, partner in base_df.iterrows():
        is_churner = partner['churned'] == 1

        for week in range(n_weeks):
            week_offset = n_weeks - week - 1  # Weeks before churn

            record = {
                'partner_id': partner['partner_id'],
                'week': week + 1,
                'weeks_before_snapshot': week_offset
            }

            # For churners, simulate gradual decline
            if is_churner:
                # Decline intensifies as we approach churn (week 12)
                decline_factor = 1 - (week / n_weeks) * 0.4

                if week > n_weeks * 0.7:  # Last 30% of weeks
                    decline_factor *= 0.7

                record['login_frequency_weekly'] = max(0.5,
                                                       partner['login_frequency_weekly'] * decline_factor * np.random.normal(1, 0.1))
                record['commission'] = max(0,
                                           partner['commission_current_month'] * decline_factor * np.random.normal(1, 0.15))
                record['client_signup_rate'] = max(0,
                                                   partner['client_signup_rate'] * decline_factor * np.random.normal(1, 0.12))
            else:
                # Non-churners have stable metrics with noise
                record['login_frequency_weekly'] = max(0.5,
                                                       partner['login_frequency_weekly'] * np.random.normal(1, 0.08))
                record['commission'] = max(0,
                                           partner['commission_current_month'] * np.random.normal(1, 0.10))
                record['client_signup_rate'] = max(0,
                                                   partner['client_signup_rate'] * np.random.normal(1, 0.08))

            record['churned'] = partner['churned']
            records.append(record)

    ts_df = pd.DataFrame(records)

    return ts_df


if __name__ == "__main__":
    # Generate the main dataset
    print("Generating Partner Churn Dataset")
    print("-" * 40)

    df = generate_partner_data(
        n_partners=Config.N_PARTNERS,
        churn_rate=Config.CHURN_RATE,
        seed=Config.RANDOM_SEED,
        save_to_file=True
    )

    print(f"\nGenerated dataset shape: {df.shape}")

    # Also generate time-series data for CUSUM
    print("\n\nGenerating Time-Series Data for CUSUM...")
    ts_df = generate_time_series_data(n_partners=500, n_weeks=12)
    ts_output_path = Config.get_data_path('partner_timeseries.csv', processed=False)
    ts_df.to_csv(ts_output_path, index=False)
    print(f"Time-series data saved to: {ts_output_path}")
    print(f"Time-series shape: {ts_df.shape}")
