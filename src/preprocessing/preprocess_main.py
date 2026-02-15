"""
Preprocessing pipeline for partner churn data.
Handles data cleaning, encoding, scaling, and train/test splitting.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config
from config.feature_schema import FeatureSchema


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for partner churn prediction.
    Handles encoding, scaling, and feature engineering.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'PreprocessingPipeline':
        """
        Fit the preprocessing pipeline on training data.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        df = df.copy()

        # Identify feature types
        self.categorical_features = FeatureSchema.get_categorical_features()
        self.numeric_features = [col for col in FeatureSchema.get_numeric_features()
                                 if col in df.columns]

        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le

        # Fit scaler on numeric features
        numeric_data = df[self.numeric_features].fillna(0)
        self.scaler.fit(numeric_data)

        # Store feature names (categorical encoded + numeric)
        self.feature_names = list(self.categorical_features) + self.numeric_features

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        df = df.copy()

        # Encode categorical features
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                unseen_mask = ~df[col].isin(le.classes_)
                if unseen_mask.any():
                    # Map unseen to most common class
                    df.loc[unseen_mask, col] = le.classes_[0]
                df[col] = le.transform(df[col])

        # Scale numeric features
        numeric_data = df[self.numeric_features].fillna(0)
        df[self.numeric_features] = self.scaler.transform(numeric_data)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def inverse_transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform categorical columns for interpretation."""
        df = df.copy()
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.inverse_transform(df[col].astype(int))
        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names after preprocessing."""
        return self.feature_names

    def save(self, path: Path):
        """Save the fitted pipeline to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'is_fitted': self.is_fitted
        }, path)

    def load(self, path: Path) -> 'PreprocessingPipeline':
        """Load a fitted pipeline from disk."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        self.is_fitted = data['is_fitted']
        return self


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw partner data.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['partner_id'])
    if len(df) < initial_rows:
        print(f"  Removed {initial_rows - len(df)} duplicate partners")

    # Handle missing values
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove outliers using IQR method for key numeric features
    outlier_cols = ['commission_current_month', 'login_frequency_weekly',
                    'client_signup_rate', 'active_clients_count']

    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional engineered features.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()

    # Engagement score (composite)
    if all(col in df.columns for col in ['login_frequency_weekly', 'session_duration_avg',
                                          'dashboard_clicks_daily']):
        df['engagement_score'] = (
                df['login_frequency_weekly'] / df['login_frequency_weekly'].max() * 0.4 +
                df['session_duration_avg'] / df['session_duration_avg'].max() * 0.3 +
                df['dashboard_clicks_daily'] / df['dashboard_clicks_daily'].max() * 0.3
        ) * 100

    # Revenue health score
    if all(col in df.columns for col in ['commission_current_month', 'commission_trend',
                                          'commission_vs_target']):
        df['revenue_health_score'] = (
                (df['commission_current_month'] / df['commission_current_month'].max() * 0.5) +
                ((df['commission_trend'] + 1) / 2 * 0.3) +
                ((df['commission_vs_target'] + 100) / 300 * 0.2)
        ) * 100

    # Client health score
    if all(col in df.columns for col in ['active_clients_count', 'client_retention_rate',
                                          'client_signup_rate']):
        df['client_health_score'] = (
                df['active_clients_count'] / df['active_clients_count'].max() * 0.4 +
                df['client_retention_rate'] / 100 * 0.4 +
                df['client_signup_rate'] / df['client_signup_rate'].max() * 0.2
        ) * 100

    # Risk indicator (composite of negative signals)
    if all(col in df.columns for col in ['competitor_mentions_count', 'escalation_count',
                                          'complaint_count', 'frustration_keyword_count']):
        df['risk_indicator'] = (
                df['competitor_mentions_count'] * 3 +
                df['escalation_count'] * 4 +
                df['complaint_count'] * 5 +
                df['frustration_keyword_count'] * 2
        )
        df['risk_indicator'] = (df['risk_indicator'] / df['risk_indicator'].max() * 100).clip(0, 100)

    # Tenure bucket
    if 'tenure_months' in df.columns:
        df['tenure_bucket'] = pd.cut(
            df['tenure_months'],
            bins=[0, 6, 12, 24, 48, float('inf')],
            labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4yr+']
        ).astype(str)

    return df


def prepare_model_data(
        df: pd.DataFrame,
        target_col: str = 'churned'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.

    Args:
        df: Preprocessed DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Columns to exclude from features
    exclude_cols = ['partner_id', target_col, 'churn_probability',
                    'days_until_churn', 'tenure_bucket']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]

    return X, y


def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = None,
        random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    test_size = test_size or Config.TEST_SIZE
    random_state = random_state or Config.RANDOM_SEED

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(
        input_path: Optional[Path] = None,
        save_output: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingPipeline]:
    """
    Complete preprocessing pipeline execution.

    Args:
        input_path: Path to raw data file
        save_output: Whether to save processed data

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, pipeline)
    """
    print("Starting preprocessing pipeline...")

    # Load data
    if input_path is None:
        input_path = Config.get_data_path('partner_data.csv', processed=False)

    print(f"  Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")

    # Clean data
    print("  Cleaning data...")
    df = clean_data(df)

    # Engineer features
    print("  Engineering features...")
    df = engineer_features(df)

    # Initialize and fit preprocessing pipeline
    print("  Fitting preprocessing pipeline...")
    pipeline = PreprocessingPipeline()

    # Prepare features and target
    X, y = prepare_model_data(df)

    # Split data
    print("  Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Fit pipeline on training data only
    pipeline.fit(X_train)

    # Transform both sets
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print(f"\n  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Churn rate (train): {y_train.mean():.2%}")
    print(f"  Churn rate (test): {y_test.mean():.2%}")

    # Save if requested
    if save_output:
        Config.ensure_directories()

        # Save processed datasets
        train_data = X_train_processed.copy()
        train_data['churned'] = y_train.values
        train_data.to_csv(Config.get_data_path('train_data.csv'), index=False)

        test_data = X_test_processed.copy()
        test_data['churned'] = y_test.values
        test_data.to_csv(Config.get_data_path('test_data.csv'), index=False)

        # Save pipeline
        pipeline.save(Config.get_data_path('preprocessing_pipeline.joblib'))

        print(f"\n  Saved processed data to: {Config.PROCESSED_DATA_DIR}")

    return X_train_processed, X_test_processed, y_train, y_test, pipeline


if __name__ == "__main__":
    # Run preprocessing
    X_train, X_test, y_train, y_test, pipeline = preprocess_data()

    print("\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nFeature names: {pipeline.get_feature_names()[:10]}...")
