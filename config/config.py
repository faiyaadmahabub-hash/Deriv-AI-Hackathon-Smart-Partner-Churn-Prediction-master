"""
Configuration module for Partner Churn Prediction System.
Contains all parameters, thresholds, API keys, and system settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class for the churn prediction system."""

    # ========== PATH CONFIGURATION ==========
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    DEMO_DIR = BASE_DIR / "demo"

    # ========== DATA GENERATION SETTINGS ==========
    N_PARTNERS = 2500
    CHURN_RATE = 0.25  # 25% churn rate
    RANDOM_SEED = 42

    # Partner tier distribution
    TIER_DISTRIBUTION = {
        'platinum': 0.05,
        'gold': 0.15,
        'silver': 0.35,
        'bronze': 0.45
    }

    # Partner type distribution
    TYPE_DISTRIBUTION = {
        'affiliate': 0.40,
        'ib': 0.35,  # Introducing Broker
        'master_affiliate': 0.15,
        'payment_agent': 0.10
    }

    # Region distribution
    REGION_DISTRIBUTION = {
        'asia_pacific': 0.30,
        'europe': 0.25,
        'latin_america': 0.20,
        'africa': 0.15,
        'middle_east': 0.10
    }

    # ========== MODEL PARAMETERS ==========
    # XGBoost hyperparameters
    XGBOOST_PARAMS = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 3,  # Handle class imbalance
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'eval_metric': 'aucpr'
    }

    # Train/test split
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.15

    # ========== CUSUM DETECTION PARAMETERS ==========
    CUSUM_CONFIG = {
        'threshold_multiplier': 4.0,  # Detection sensitivity
        'drift_rate': 0.5,  # Expected drift per period
        'min_periods': 4,  # Minimum periods before detection
        'features_to_monitor': [
            'login_frequency_weekly',
            'commission_current_month',
            'client_signup_rate',
            'session_duration_avg',
            'active_clients_count'
        ],
        'alert_window_days': 21  # Alert this many days before predicted churn
    }

    # ========== RISK TIER THRESHOLDS ==========
    RISK_TIERS = {
        'critical': {'min_prob': 0.80, 'color': '#FF0000', 'priority': 1},
        'high': {'min_prob': 0.60, 'color': '#FF6B00', 'priority': 2},
        'medium': {'min_prob': 0.40, 'color': '#FFB800', 'priority': 3},
        'low': {'min_prob': 0.20, 'color': '#00C853', 'priority': 4},
        'minimal': {'min_prob': 0.00, 'color': '#00E676', 'priority': 5}
    }

    # ========== BUSINESS IMPACT SETTINGS ==========
    # Revenue multipliers by tier
    TIER_REVENUE_MULTIPLIERS = {
        'platinum': 5.0,
        'gold': 2.5,
        'silver': 1.5,
        'bronze': 1.0
    }

    # Average monthly commission by tier (USD)
    AVG_MONTHLY_COMMISSION = {
        'platinum': 50000,
        'gold': 20000,
        'silver': 8000,
        'bronze': 3000
    }

    # Intervention costs (USD)
    INTERVENTION_COSTS = {
        'personal_call': 50,
        'account_review': 100,
        'custom_incentive': 500,
        'strategic_meeting': 1000,
        'retention_package': 2500
    }

    # Intervention success rates (baseline)
    INTERVENTION_SUCCESS_RATES = {
        'personal_call': 0.15,
        'account_review': 0.25,
        'custom_incentive': 0.40,
        'strategic_meeting': 0.50,
        'retention_package': 0.65
    }

    # ========== COMPETITIVE INTELLIGENCE SETTINGS ==========
    COMPETITOR_KEYWORDS = {
        'exness': ['exness', 'exness global'],
        'xm': ['xm', 'xm global', 'xm group'],
        'fbs': ['fbs', 'fbs markets'],
        'octafx': ['octafx', 'octa fx', 'octa'],
        'icmarkets': ['icmarkets', 'ic markets', 'ic market'],
        'pepperstone': ['pepperstone'],
        'fxtm': ['fxtm', 'forextime'],
        'hotforex': ['hotforex', 'hot forex', 'hfm'],
        'avatrade': ['avatrade', 'ava trade'],
        'etoro': ['etoro', 'e-toro'],
        'plus500': ['plus500', 'plus 500'],
        'oanda': ['oanda'],
        'tickmill': ['tickmill'],
        'fxpro': ['fxpro', 'fx pro'],
        'admiralmarkets': ['admiralmarkets', 'admiral markets', 'admirals'],
        'ig': ['ig markets', 'ig group'],
        'saxo': ['saxo', 'saxo bank'],
        'cmc': ['cmc', 'cmc markets']
    }

    THREAT_CATEGORIES = {
        'pricing': ['commission', 'spread', 'fees', 'rebate', 'payout', 'rate'],
        'features': ['platform', 'tools', 'api', 'mobile', 'trading', 'signals'],
        'support': ['support', 'manager', 'response', 'help', 'service'],
        'rates': ['conversion', 'cpa', 'revenue share', 'hybrid']
    }

    # Alias for modules that reference THREAT_TYPE_KEYWORDS
    THREAT_TYPE_KEYWORDS = THREAT_CATEGORIES

    # Region definitions
    REGIONS = list(REGION_DISTRIBUTION.keys())

    # ========== GEMINI API SETTINGS ==========
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_API_KEY', '')
    GEMINI_MODEL = 'gemini-1.5-flash'
    GEMINI_MAX_TOKENS = 1024
    GEMINI_TEMPERATURE = 0.7

    # ========== SHAP SETTINGS ==========
    SHAP_CONFIG = {
        'top_k_features': 5,  # Number of top drivers to show
        'background_samples': 100,  # Samples for SHAP background
        'max_display': 10  # Max features in plots
    }

    # ========== COHORT DETECTION SETTINGS ==========
    COHORT_CONFIG = {
        'min_cohort_size': 3,  # Minimum partners for cohort
        'similarity_threshold': 0.7,  # Min similarity for grouping
        'time_window_days': 30,  # Window for temporal cohorts
        'grouping_dimensions': ['region', 'tier', 'type', 'tenure_bucket']
    }

    # ========== FEEDBACK LOOP SETTINGS ==========
    FEEDBACK_CONFIG = {
        'tracking_period_days': 90,  # How long to track intervention outcomes
        'success_threshold': 0.5,  # Probability reduction for success
        'min_interventions_for_learning': 20,  # Min data for pattern mining
        'retraining_threshold': 100  # Retrain model after N interventions
    }

    # ========== DASHBOARD SETTINGS ==========
    DASHBOARD_CONFIG = {
        'page_title': 'Partner Churn Prediction System',
        'page_icon': 'P',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded',
        'refresh_interval_seconds': 300  # Auto-refresh every 5 minutes
    }

    # Color scheme for dashboard
    COLORS = {
        'primary': '#1E88E5',
        'secondary': '#FFC107',
        'success': '#4CAF50',
        'danger': '#F44336',
        'warning': '#FF9800',
        'info': '#00BCD4',
        'background': '#0E1117',
        'text': '#FFFFFF'
    }

    # ========== DECLINE CASCADE SETTINGS ==========
    # These define how churn manifests in partner behavior
    DECLINE_CASCADE = {
        'phase_1': {  # Early warning (8-12 weeks before)
            'login_frequency_drop': (0.1, 0.2),
            'session_duration_drop': (0.05, 0.15),
            'dashboard_clicks_drop': (0.1, 0.25)
        },
        'phase_2': {  # Engagement decline (4-8 weeks before)
            'login_frequency_drop': (0.2, 0.4),
            'support_tickets_increase': (0.2, 0.5),
            'event_attendance_drop': (0.3, 0.5),
            'training_completion_drop': (0.2, 0.4)
        },
        'phase_3': {  # Performance impact (2-4 weeks before)
            'client_signup_drop': (0.2, 0.4),
            'client_retention_drop': (0.1, 0.3),
            'commission_drop': (0.15, 0.35)
        },
        'phase_4': {  # Imminent churn (0-2 weeks before)
            'login_frequency_drop': (0.5, 0.8),
            'commission_drop': (0.3, 0.6),
            'active_clients_drop': (0.2, 0.4),
            'competitor_mentions_increase': (0.3, 0.8)
        }
    }

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
                          cls.MODELS_DIR, cls.DEMO_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls, model_name: str = 'xgboost_churn_model.joblib') -> Path:
        """Get path for saving/loading model."""
        return cls.MODELS_DIR / model_name

    @classmethod
    def get_data_path(cls, filename: str, processed: bool = True) -> Path:
        """Get path for data files."""
        if processed:
            return cls.PROCESSED_DATA_DIR / filename
        return cls.RAW_DATA_DIR / filename
