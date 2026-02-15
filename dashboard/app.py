"""
Partner Churn Prediction System - Main Streamlit Dashboard
Entry point for the application with navigation and global state.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Partner Churn Prediction System",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_sample_data():
    """Load or generate sample partner data for demo."""
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'partner_data.csv'

    if data_path.exists():
        return pd.read_csv(data_path)

    # Generate synthetic data for demo
    np.random.seed(42)
    n_partners = 200

    tiers = np.random.choice(['platinum', 'gold', 'silver', 'bronze'], n_partners,
                             p=[0.05, 0.15, 0.35, 0.45])
    regions = np.random.choice(['SEA', 'MENA', 'Europe', 'Africa', 'LATAM'], n_partners,
                               p=[0.25, 0.20, 0.25, 0.15, 0.15])

    # Generate features matching the actual feature schema
    data = pd.DataFrame({
        'partner_id': [f'P{i:04d}' for i in range(n_partners)],
        'tier': tiers,
        'region': regions,
        'partner_type': np.random.choice(['ib', 'affiliate'], n_partners, p=[0.6, 0.4]),
        'tenure_months': np.random.randint(1, 60, n_partners),
        'commission_current_month': np.random.gamma(3, 5000, n_partners),
        'commission_3month_avg': np.random.gamma(3, 5000, n_partners),
        'login_frequency_weekly': np.random.poisson(4, n_partners).astype(float),
        'days_since_last_login': np.random.exponential(7, n_partners).astype(int),
        'active_clients_count': np.random.poisson(25, n_partners),
        'client_signup_rate': np.random.normal(2.0, 1.5, n_partners).clip(0),
        'support_tickets_total': np.random.poisson(2, n_partners),
        'unresolved_tickets_count': np.random.poisson(0.5, n_partners),
        'competitor_mentions_count': np.random.poisson(0.8, n_partners),
        'primary_competitor': np.random.choice(
            [None, 'Exness', 'XM', 'IC Markets', 'Pepperstone', 'FXTM'],
            n_partners, p=[0.6, 0.15, 0.10, 0.07, 0.05, 0.03]
        )
    })

    # Generate churn probabilities based on features
    base_prob = 0.25

    # Adjust based on engagement
    data['churn_probability'] = base_prob
    data.loc[data['days_since_last_login'] > 14, 'churn_probability'] += 0.2
    data.loc[data['login_frequency_weekly'] < 2, 'churn_probability'] += 0.15
    data.loc[data['client_signup_rate'] < 0.5, 'churn_probability'] += 0.15
    data.loc[data['unresolved_tickets_count'] > 0, 'churn_probability'] += 0.1
    data.loc[data['competitor_mentions_count'] > 2, 'churn_probability'] += 0.15

    # Add some noise
    data['churn_probability'] += np.random.normal(0, 0.05, n_partners)
    data['churn_probability'] = data['churn_probability'].clip(0.05, 0.95)

    # Assign risk tier (matching Config.RISK_TIERS)
    data['risk_tier'] = data['churn_probability'].apply(
        lambda p: 'critical' if p >= 0.8 else 'high' if p >= 0.6 else 'medium' if p >= 0.4 else 'low' if p >= 0.2 else 'minimal'
    )

    # Save for future use
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)

    return data


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = load_sample_data()

    if 'selected_partner' not in st.session_state:
        st.session_state.selected_partner = None


def main():
    """Main dashboard entry point."""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Partner Churn")
        st.markdown("---")

        # Navigation
        st.markdown("### Navigation")
        st.page_link("app.py", label="Home")
        st.page_link("pages/1_Individual_Risk.py", label="Individual Risk")
        st.page_link("pages/2_Portfolio_View.py", label="Portfolio View")
        st.page_link("pages/3_Business_Impact.py", label="Business Impact")
        st.page_link("pages/4_Competitive_Intel.py", label="Competitive Intel")
        st.page_link("pages/5_Feedback_Loop.py", label="Feedback Loop")

        st.markdown("---")

        # Quick stats
        data = st.session_state.data
        high_risk = (data['churn_probability'] >= 0.7).sum()
        st.metric("High Risk Partners", high_risk)
        st.metric("Total Partners", len(data))

    # Main content - Home page
    st.title("Partner Churn Prediction System")
    st.markdown("**Early Warning and Intervention Platform**")

    st.markdown("---")

    # Key metrics row
    data = st.session_state.data

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_risk = (data['churn_probability'] >= 0.7).sum()
        st.metric(
            "High Risk",
            high_risk,
            delta=f"{high_risk/len(data)*100:.0f}% of portfolio",
            delta_color="inverse"
        )

    with col2:
        medium_risk = ((data['churn_probability'] >= 0.4) & (data['churn_probability'] < 0.7)).sum()
        st.metric(
            "Medium Risk",
            medium_risk,
            delta=f"{medium_risk/len(data)*100:.0f}% of portfolio"
        )

    with col3:
        low_risk = (data['churn_probability'] < 0.4).sum()
        st.metric(
            "Low Risk",
            low_risk,
            delta=f"{low_risk/len(data)*100:.0f}% of portfolio"
        )

    with col4:
        total_value = data['commission_current_month'].sum() * 12
        st.metric(
            "Annual Portfolio Value",
            f"${total_value/1e6:.1f}M"
        )

    st.markdown("---")

    # Two column layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Risk Distribution by Region")

        # Create risk distribution chart
        import plotly.express as px

        risk_by_region = data.groupby(['region', 'risk_tier']).size().reset_index(name='count')

        fig = px.bar(
            risk_by_region,
            x='region',
            y='count',
            color='risk_tier',
            color_discrete_map={
                'critical': '#FF0000', 'high': '#FF6B00', 'medium': '#FFB800',
                'low': '#00C853', 'minimal': '#00E676'
            },
            barmode='stack',
            title='Partner Risk Distribution by Region',
            category_orders={'risk_tier': ['critical', 'high', 'medium', 'low', 'minimal']}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend_title_text='Risk Level'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Quick Actions")

        st.markdown("""
        **Most Urgent:**
        - **{}** partners need immediate attention
        - **${}K** monthly revenue at high risk

        **Recommended Actions:**
        1. Review high-risk partners
        2. Check competitive threats
        3. Plan interventions
        """.format(
            high_risk,
            int(data[data['churn_probability'] >= 0.7]['commission_current_month'].sum() / 1000)
        ))

        if st.button("Start Analysis", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Individual_Risk.py")

    st.markdown("---")

    # Top at-risk partners table
    st.subheader("Top 10 At-Risk Partners")

    display_cols = ['partner_id', 'tier', 'region', 'churn_probability', 'commission_current_month']
    if 'days_since_last_login' in data.columns:
        display_cols.append('days_since_last_login')

    top_risk = data.nlargest(10, 'churn_probability')[display_cols].copy()

    top_risk['churn_probability'] = top_risk['churn_probability'].apply(lambda x: f"{x:.0%}")
    top_risk['commission_current_month'] = top_risk['commission_current_month'].apply(lambda x: f"${x:,.0f}")

    col_rename = {
        'partner_id': 'Partner ID', 'tier': 'Tier', 'region': 'Region',
        'churn_probability': 'Churn Risk', 'commission_current_month': 'Monthly Commission',
        'days_since_last_login': 'Days Since Login'
    }
    top_risk.columns = [col_rename.get(c, c) for c in top_risk.columns]

    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()
