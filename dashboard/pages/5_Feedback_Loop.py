"""
Feedback Loop Page
Intervention tracking and effectiveness analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Feedback Loop - Partner Churn", page_icon="P", layout="wide")


def generate_sample_interventions() -> pd.DataFrame:
    """Generate sample intervention data for demo."""
    np.random.seed(42)
    n = 50

    intervention_types = ['dispute_resolution', 'personal_outreach', 'rate_review',
                         'client_reactivation', 'escalation_resolution']
    outcomes = ['Retained', 'Stabilized', 'Churned', 'Declined']

    dates = [datetime.now() - timedelta(days=np.random.randint(1, 90)) for _ in range(n)]

    return pd.DataFrame({
        'intervention_id': [f'INT_{i:04d}' for i in range(n)],
        'partner_id': [f'P{np.random.randint(1, 200):04d}' for _ in range(n)],
        'intervention_date': dates,
        'intervention_type': np.random.choice(intervention_types, n),
        'risk_tier_at_trigger': np.random.choice(['high', 'medium'], n, p=[0.6, 0.4]),
        'intervention_cost': np.random.choice([100, 150, 300, 500, 800], n),
        'outcome_30d': np.random.choice(outcomes, n, p=[0.35, 0.25, 0.25, 0.15]),
        'risk_score_at_trigger': np.random.uniform(0.5, 0.9, n),
        'risk_score_30d': np.random.uniform(0.2, 0.6, n),
        'commission_change_30d_pct': np.random.uniform(-10, 25, n)
    })


def calculate_effectiveness(interventions: pd.DataFrame) -> dict:
    """Calculate intervention effectiveness metrics."""
    completed = interventions[interventions['outcome_30d'].notna()]

    success_outcomes = ['Retained', 'Stabilized']
    success = completed[completed['outcome_30d'].isin(success_outcomes)]

    success_rate = len(success) / len(completed) if len(completed) > 0 else 0

    # By type
    type_effectiveness = {}
    for itype in completed['intervention_type'].unique():
        type_data = completed[completed['intervention_type'] == itype]
        type_success = type_data[type_data['outcome_30d'].isin(success_outcomes)]
        type_effectiveness[itype] = {
            'count': len(type_data),
            'success_rate': len(type_success) / len(type_data) if len(type_data) > 0 else 0,
            'avg_cost': type_data['intervention_cost'].mean()
        }

    return {
        'total_interventions': len(interventions),
        'completed': len(completed),
        'overall_success_rate': success_rate,
        'total_cost': interventions['intervention_cost'].sum(),
        'avg_risk_reduction': (interventions['risk_score_at_trigger'] - interventions['risk_score_30d']).mean(),
        'type_effectiveness': type_effectiveness
    }


def create_success_by_type_chart(type_effectiveness: dict) -> go.Figure:
    """Create success rate by intervention type chart."""
    types = list(type_effectiveness.keys())
    rates = [type_effectiveness[t]['success_rate'] * 100 for t in types]
    counts = [type_effectiveness[t]['count'] for t in types]

    fig = go.Figure(data=[
        go.Bar(
            x=[t.replace('_', ' ').title() for t in types],
            y=rates,
            marker_color=['#00cc66' if r >= 50 else '#ffa500' if r >= 30 else '#ff4b4b' for r in rates],
            text=[f"{r:.0f}%" for r in rates],
            textposition='outside',
            hovertemplate="Type: %{x}<br>Success Rate: %{y:.0f}%<br>Count: %{customdata}<extra></extra>",
            customdata=counts
        )
    ])

    fig.update_layout(
        title="Intervention Success Rate by Type",
        xaxis_title="Intervention Type",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 100],
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_trend_chart(interventions: pd.DataFrame) -> go.Figure:
    """Create intervention trend chart over time."""
    interventions['week'] = pd.to_datetime(interventions['intervention_date']).dt.to_period('W').astype(str)

    weekly = interventions.groupby('week').agg({
        'intervention_id': 'count',
        'outcome_30d': lambda x: (x.isin(['Retained', 'Stabilized'])).sum()
    }).reset_index()

    weekly.columns = ['Week', 'Interventions', 'Successful']
    weekly['Success Rate'] = weekly['Successful'] / weekly['Interventions'] * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=weekly['Week'],
        y=weekly['Success Rate'],
        mode='lines+markers',
        name='Success Rate',
        line=dict(color='#00cc66', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Weekly Intervention Success Rate",
        xaxis_title="Week",
        yaxis_title="Success Rate (%)",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def main():
    """Feedback loop page."""
    st.title("Feedback Loop")
    st.markdown("Intervention outcome tracking and effectiveness analysis.")

    st.markdown("---")

    # Load or generate intervention data
    if 'interventions' not in st.session_state:
        st.session_state.interventions = generate_sample_interventions()

    interventions = st.session_state.interventions
    effectiveness = calculate_effectiveness(interventions)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Interventions",
            effectiveness['total_interventions']
        )

    with col2:
        st.metric(
            "Overall Success Rate",
            f"{effectiveness['overall_success_rate']:.0%}"
        )

    with col3:
        st.metric(
            "Total Investment",
            f"${effectiveness['total_cost']:,.0f}"
        )

    with col4:
        st.metric(
            "Avg Risk Reduction",
            f"{effectiveness['avg_risk_reduction']:.0%}"
        )

    st.markdown("---")

    # Charts row
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_success_by_type_chart(effectiveness['type_effectiveness']),
                       use_container_width=True)

    with col_right:
        st.plotly_chart(create_trend_chart(interventions), use_container_width=True)

    st.markdown("---")

    # Recent interventions table
    st.subheader("Recent Interventions")

    recent = interventions.nlargest(10, 'intervention_date').copy()
    recent['intervention_date'] = pd.to_datetime(recent['intervention_date']).dt.strftime('%Y-%m-%d')
    recent['risk_score_at_trigger'] = recent['risk_score_at_trigger'].apply(lambda x: f"{x:.0%}")
    recent['risk_score_30d'] = recent['risk_score_30d'].apply(lambda x: f"{x:.0%}")
    recent['intervention_cost'] = recent['intervention_cost'].apply(lambda x: f"${x:,.0f}")
    recent['intervention_type'] = recent['intervention_type'].apply(lambda x: x.replace('_', ' ').title())

    display_cols = ['partner_id', 'intervention_date', 'intervention_type',
                   'risk_score_at_trigger', 'outcome_30d', 'intervention_cost']
    display_df = recent[display_cols].copy()
    display_df.columns = ['Partner', 'Date', 'Type', 'Initial Risk', 'Outcome', 'Cost']

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Log new intervention form
    st.subheader("Log New Intervention")

    with st.form("new_intervention"):
        col1, col2 = st.columns(2)

        with col1:
            partner_id = st.text_input("Partner ID", placeholder="P0001")
            intervention_type = st.selectbox(
                "Intervention Type",
                ['dispute_resolution', 'personal_outreach', 'rate_review',
                 'client_reactivation', 'escalation_resolution']
            )

        with col2:
            risk_score = st.slider("Risk Score at Trigger", 0.0, 1.0, 0.65)
            cost = st.number_input("Intervention Cost ($)", 100, 1000, 300)

        notes = st.text_area("Notes", placeholder="Add any relevant notes...")

        submitted = st.form_submit_button("Log Intervention", type="primary")

        if submitted:
            if partner_id:
                new_intervention = pd.DataFrame([{
                    'intervention_id': f'INT_{len(interventions):04d}',
                    'partner_id': partner_id,
                    'intervention_date': datetime.now(),
                    'intervention_type': intervention_type,
                    'risk_tier_at_trigger': 'high' if risk_score >= 0.7 else 'medium',
                    'intervention_cost': cost,
                    'outcome_30d': None,
                    'risk_score_at_trigger': risk_score,
                    'risk_score_30d': None,
                    'commission_change_30d_pct': None
                }])

                st.session_state.interventions = pd.concat([interventions, new_intervention],
                                                          ignore_index=True)
                st.success(f"Intervention logged for {partner_id}")
                st.rerun()
            else:
                st.error("Please enter a Partner ID")

    st.markdown("---")

    # Insights section
    st.subheader("Key Insights")

    # Find best and worst performing intervention types
    if effectiveness['type_effectiveness']:
        sorted_types = sorted(
            effectiveness['type_effectiveness'].items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )

        best_type = sorted_types[0]
        worst_type = sorted_types[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"""
            **Best Performing:** {best_type[0].replace('_', ' ').title()}
            - Success Rate: {best_type[1]['success_rate']:.0%}
            - Sample Size: {best_type[1]['count']} interventions
            """)

        with col2:
            st.warning(f"""
            **Needs Improvement:** {worst_type[0].replace('_', ' ').title()}
            - Success Rate: {worst_type[1]['success_rate']:.0%}
            - Sample Size: {worst_type[1]['count']} interventions
            """)


if __name__ == "__main__":
    main()
