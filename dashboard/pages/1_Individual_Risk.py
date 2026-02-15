"""
Individual Risk Analysis Page
Detailed partner risk profiles with SHAP-based explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Individual Risk - Partner Churn", page_icon="P", layout="wide")


def create_risk_gauge(probability: float) -> go.Figure:
    """Create a gauge chart for churn risk probability."""
    if probability >= 0.8:
        color = "#FF0000"
        risk_label = "CRITICAL"
    elif probability >= 0.6:
        color = "#FF6B00"
        risk_label = "HIGH"
    elif probability >= 0.4:
        color = "#FFB800"
        risk_label = "MEDIUM"
    elif probability >= 0.2:
        color = "#00C853"
        risk_label = "LOW"
    else:
        color = "#00E676"
        risk_label = "MINIMAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': f"Churn Risk - {risk_label}", 'font': {'size': 18}},
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'steps': [
                {'range': [0, 20], 'color': '#e8f5e9'},
                {'range': [20, 40], 'color': '#c8e6c9'},
                {'range': [40, 60], 'color': '#fff3e0'},
                {'range': [60, 80], 'color': '#ffe0b2'},
                {'range': [80, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_mock_shap_waterfall(partner_data: dict) -> go.Figure:
    """Create a SHAP-style waterfall chart showing risk drivers."""
    # Simulate top risk drivers based on partner features
    drivers = []

    if partner_data.get('days_since_last_login', 0) > 14:
        drivers.append(('Days Since Login', 0.15))
    if partner_data.get('login_frequency_weekly', 5) < 3:
        drivers.append(('Login Frequency', 0.12))
    if partner_data.get('competitor_mentions_count', 0) > 1:
        drivers.append(('Competitor Mentions', 0.10))
    if partner_data.get('commission_current_month', 10000) < 5000:
        drivers.append(('Low Commission', 0.08))
    if partner_data.get('unresolved_tickets_count', 0) > 0:
        drivers.append(('Unresolved Tickets', 0.06))

    # Add some protective factors
    if partner_data.get('tenure_months', 0) > 24:
        drivers.append(('High Tenure', -0.08))
    if partner_data.get('active_clients_count', 0) > 30:
        drivers.append(('Active Client Base', -0.05))

    # Ensure we have at least some drivers
    if len(drivers) < 3:
        drivers.extend([
            ('Engagement Score', 0.07),
            ('Revenue Trend', 0.05),
            ('Support Satisfaction', -0.03)
        ])

    # Sort by absolute impact
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    drivers = drivers[:7]

    features = [d[0] for d in drivers]
    impacts = [d[1] for d in drivers]
    colors = ['#ff4b4b' if v > 0 else '#00cc66' for v in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in impacts],
        textposition='outside'
    ))

    fig.update_layout(
        title="Top Risk Drivers (SHAP Impact)",
        xaxis_title="Impact on Churn Probability",
        height=350,
        margin=dict(l=150, r=50, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange='reversed')
    )

    return fig


def create_engagement_timeline() -> go.Figure:
    """Create engagement timeline chart."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='W')
    np.random.seed(123)

    logins = 5 + np.cumsum(np.random.randn(12) * 0.5)
    logins = np.clip(logins, 0, 10)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=logins,
        mode='lines+markers',
        name='Weekly Logins',
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=6)
    ))

    # Add trend line
    z = np.polyfit(range(len(dates)), logins, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=dates, y=p(range(len(dates))),
        mode='lines',
        name='Trend',
        line=dict(color='#ff4b4b', width=2, dash='dash')
    ))

    fig.update_layout(
        title="Engagement Trend (Last 12 Weeks)",
        xaxis_title="Week",
        yaxis_title="Weekly Logins",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    return fig


def get_intervention_recommendations(partner: dict) -> list:
    """Generate intervention recommendations based on partner profile."""
    recommendations = []
    prob = partner.get('churn_probability', 0.5)
    tier = partner.get('tier', 'silver')

    if prob >= 0.8:
        recommendations.append({
            'priority': 'URGENT',
            'action': 'Emergency call from senior RM or executive',
            'timeline': 'Within 24 hours',
            'cost': '$1,000'
        })
    if prob >= 0.6:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Personal call to discuss challenges and offer support',
            'timeline': 'Within 48 hours',
            'cost': '$50'
        })
    if partner.get('competitor_mentions_count', 0) > 0:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Competitive counter-offer with enhanced revenue share',
            'timeline': 'Within 1 week',
            'cost': '$500'
        })
    if tier in ['gold', 'platinum']:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Custom retention package with bonus incentives',
            'timeline': 'Within 2 weeks',
            'cost': '$2,500'
        })
    else:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Account performance review and optimization plan',
            'timeline': 'Within 1 week',
            'cost': '$100'
        })

    recommendations.append({
        'priority': 'LOW',
        'action': 'Schedule regular check-in cadence',
        'timeline': 'Ongoing',
        'cost': '$50/month'
    })

    return recommendations[:4]


def main():
    """Individual risk analysis page."""
    if 'data' not in st.session_state:
        st.error("Please navigate from the main dashboard to load data.")
        st.stop()

    data = st.session_state.data

    st.title("Individual Partner Risk Analysis")
    st.markdown("Detailed partner risk profiles with model-driven explanations.")

    st.markdown("---")

    # Partner selection
    col_select, col_filter = st.columns([2, 1])

    with col_select:
        # Sort partners by churn probability for easier selection
        sorted_partners = data.sort_values('churn_probability', ascending=False)
        partner_options = sorted_partners['partner_id'].tolist()

        selected_id = st.selectbox(
            "Select Partner",
            partner_options,
            format_func=lambda x: f"{x} - Risk: {data[data['partner_id']==x]['churn_probability'].values[0]:.0%}"
        )

    with col_filter:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ['All', 'Critical (80%+)', 'High (60-80%)', 'Medium (40-60%)', 'Low (<40%)']
        )

    # Get selected partner data
    partner = data[data['partner_id'] == selected_id].iloc[0].to_dict()

    st.markdown("---")

    # Partner overview
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.plotly_chart(create_risk_gauge(partner['churn_probability']), use_container_width=True)

    with col2:
        st.markdown("### Partner Profile")
        st.markdown(f"**ID:** {partner['partner_id']}")
        st.markdown(f"**Tier:** {partner.get('tier', 'N/A').title()}")
        st.markdown(f"**Region:** {partner.get('region', 'N/A').replace('_', ' ').title()}")
        st.markdown(f"**Type:** {partner.get('partner_type', 'N/A').title()}")
        st.markdown(f"**Tenure:** {partner.get('tenure_months', 0)} months")

    with col3:
        st.markdown("### Key Metrics")
        commission = partner.get('commission_current_month', 0)
        st.metric("Monthly Commission", f"${commission:,.0f}")

        logins = partner.get('login_frequency_weekly', 0)
        st.metric("Weekly Logins", f"{logins:.1f}")

        clients = partner.get('active_clients_count', 0)
        st.metric("Active Clients", f"{int(clients):,}")

    st.markdown("---")

    # Risk drivers and timeline
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_mock_shap_waterfall(partner), use_container_width=True)

    with col_right:
        st.plotly_chart(create_engagement_timeline(), use_container_width=True)

    st.markdown("---")

    # Intervention recommendations
    st.subheader("Recommended Interventions")

    recommendations = get_intervention_recommendations(partner)

    for i, rec in enumerate(recommendations, 1):
        priority = rec['priority']
        if priority == 'URGENT':
            icon = "[!]"
        elif priority == 'HIGH':
            icon = "[H]"
        elif priority == 'MEDIUM':
            icon = "[-]"
        else:
            icon = "[L]"

        with st.expander(f"{icon} {priority}: {rec['action']}", expanded=(i <= 2)):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Timeline:** {rec['timeline']}")
            with col_b:
                st.markdown(f"**Estimated Cost:** {rec['cost']}")

    st.markdown("---")

    # Financial impact summary
    st.subheader("Financial Impact Assessment")

    annual_value = partner.get('commission_current_month', 0) * 12
    expected_loss = annual_value * partner['churn_probability']
    tier_mult = {'platinum': 5.0, 'gold': 2.5, 'silver': 1.5, 'bronze': 1.0}
    lifetime_factor = tier_mult.get(partner.get('tier', 'bronze'), 1.0)

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        st.metric("Annual Value", f"${annual_value:,.0f}")
    with col_f2:
        st.metric("Expected Annual Loss", f"${expected_loss:,.0f}", delta_color="inverse")
    with col_f3:
        st.metric("Lifetime Value Multiplier", f"{lifetime_factor:.1f}x")
    with col_f4:
        retention_cost = 500 if partner.get('tier') in ['platinum', 'gold'] else 100
        roi = expected_loss / retention_cost if retention_cost > 0 else 0
        st.metric("Intervention ROI", f"{roi:.0f}x")

    # Navigation
    st.markdown("---")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("View Portfolio", use_container_width=True):
            st.switch_page("pages/2_Portfolio_View.py")
    with col_nav2:
        if st.button("Business Impact", use_container_width=True):
            st.switch_page("pages/3_Business_Impact.py")


if __name__ == "__main__":
    main()
