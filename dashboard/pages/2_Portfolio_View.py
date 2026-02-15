"""
Portfolio View Page
Portfolio-level risk distribution and cohort analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Portfolio View - Partner Churn", page_icon="📊", layout="wide")


def create_risk_distribution_pie(data: pd.DataFrame) -> go.Figure:
    """Create risk distribution pie chart."""
    high = (data['churn_probability'] >= 0.7).sum()
    medium = ((data['churn_probability'] >= 0.4) & (data['churn_probability'] < 0.7)).sum()
    low = (data['churn_probability'] < 0.4).sum()
    
    fig = go.Figure(data=[go.Pie(
        labels=['High Risk', 'Medium Risk', 'Low Risk'],
        values=[high, medium, low],
        marker_colors=['#ff4b4b', '#ffa500', '#00cc66'],
        hole=0.4,
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title="Portfolio Risk Distribution",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_tier_risk_heatmap(data: pd.DataFrame) -> go.Figure:
    """Create tier x region risk heatmap."""
    pivot = data.pivot_table(
        values='churn_probability',
        index='tier',
        columns='region',
        aggfunc='mean'
    )
    
    # Order tiers
    tier_order = ['platinum', 'gold', 'silver', 'bronze']
    pivot = pivot.reindex([t for t in tier_order if t in pivot.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=[[f"{v:.0f}%" for v in row] for row in pivot.values * 100],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Tier: %{y}<br>Region: %{x}<br>Avg Risk: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Average Churn Risk by Tier & Region",
        xaxis_title="Region",
        yaxis_title="Tier",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_risk_trend_chart(data: pd.DataFrame) -> go.Figure:
    """Create simulated risk trend over time."""
    # Simulate historical trend
    dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='M')
    
    np.random.seed(42)
    high_risk_trend = 15 + np.cumsum(np.random.randn(12) * 2)
    high_risk_trend = np.clip(high_risk_trend, 10, 30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=high_risk_trend,
        mode='lines+markers',
        name='High Risk Partners',
        line=dict(color='#ff4b4b', width=3),
        marker=dict(size=8)
    ))
    
    # Add current value marker
    current_high = (data['churn_probability'] >= 0.7).sum()
    fig.add_trace(go.Scatter(
        x=[dates[-1]],
        y=[current_high],
        mode='markers',
        name='Current',
        marker=dict(color='#ff4b4b', size=15, symbol='star')
    ))
    
    fig.update_layout(
        title="High-Risk Partner Trend (12 Months)",
        xaxis_title="Month",
        yaxis_title="Number of High-Risk Partners",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig


def identify_cohort_risks(data: pd.DataFrame) -> pd.DataFrame:
    """Identify cohort-level risk patterns."""
    cohorts = data.groupby(['tier', 'region']).agg({
        'partner_id': 'count',
        'churn_probability': ['mean', 'std'],
        'commission_current_month': 'sum'
    }).reset_index()
    
    cohorts.columns = ['Tier', 'Region', 'Partners', 'Avg Risk', 'Risk StdDev', 'Monthly Value']
    
    # Flag high-risk cohorts
    cohorts['Alert'] = cohorts.apply(
        lambda x: '🚨 ALERT' if x['Avg Risk'] >= 0.5 and x['Partners'] >= 5 else '',
        axis=1
    )
    
    # Format values
    cohorts['Avg Risk'] = cohorts['Avg Risk'].apply(lambda x: f"{x:.0%}")
    cohorts['Monthly Value'] = cohorts['Monthly Value'].apply(lambda x: f"${x:,.0f}")
    
    return cohorts.sort_values('Alert', ascending=False)


def main():
    """Portfolio view page."""
    if 'data' not in st.session_state:
        st.error("Please navigate from the main dashboard to load data.")
        st.stop()
    
    data = st.session_state.data
    
    st.title("📊 Portfolio View")
    st.markdown("Overview of portfolio-wide risk distribution and cohort analysis.")
    
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = (data['churn_probability'] >= 0.7).sum()
    medium_risk = ((data['churn_probability'] >= 0.4) & (data['churn_probability'] < 0.7)).sum()
    
    with col1:
        st.metric("Total Partners", len(data))
    with col2:
        st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/len(data)*100:.0f}%")
    with col3:
        avg_risk = data['churn_probability'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.0%}")
    with col4:
        total_value = data['commission_current_month'].sum()
        st.metric("Monthly Portfolio Value", f"${total_value/1e3:.0f}K")
    
    st.markdown("---")
    
    # Charts row
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.plotly_chart(create_risk_distribution_pie(data), use_container_width=True)
    
    with col_right:
        st.plotly_chart(create_tier_risk_heatmap(data), use_container_width=True)
    
    st.markdown("---")
    
    # Trend chart
    st.plotly_chart(create_risk_trend_chart(data), use_container_width=True)
    
    st.markdown("---")
    
    # Cohort analysis
    st.subheader("🔍 Cohort Risk Analysis")
    st.markdown("Identifying cohorts (tier + region combinations) with elevated risk patterns.")
    
    cohorts = identify_cohort_risks(data)
    
    # Filter to show alerts first
    alerts = cohorts[cohorts['Alert'] == '🚨 ALERT']
    
    if len(alerts) > 0:
        st.warning(f"⚠️ {len(alerts)} cohort(s) showing elevated risk patterns!")
        st.dataframe(alerts, use_container_width=True, hide_index=True)
    
    # Full cohort table with expander
    with st.expander("View All Cohorts"):
        st.dataframe(cohorts, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Risk by tier breakdown
    st.subheader("📈 Risk Distribution by Tier")
    
    tier_data = data.groupby('tier').agg({
        'partner_id': 'count',
        'churn_probability': 'mean',
        'commission_current_month': 'sum'
    }).reset_index()
    
    tier_data.columns = ['Tier', 'Partners', 'Avg Churn Risk', 'Monthly Commission']
    
    # Order by tier importance
    tier_order = {'platinum': 0, 'gold': 1, 'silver': 2, 'bronze': 3}
    tier_data['_order'] = tier_data['Tier'].map(tier_order)
    tier_data = tier_data.sort_values('_order').drop('_order', axis=1)
    
    fig = px.bar(
        tier_data,
        x='Tier',
        y='Avg Churn Risk',
        color='Avg Churn Risk',
        color_continuous_scale='RdYlGn_r',
        text=tier_data['Avg Churn Risk'].apply(lambda x: f"{x:.0%}"),
        title="Average Churn Risk by Partner Tier"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
