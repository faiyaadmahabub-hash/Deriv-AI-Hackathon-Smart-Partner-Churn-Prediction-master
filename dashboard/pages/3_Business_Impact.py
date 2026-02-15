"""
Business Impact Page
Financial exposure analysis and intervention ROI calculator.
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

st.set_page_config(page_title="Business Impact - Partner Churn", page_icon="P", layout="wide")


def calculate_exposure(data: pd.DataFrame) -> dict:
    """Calculate portfolio exposure metrics."""
    data = data.copy()

    # Calculate individual exposures
    data['annual_value'] = data['commission_current_month'] * 12
    data['value_at_risk'] = data['annual_value'] * data['churn_probability']
    data['expected_loss'] = data['value_at_risk'] * 0.8  # 80% loss on churn

    return {
        'total_annual_value': data['annual_value'].sum(),
        'total_value_at_risk': data['value_at_risk'].sum(),
        'total_expected_loss': data['expected_loss'].sum(),
        'high_risk_value': data[data['churn_probability'] >= 0.7]['value_at_risk'].sum(),
        'medium_risk_value': data[(data['churn_probability'] >= 0.4) &
                                   (data['churn_probability'] < 0.7)]['value_at_risk'].sum(),
        'data': data
    }


def create_exposure_waterfall(exposure: dict) -> go.Figure:
    """Create value at risk waterfall by tier."""
    data = exposure['data']

    tier_exposure = data.groupby('tier')['value_at_risk'].sum().sort_values(ascending=False)

    fig = go.Figure(go.Waterfall(
        x=tier_exposure.index.tolist() + ['Total'],
        y=tier_exposure.tolist() + [tier_exposure.sum()],
        measure=['relative'] * len(tier_exposure) + ['total'],
        text=[f"${v/1e3:.0f}K" for v in tier_exposure.tolist() + [tier_exposure.sum()]],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#ff4b4b"}},
        totals={"marker": {"color": "#1f77b4"}}
    ))

    fig.update_layout(
        title="Value at Risk by Partner Tier",
        yaxis_title="Value at Risk ($)",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_roi_projection(budget: float, data: pd.DataFrame) -> pd.DataFrame:
    """Calculate intervention ROI projection."""
    data = data.copy()

    # Calculate value at risk and sort by priority
    data['value_at_risk'] = data['commission_current_month'] * 12 * data['churn_probability']
    data['intervention_cost'] = data['tier'].map({
        'platinum': 800, 'gold': 500, 'silver': 300, 'bronze': 150
    }).fillna(300)
    data['success_rate'] = 0.65  # Base success rate
    data['expected_save'] = data['value_at_risk'] * data['success_rate']
    data['roi'] = (data['expected_save'] - data['intervention_cost']) / data['intervention_cost']

    # Sort by ROI
    data = data.sort_values('roi', ascending=False)

    # Accumulate until budget exhausted
    scenarios = []
    cumulative_cost = 0
    cumulative_save = 0
    partners_saved = 0

    for _, row in data.iterrows():
        if cumulative_cost + row['intervention_cost'] > budget:
            break

        cumulative_cost += row['intervention_cost']
        cumulative_save += row['expected_save']
        partners_saved += 1

        if partners_saved in [1, 3, 5, 10, 15, 20]:
            scenarios.append({
                'Partners': partners_saved,
                'Total Cost': f"${cumulative_cost:,.0f}",
                'Expected Value Saved': f"${cumulative_save:,.0f}",
                'Net Benefit': f"${cumulative_save - cumulative_cost:,.0f}",
                'ROI': f"{(cumulative_save - cumulative_cost) / cumulative_cost:.1f}x"
            })

    return pd.DataFrame(scenarios)


def main():
    """Business impact page."""
    if 'data' not in st.session_state:
        st.error("Please navigate from the main dashboard to load data.")
        st.stop()

    data = st.session_state.data
    exposure = calculate_exposure(data)

    st.title("Business Impact Analysis")
    st.markdown("Financial exposure analysis and intervention ROI projections.")

    st.markdown("---")

    # Key financial metrics
    st.subheader("Portfolio Financial Exposure")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Annual Portfolio Value",
            f"${exposure['total_annual_value']/1e6:.2f}M"
        )

    with col2:
        st.metric(
            "Total Value at Risk",
            f"${exposure['total_value_at_risk']/1e6:.2f}M",
            delta=f"{exposure['total_value_at_risk']/exposure['total_annual_value']*100:.0f}% of portfolio"
        )

    with col3:
        st.metric(
            "High-Risk Exposure",
            f"${exposure['high_risk_value']/1e3:.0f}K"
        )

    with col4:
        st.metric(
            "Expected Annual Loss",
            f"${exposure['total_expected_loss']/1e3:.0f}K"
        )

    st.markdown("---")

    # Exposure charts
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_exposure_waterfall(exposure), use_container_width=True)

    with col_right:
        # Exposure by region
        region_exposure = exposure['data'].groupby('region')['value_at_risk'].sum().reset_index()

        fig = px.pie(
            region_exposure,
            values='value_at_risk',
            names='region',
            title='Value at Risk by Region',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Intervention ROI Calculator
    st.subheader("Intervention ROI Calculator")
    st.markdown("Projected return on retention investment by partner priority.")

    col1, col2 = st.columns([1, 2])

    with col1:
        budget = st.slider(
            "Intervention Budget ($)",
            min_value=5000,
            max_value=50000,
            value=15000,
            step=1000,
            format="$%d"
        )

        st.markdown(f"""
        **Budget: ${budget:,}**

        This calculator optimizes partner selection by ROI,
        prioritizing partners where intervention has the
        highest expected return.
        """)

    with col2:
        scenarios = create_roi_projection(budget, data)

        if len(scenarios) > 0:
            st.dataframe(scenarios, use_container_width=True, hide_index=True)

            # Highlight best scenario
            if len(scenarios) > 0:
                best = scenarios.iloc[-1]
                st.success(f"""
                **Recommended Scenario**: Intervene with **{best['Partners']}** partners

                - Investment: {best['Total Cost']}
                - Expected Return: {best['Expected Value Saved']}
                - Net Benefit: {best['Net Benefit']}
                - ROI: {best['ROI']}
                """)
        else:
            st.warning("Budget too low for any interventions.")

    st.markdown("---")

    # Top intervention opportunities
    st.subheader("Top Intervention Opportunities")

    # Calculate and sort by value at risk
    top_opps = data.copy()
    top_opps['value_at_risk'] = top_opps['commission_current_month'] * 12 * top_opps['churn_probability']
    top_opps = top_opps.nlargest(10, 'value_at_risk')[
        ['partner_id', 'tier', 'region', 'churn_probability', 'commission_current_month', 'value_at_risk']
    ]

    top_opps['churn_probability'] = top_opps['churn_probability'].apply(lambda x: f"{x:.0%}")
    top_opps['commission_current_month'] = top_opps['commission_current_month'].apply(lambda x: f"${x:,.0f}")
    top_opps['value_at_risk'] = top_opps['value_at_risk'].apply(lambda x: f"${x:,.0f}")
    top_opps.columns = ['Partner ID', 'Tier', 'Region', 'Churn Risk', 'Monthly Commission', 'Value at Risk']

    st.dataframe(top_opps, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
