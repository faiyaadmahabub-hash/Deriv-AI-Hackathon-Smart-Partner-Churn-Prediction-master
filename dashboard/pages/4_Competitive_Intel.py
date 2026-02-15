"""
Competitive Intelligence Page
Regional threat mapping and strategic response recommendations.
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

st.set_page_config(page_title="Competitive Intel - Partner Churn", page_icon="P", layout="wide")


def analyze_competitive_threats(data: pd.DataFrame) -> dict:
    """Analyze competitive threats from partner data."""
    # Get competitor mentions
    competitors = data['primary_competitor'].dropna()
    competitor_counts = competitors.value_counts()
    
    # Regional analysis
    regional_threats = {}
    for region in data['region'].unique():
        region_data = data[data['region'] == region]
        region_competitors = region_data['primary_competitor'].dropna()
        
        if len(region_competitors) > 0:
            top_competitor = region_competitors.mode().iloc[0] if len(region_competitors.mode()) > 0 else None
            affected = len(region_competitors)
            total = len(region_data)
            
            regional_threats[region] = {
                'top_competitor': top_competitor,
                'affected_partners': affected,
                'total_partners': total,
                'penetration': affected / total if total > 0 else 0,
                'avg_risk': region_data[region_data['primary_competitor'].notna()]['churn_probability'].mean()
            }
    
    return {
        'competitor_counts': competitor_counts,
        'regional_threats': regional_threats,
        'total_affected': len(competitors),
        'total_partners': len(data),
        'competitive_penetration': len(competitors) / len(data)
    }


def create_competitor_bar_chart(competitor_counts: pd.Series) -> go.Figure:
    """Create competitor mention bar chart."""
    fig = go.Figure(go.Bar(
        x=competitor_counts.values,
        y=competitor_counts.index,
        orientation='h',
        marker_color='#ff4b4b',
        text=competitor_counts.values,
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Partner Mentions by Competitor",
        xaxis_title="Number of Partners",
        yaxis_title="Competitor",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_regional_threat_map(regional_threats: dict) -> go.Figure:
    """Create regional threat heatmap."""
    regions = list(regional_threats.keys())
    competitors = list(set(r['top_competitor'] for r in regional_threats.values() if r['top_competitor']))
    
    # Create matrix
    data_matrix = []
    for region in regions:
        row = []
        region_data = regional_threats[region]
        for competitor in competitors:
            if region_data['top_competitor'] == competitor:
                row.append(region_data['penetration'] * 100)
            else:
                row.append(0)
        data_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=competitors,
        y=regions,
        colorscale='Reds',
        text=[[f"{v:.0f}%" if v > 0 else "" for v in row] for row in data_matrix],
        texttemplate="%{text}",
        hovertemplate="Region: %{y}<br>Competitor: %{x}<br>Penetration: %{z:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Competitor Penetration by Region",
        xaxis_title="Competitor",
        yaxis_title="Region",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def generate_strategic_recommendations(regional_threats: dict) -> list:
    """Generate strategic recommendations for each high-threat region."""
    recommendations = []
    
    for region, data in regional_threats.items():
        if data['penetration'] >= 0.2 or data.get('avg_risk', 0) >= 0.6:
            competitor = data['top_competitor']
            
            # Generate recommendation based on competitor
            competitor_strategies = {
                'Exness': {
                    'title': 'Rate Competitiveness Review',
                    'action': 'Conduct rate comparison and offer 5-10% commission boost',
                    'priority': 'HIGH'
                },
                'XM': {
                    'title': 'Platform Enhancement',
                    'action': 'Highlight unique platform features and offer training',
                    'priority': 'MEDIUM'
                },
                'IC Markets': {
                    'title': 'Professional Support Upgrade',
                    'action': 'Assign dedicated account managers to at-risk partners',
                    'priority': 'HIGH'
                },
                'Pepperstone': {
                    'title': 'Payment Terms Optimization',
                    'action': 'Offer faster payout cycles',
                    'priority': 'MEDIUM'
                },
                'FXTM': {
                    'title': 'Emerging Market Focus',
                    'action': 'Enhance local currency support',
                    'priority': 'MEDIUM'
                }
            }
            
            strategy = competitor_strategies.get(competitor, {
                'title': 'Retention Campaign',
                'action': 'Personal outreach to affected partners',
                'priority': 'MEDIUM'
            })
            
            recommendations.append({
                'region': region,
                'competitor': competitor,
                'affected_partners': data['affected_partners'],
                'penetration': f"{data['penetration']:.0%}",
                'recommendation': strategy['title'],
                'action': strategy['action'],
                'priority': strategy['priority']
            })
    
    return sorted(recommendations, key=lambda x: x['priority'] == 'HIGH', reverse=True)


def main():
    """Competitive intelligence page."""
    if 'data' not in st.session_state:
        st.error("Please navigate from the main dashboard to load data.")
        st.stop()
    
    data = st.session_state.data
    analysis = analyze_competitive_threats(data)
    
    st.title("Competitive Intelligence")
    st.markdown("Competitive threat monitoring and strategic response planning.")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Partners with Competitor Mentions",
            analysis['total_affected']
        )
    
    with col2:
        st.metric(
            "Competitive Penetration",
            f"{analysis['competitive_penetration']:.0%}"
        )
    
    with col3:
        if len(analysis['competitor_counts']) > 0:
            top_competitor = analysis['competitor_counts'].index[0]
            st.metric("Top Competitor Threat", top_competitor)
        else:
            st.metric("Top Competitor Threat", "N/A")
    
    with col4:
        high_threat_regions = sum(1 for r in analysis['regional_threats'].values() if r['penetration'] >= 0.25)
        st.metric("High-Threat Regions", high_threat_regions)
    
    st.markdown("---")
    
    # Charts
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        if len(analysis['competitor_counts']) > 0:
            st.plotly_chart(create_competitor_bar_chart(analysis['competitor_counts']), use_container_width=True)
        else:
            st.info("No competitor mentions detected in current data.")
    
    with col_right:
        if len(analysis['regional_threats']) > 0:
            st.plotly_chart(create_regional_threat_map(analysis['regional_threats']), use_container_width=True)
        else:
            st.info("No regional threat data available.")
    
    st.markdown("---")
    
    # Regional threat table
    st.subheader("Regional Threat Analysis")
    
    regional_df = pd.DataFrame([
        {
            'Region': region,
            'Top Competitor': data['top_competitor'] or 'N/A',
            'Affected Partners': data['affected_partners'],
            'Penetration': f"{data['penetration']:.0%}",
            'Avg Risk': f"{data.get('avg_risk', 0):.0%}",
            'Alert': 'HIGH' if data['penetration'] >= 0.25 else ('WATCH' if data['penetration'] >= 0.15 else 'OK')
        }
        for region, data in analysis['regional_threats'].items()
    ])
    
    if len(regional_df) > 0:
        st.dataframe(regional_df.sort_values('Affected Partners', ascending=False), 
                     use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Strategic recommendations
    st.subheader("Strategic Recommendations")
    
    recommendations = generate_strategic_recommendations(analysis['regional_threats'])
    
    if len(recommendations) > 0:
        for rec in recommendations:
            priority_color = "[!]" if rec['priority'] == 'HIGH' else "[-]"
            
            with st.expander(f"{priority_color} {rec['region']} - Counter {rec['competitor']}"):
                st.markdown(f"""
                **Region:** {rec['region']}  
                **Threat:** {rec['competitor']} ({rec['penetration']} penetration, {rec['affected_partners']} partners)
                
                **Recommendation:** {rec['recommendation']}
                
                **Action:** {rec['action']}
                
                **Priority:** {rec['priority']}
                """)
    else:
        st.success("No high-priority competitive threats detected.")


if __name__ == "__main__":
    main()
