"""
Reusable chart components for the dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Optional


def create_risk_gauge(probability: float, title: str = "Churn Risk") -> go.Figure:
    """
    Create a gauge chart for risk visualization.
    
    Args:
        probability: Risk probability (0-1)
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    if probability >= 0.7:
        color = "#ff4b4b"
        risk_label = "HIGH"
    elif probability >= 0.4:
        color = "#ffa500"
        risk_label = "MEDIUM"
    else:
        color = "#00cc66"
        risk_label = "LOW"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': f"{title} - {risk_label}", 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': '#e8f5e9'},
                {'range': [40, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    orientation: str = 'v'
) -> go.Figure:
    """
    Create a standardized bar chart.
    
    Args:
        df: Data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        color: Color column (optional)
        orientation: 'v' for vertical, 'h' for horizontal
        
    Returns:
        Plotly Figure
    """
    fig = px.bar(
        df,
        x=x if orientation == 'v' else y,
        y=y if orientation == 'v' else x,
        color=color,
        title=title,
        orientation=orientation
    )
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    values: str,
    names: str,
    title: str = "",
    colors: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a standardized pie/donut chart.
    
    Args:
        df: Data
        values: Values column
        names: Names/labels column
        title: Chart title
        colors: Optional color list
        
    Returns:
        Plotly Figure
    """
    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        color_discrete_sequence=colors,
        hole=0.4
    )
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    values: str,
    title: str = "",
    colorscale: str = 'RdYlGn_r'
) -> go.Figure:
    """
    Create a standardized heatmap.
    
    Args:
        df: Data (should be pivoted or pivot table)
        x: X-axis column
        y: Y-axis column
        values: Values column
        title: Chart title
        colorscale: Plotly colorscale name
        
    Returns:
        Plotly Figure
    """
    pivot = df.pivot_table(values=values, index=y, columns=x, aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=colorscale,
        text=[[f"{v:.0%}" for v in row] for row in pivot.values],
        texttemplate="%{text}"
    ))
    
    fig.update_layout(
        title=title,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str = '#1f77b4'
) -> go.Figure:
    """
    Create a standardized line chart.
    
    Args:
        df: Data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        color: Line color
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure(go.Scatter(
        x=df[x],
        y=df[y],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
