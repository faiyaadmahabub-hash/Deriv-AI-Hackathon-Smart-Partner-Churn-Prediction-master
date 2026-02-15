"""
Metric card components for the dashboard.
"""
import streamlit as st
from typing import Optional


def metric_row(metrics: list):
    """
    Display a row of metrics.
    
    Args:
        metrics: List of dicts with label, value, delta, delta_color
    """
    cols = st.columns(len(metrics))
    
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal')
            )


def risk_status_badge(risk_tier: str) -> str:
    """
    Return text badge for risk tier.

    Args:
        risk_tier: 'critical', 'high', 'medium', 'low', or 'minimal'

    Returns:
        Badge string
    """
    badges = {
        'critical': '[!!]',
        'high': '[!]',
        'medium': '[-]',
        'low': '[ok]',
        'minimal': '[--]'
    }
    return badges.get(risk_tier.lower(), '[--]')


def alert_card(title: str, message: str, alert_type: str = 'info'):
    """
    Display a styled alert card.
    
    Args:
        title: Alert title
        message: Alert message
        alert_type: 'info', 'success', 'warning', 'error'
    """
    if alert_type == 'success':
        st.success(f"**{title}**\n\n{message}")
    elif alert_type == 'warning':
        st.warning(f"**{title}**\n\n{message}")
    elif alert_type == 'error':
        st.error(f"**{title}**\n\n{message}")
    else:
        st.info(f"**{title}**\n\n{message}")


def info_card(title: str, items: dict):
    """
    Display an information card with key-value pairs.
    
    Args:
        title: Card title
        items: Dict of label-value pairs
    """
    st.markdown(f"### {title}")
    
    for label, value in items.items():
        st.markdown(f"**{label}:** {value}")


def progress_metric(
    label: str,
    current: float,
    target: float,
    format_str: str = "{:.0%}"
):
    """
    Display a metric with progress bar.
    
    Args:
        label: Metric label
        current: Current value
        target: Target value
        format_str: Format string for display
    """
    progress = min(current / target, 1.0) if target > 0 else 0
    
    st.markdown(f"**{label}**")
    st.progress(progress)
    st.caption(f"{format_str.format(current)} / {format_str.format(target)}")
