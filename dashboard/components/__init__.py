"""
Dashboard components module.
Provides reusable UI components for the Streamlit dashboard.
"""

from .charts import (
    create_risk_gauge,
    create_bar_chart,
    create_pie_chart,
    create_heatmap,
    create_line_chart
)

from .metrics import (
    metric_row,
    risk_status_badge,
    alert_card,
    info_card,
    progress_metric
)

from .tables import (
    format_currency,
    format_percent,
    apply_risk_styling,
    create_summary_table,
    display_paginated_table,
    highlight_top_n
)

__all__ = [
    # Charts
    'create_risk_gauge',
    'create_bar_chart',
    'create_pie_chart',
    'create_heatmap',
    'create_line_chart',
    # Metrics
    'metric_row',
    'risk_status_badge',
    'alert_card',
    'info_card',
    'progress_metric',
    # Tables
    'format_currency',
    'format_percent',
    'apply_risk_styling',
    'create_summary_table',
    'display_paginated_table',
    'highlight_top_n'
]
