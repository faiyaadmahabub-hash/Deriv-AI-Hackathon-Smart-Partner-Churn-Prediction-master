"""
Table formatting utilities for the dashboard.
"""
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional


def format_currency(value: float, symbol: str = '$') -> str:
    """Format value as currency."""
    if value >= 1e6:
        return f"{symbol}{value/1e6:.1f}M"
    elif value >= 1e3:
        return f"{symbol}{value/1e3:.0f}K"
    else:
        return f"{symbol}{value:,.0f}"


def format_percent(value: float, decimals: int = 0) -> str:
    """Format value as percentage."""
    return f"{value*100:.{decimals}f}%"


def apply_risk_styling(df: pd.DataFrame, risk_col: str) -> pd.DataFrame:
    """
    Apply styling based on risk values.
    
    Args:
        df: DataFrame
        risk_col: Column name containing risk values
        
    Returns:
        Styled DataFrame
    """
    def color_risk(val):
        if isinstance(val, str):
            val = float(val.rstrip('%')) / 100
        
        if val >= 0.7:
            return 'background-color: #ffebee'
        elif val >= 0.4:
            return 'background-color: #fff3e0'
        else:
            return 'background-color: #e8f5e9'
    
    return df.style.applymap(color_risk, subset=[risk_col])


def create_summary_table(
    data: List[Dict],
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a summary table from list of dicts.
    
    Args:
        data: List of dicts
        columns: Optional column order
        
    Returns:
        DataFrame
    """
    df = pd.DataFrame(data)
    
    if columns:
        available_cols = [c for c in columns if c in df.columns]
        df = df[available_cols]
    
    return df


def display_paginated_table(
    df: pd.DataFrame,
    page_size: int = 10,
    key: str = 'table'
):
    """
    Display a paginated table with navigation.
    
    Args:
        df: DataFrame to display
        page_size: Rows per page
        key: Unique key for session state
    """
    if len(df) <= page_size:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    
    total_pages = (len(df) - 1) // page_size + 1
    
    # Page selector
    page = st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        key=f"{key}_page"
    )
    
    # Calculate slice
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    st.dataframe(
        df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(df))} of {len(df)} rows")


def highlight_top_n(
    df: pd.DataFrame,
    value_col: str,
    n: int = 5,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Highlight top N rows in a DataFrame.
    
    Args:
        df: DataFrame
        value_col: Column to sort by
        n: Number of rows to highlight
        ascending: Sort order
        
    Returns:
        Styled DataFrame
    """
    sorted_df = df.sort_values(value_col, ascending=ascending)
    top_indices = sorted_df.head(n).index.tolist()
    
    def highlight_row(row):
        if row.name in top_indices:
            return ['background-color: #ffffd0'] * len(row)
        return [''] * len(row)
    
    return df.style.apply(highlight_row, axis=1)
