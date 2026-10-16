"""Shared UI helpers for WalmartCast Streamlit app."""
import os
import streamlit as st
import plotly.graph_objects as go


def load_css():
    """Inject app/style.css into the page."""
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ''):
    """Render a consistent page header."""
    st.markdown(f'# {title}')
    if subtitle:
        st.markdown(
            f'<p style="color:#64748B;font-size:1rem;margin-top:-0.5rem;margin-bottom:1rem">{subtitle}</p>',
            unsafe_allow_html=True
        )


def kpi_row(items: list):
    """Render a row of KPI cards. Each item is a dict with label, value, and optional delta/help."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            with st.container(border=True):
                st.metric(
                    label=item['label'],
                    value=item['value'],
                    delta=item.get('delta'),
                    help=item.get('help')
                )


def callout(kind: str, text: str):
    """Render an info/warning/error callout box."""
    colors = {'info': '#EFF6FF', 'warning': '#FFFBEB', 'error': '#FEF2F2'}
    borders = {'info': '#BFDBFE', 'warning': '#FDE68A', 'error': '#FECACA'}
    icons = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌'}
    bg = colors.get(kind, colors['info'])
    border = borders.get(kind, borders['info'])
    icon = icons.get(kind, icons['info'])
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {border};'
        f'padding:0.75rem 1rem;border-radius:0 8px 8px 0;margin:1rem 0">'
        f'{icon} {text}</div>',
        unsafe_allow_html=True
    )


def chart_layout(fig: go.Figure) -> go.Figure:
    """Apply shared Plotly styling to a figure."""
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Inter, system-ui, sans-serif', color='#0F172A'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        hoverlabel=dict(font_family='Inter, sans-serif'),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#F1F5F9', gridwidth=1,
        linecolor='#E2E8F0', tickfont=dict(size=11)
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#F1F5F9', gridwidth=1,
        linecolor='#E2E8F0', tickfont=dict(size=11),
        tickformat='$,.0f'
    )
    # Apply thousands separators to hover text
    fig.update_traces(hovertemplate='%{y:$,.0f}<extra></extra>')
    return fig
