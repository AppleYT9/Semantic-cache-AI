"""
UI Design System and Visual Components for SemanticCache AI Dashboard.
Implements the clean, high-contrast Scientific Ice-Blue / Minimalist Light theme:
- Soft Ice-Blue canvas background (#EAF4F9)
- Pure White card containers (#FFFFFF) with subtle hairline borders (#E2EDF4)
- Deep charcoal headings (#0F172A) and muted slate body (#475569)
- High-contrast visual accents: Cerulean Blue (#0284C7), Golden Amber (#F59E0B), Coral Red (#EF4444), Mint Green (#10B981)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

CUSTOM_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
    }

    /* Page Canvas */
    .stApp {
        background-color: #eaf4f9 !important;
    }

    code, pre, .stCodeBlock {
        font-family: Consolas, "Courier New", monospace !important;
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1300px !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2edf4 !important;
        box-shadow: 2px 0 12px rgba(15, 23, 42, 0.02) !important;
    }

    /* Clean Card Container */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2edf4;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 14px;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -4px rgba(2, 132, 199, 0.08);
        border-color: #bae6fd;
    }

    .kpi-title {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
        color: #0f172a;
    }

    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Clean Badge Pills */
    .badge-hit {
        display: inline-flex;
        align-items: center;
        padding: 5px 14px;
        border-radius: 9999px;
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }

    .badge-miss {
        display: inline-flex;
        align-items: center;
        padding: 5px 14px;
        border-radius: 9999px;
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }

    .badge-tier1 {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 9999px;
        background: #f0f9ff;
        color: #0284c7;
        border: 1px solid #bae6fd;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-tier2 {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 9999px;
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Minimalist Top Hero Header */
    .hero-banner {
        background: #ffffff;
        border: 1px solid #e2edf4;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
    }

    .hero-tag {
        font-size: 0.74rem;
        font-weight: 800;
        color: #0284c7;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 4px;
        display: block;
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
        font-weight: 400;
    }

    /* Form Input & Button Styling */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 10px !important;
    }

    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.15) !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
    }

    /* Primary Red / Accent Process Button */
    button[kind="primary"] {
        background: #0284c7 !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    }

    button[kind="primary"]:hover {
        background: #0369a1 !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.4) !important;
        transform: translateY(-1px);
    }
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#334155", size=12),
    margin=dict(l=40, r=30, t=50, b=40),
    xaxis=dict(
        gridcolor="#e2edf4",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
        tickfont=dict(color="#475569")
    ),
    yaxis=dict(
        gridcolor="#e2edf4",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
        tickfont=dict(color="#475569")
    ),
    legend=dict(
        bgcolor="#ffffff",
        bordercolor="#e2edf4",
        borderwidth=1,
        font=dict(color="#334155")
    )
)

def render_kpi_card(title: str, value: str, subtext: str = "", border_color: Optional[str] = None):
    """Render a styled KPI metric container in crisp white."""
    border_style = f"border-color: {border_color};" if border_color else ""
    st.markdown(f"""
        <div class="kpi-card" style="{border_style}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtext">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

def render_hero(title: str, subtitle: str, tag: str = "PRODUCTION READY"):
    """Render page top hero header."""
    st.markdown(f"""
        <div class="hero-banner">
            <span class="hero-tag">— {tag}</span>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)
