# utils/ui_components.py
"""
InsightPulse AI - Enterprise SaaS UI & Visual Intelligence Components
Provides high-fidelity Light/Dark CSS themes, modern Robot Avatar identity,
theme-aware Plotly charts, and browser-native Voice Input/Output widgets.
"""

import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Tuple, Dict, Any

# SaaS Brand Colors
COLOR_POS = "#10B981"  # Emerald Green
COLOR_NEU = "#F59E0B"  # Amber Orange
COLOR_NEG = "#EF4444"  # Rose Red
COLOR_PRIMARY = "#3B82F6"  # Royal Blue
COLOR_ACCENT = "#6366F1"  # Indigo

COLOR_MAP = {
    'Positive': COLOR_POS,
    'Neutral': COLOR_NEU,
    'Negative': COLOR_NEG
}

def inject_custom_css(theme: str = "dark"):
    """
    Inject modern enterprise SaaS CSS styling with dynamic support
    for both Light Mode and Dark Mode. Overrides Streamlit root variables
    and every UI element to guarantee 100% contrast and readability.
    """
    if theme == "dark":
        root_text = "#F8FAFC"
        root_bg = "#0B0F19"
        root_sec_bg = "#151D2E"
        card_bg = "#151D2E"
        card_border = "rgba(255, 255, 255, 0.12)"
        text_primary = "#F8FAFC"
        text_secondary = "#CBD5E1"
        sidebar_bg = "#0F172A"
        sidebar_border = "rgba(255, 255, 255, 0.1)"
        insight_bg = "#162032"
        table_bg = "#151D2E"
        table_header_bg = "#1E293B"
        input_bg = "#1E293B"
        input_border = "rgba(255, 255, 255, 0.18)"
        divider_color = "rgba(255, 255, 255, 0.12)"
    else:
        root_text = "#0F172A"
        root_bg = "#F8FAFC"
        root_sec_bg = "#FFFFFF"
        card_bg = "#FFFFFF"
        card_border = "#E2E8F0"
        text_primary = "#0F172A"
        text_secondary = "#64748B"
        sidebar_bg = "#F1F5F9"
        sidebar_border = "#CBD5E1"
        insight_bg = "#F8FAFC"
        table_bg = "#FFFFFF"
        table_header_bg = "#F1F5F9"
        input_bg = "#FFFFFF"
        input_border = "#CBD5E1"
        divider_color = "#E2E8F0"

    css_styles = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Streamlit Root Variable Overrides */
    :root {{
        --text-color: {root_text} !important;
        --background-color: {root_bg} !important;
        --secondary-background-color: {root_sec_bg} !important;
    }}
    
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: {root_bg} !important;
        color: {text_primary} !important;
    }}

    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}

    /* Global Headings & Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-weight: 700 !important;
    }}

    p, span, label, li, a {{
        color: {text_primary};
    }}

    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] span, 
    [data-testid="stMarkdownContainer"] li {{
        color: {text_primary} !important;
    }}

    [data-testid="stMarkdownContainer"] strong {{
        color: {'#FFFFFF' if theme == 'dark' else '#0F172A'} !important;
        font-weight: 700 !important;
    }}

    .stCaption, [data-testid="stCaptionContainer"], small {{
        color: {text_secondary} !important;
    }}

    /* Horizontal Dividers */
    hr {{
        border-color: {divider_color} !important;
        margin: 1.5rem 0 !important;
    }}

    /* Brand Header Container */
    .brand-container {{
        background: linear-gradient(135deg, {'#0f172a 0%, #1e293b 100%' if theme == 'dark' else '#EFF6FF 0%, #DBEAFE 50%, #F8FAFC 100%'});
        padding: 2.2rem 2.6rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: {'0 12px 28px -6px rgba(0, 0, 0, 0.35)' if theme == 'dark' else '0 8px 24px -4px rgba(37, 99, 235, 0.08)'};
        border: 1px solid {'rgba(59, 130, 246, 0.25)' if theme == 'dark' else '#BFDBFE'};
        position: relative;
        overflow: hidden;
    }}
    .brand-container::after {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, {'rgba(59, 130, 246, 0.22)' if theme == 'dark' else 'rgba(37, 99, 235, 0.12)'} 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .brand-title {{
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: {'linear-gradient(90deg, #60a5fa, #38bdf8)' if theme == 'dark' else 'linear-gradient(90deg, #1d4ed8, #2563eb)'};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .brand-subtitle {{
        font-size: 1.12rem;
        color: {'#93C5FD' if theme == 'dark' else '#2563EB'} !important;
        margin-top: 0.45rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }}
    .brand-desc {{
        font-size: 0.92rem;
        color: {'#CBD5E1' if theme == 'dark' else '#475569'} !important;
        margin-top: 0.4rem;
        font-weight: 400;
        max-width: 820px;
        line-height: 1.55;
    }}

    /* KPI Metric Cards */
    .kpi-card {{
        background-color: {card_bg} !important;
        padding: 1.3rem 1.4rem;
        border-radius: 14px;
        border: 1px solid {card_border} !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
    }}
    .kpi-label {{
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: {text_secondary} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: {text_primary} !important;
        line-height: 1.15;
    }}
    .kpi-sub {{
        font-size: 0.78rem !important;
        color: {text_secondary} !important;
        margin-top: 0.4rem;
    }}

    /* Insight & Takeaway Cards */
    .insight-card {{
        background-color: {insight_bg} !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 1.1rem 1.3rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.9rem;
        font-size: 0.95rem;
        color: {text_primary} !important;
        border-top: 1px solid {card_border} !important;
        border-right: 1px solid {card_border} !important;
        border-bottom: 1px solid {card_border} !important;
    }}
    .insight-card * {{
        color: {text_primary} !important;
    }}

    /* Robot Assistant Identity Banner */
    .robot-banner {{
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.35) !important;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        display: flex;
        align-items: center;
        gap: 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.35);
    }}
    .robot-avatar-wrap {{
        flex-shrink: 0;
        width: 64px;
        height: 64px;
        background: radial-gradient(circle, #3b82f6 0%, #1e293b 80%);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.45);
        border: 1px solid #60a5fa;
    }}
    .robot-info-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .robot-info-sub {{
        font-size: 0.9rem;
        color: #94a3b8 !important;
        margin-top: 0.25rem;
    }}
    .robot-status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(16, 185, 129, 0.18);
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        margin-top: 0.35rem;
    }}
    .robot-status-dot {{
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }}

    /* Tables & DataFrames */
    table, .stTable, [data-testid="stTable"] table {{
        background-color: {table_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {card_border} !important;
        border-collapse: collapse !important;
        border-radius: 10px !important;
        width: 100% !important;
    }}
    table th, .stTable th, [data-testid="stTable"] th {{
        background-color: {table_header_bg} !important;
        color: {'#93C5FD' if theme == 'dark' else '#1E293B'} !important;
        font-weight: 700 !important;
        border-bottom: 2px solid {card_border} !important;
        padding: 10px 14px !important;
    }}
    table td, .stTable td, [data-testid="stTable"] td {{
        background-color: {table_bg} !important;
        color: {text_primary} !important;
        border-bottom: 1px solid {card_border} !important;
        padding: 9px 14px !important;
    }}
    table tr:hover td {{
        background-color: {'#1E293B' if theme == 'dark' else '#F1F5F9'} !important;
    }}

    /* Inputs, Selectboxes, Multiselects */
    input, textarea, select {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {input_border} !important;
        border-radius: 8px !important;
    }}
    ::placeholder, ::-webkit-input-placeholder, :-ms-input-placeholder {{
        color: {text_secondary} !important;
        opacity: 1 !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
        color: {text_primary} !important;
    }}
    div[data-baseweb="select"] * {{
        color: {text_primary} !important;
    }}
    div[data-baseweb="popover"] ul {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
    }}
    div[data-baseweb="popover"] li {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: {'#1E293B' if theme == 'dark' else '#E2E8F0'} !important;
        color: {'#60A5FA' if theme == 'dark' else '#1D4ED8'} !important;
    }}
    div[data-baseweb="tag"] {{
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }}
    div[data-baseweb="tag"] * {{
        color: #FFFFFF !important;
    }}

    /* Forms & In-Tab Typing Box */
    [data-testid="stForm"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 14px !important;
        padding: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }}
    [data-testid="stTextInput"] input {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {input_border} !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35) !important;
    }}

    /* Clean chat input: Hide permanent/large 'Press Enter to submit' instruction */
    [data-testid="InputInstructions"],
    .st-emotion-cache-1629p8f,
    [data-testid="stTextInput"] [data-testid="InputInstructions"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 0 !important;
    }}

    /* Buttons: Secondary / Suggested Questions */
    button[kind="secondary"], .stButton > button {{
        background-color: {'#1E293B' if theme == 'dark' else '#FFFFFF'} !important;
        color: {text_primary} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    button[kind="secondary"]:hover, .stButton > button:hover {{
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }}

    /* Primary Buttons & Form Submit Buttons (Send action) */
    button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] > button {{
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }}
    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 14px !important;
        color: {text_primary} !important;
        margin-bottom: 0.9rem !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
    }}
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] span, 
    [data-testid="stChatMessage"] li {{
        color: {text_primary} !important;
    }}
    [data-testid="stChatMessage"] strong {{
        color: {'#60A5FA' if theme == 'dark' else '#1D4ED8'} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stChatMessage"] em {{
        color: {text_secondary} !important;
    }}
    [data-testid="stChatMessage"] code {{
        background-color: {'#1E293B' if theme == 'dark' else '#F1F5F9'} !important;
        color: {'#38BDF8' if theme == 'dark' else '#0284C7'} !important;
        border: 1px solid {card_border} !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }}

    /* Prevent any fixed bottom overlay from covering dashboard */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {{
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }}
    
    /* Hide helper audio/TTS iframes cleanly without suspending JavaScript execution */
    iframe[height="0"],
    iframe[style*="height: 0"],
    iframe[height="1"],
    iframe[style*="height: 1px"],
    div[data-testid="stCustomComponentV1"]:has(iframe[height="0"]),
    div[data-testid="stCustomComponentV1"]:has(iframe[height="1"]) {{
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0.001 !important;
        pointer-events: none !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }}

    /* Right-Side Pulse AI Assistant Panel Container */
    .pulse-panel-card {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.2rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        margin-bottom: 1.2rem !important;
    }}
    .pulse-section-title {{
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: {text_primary} !important;
        margin-bottom: 0.4rem !important;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* Tabs Styling */
    button[data-baseweb="tab"] {{
        color: {text_secondary} !important;
        font-weight: 600 !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {'#60A5FA' if theme == 'dark' else '#2563EB'} !important;
        border-bottom-color: #3B82F6 !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {text_primary} !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {card_border} !important;
    }}

    /* Metrics Native */
    [data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {text_primary} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {text_secondary} !important;
    }}

    /* Footer Container */
    .footer-container {{
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        color: {text_secondary} !important;
        font-size: 0.85rem;
        border-top: 1px solid {divider_color} !important;
        margin-top: 3rem;
    }}
    .footer-container strong {{
        color: {text_primary} !important;
    }}
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)

def render_header(theme: str = "dark"):
    """Render the clean, premium commercial SaaS opening hero screen."""
    st.markdown(
        '<div class="brand-container">'
        '<div class="brand-title">⚡ InsightPulse AI</div>'
        '<div class="brand-subtitle">AI-Powered Customer Sentiment & Experience Analytics</div>'
        '<div class="brand-desc">Transform customer feedback into actionable business intelligence with AI-powered sentiment and customer experience analytics.</div>'
        '</div>',
        unsafe_allow_html=True
    )

def render_robot_assistant_header(theme: str = "dark", compact: bool = False):
    """Render the sleek AI Robot Avatar and Assistant Status identity."""
    svg_size = 32 if compact else 42
    robot_svg = (
        f'<svg width="{svg_size}" height="{svg_size}" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="50" cy="12" r="6" fill="#60A5FA" />'
        '<line x1="50" y1="18" x2="50" y2="28" stroke="#60A5FA" stroke-width="4" stroke-linecap="round"/>'
        '<rect x="20" y="28" width="60" height="46" rx="14" fill="#1E293B" stroke="#3B82F6" stroke-width="3.5"/>'
        '<path d="M 28 32 Q 50 29 72 32" stroke="rgba(255,255,255,0.2)" stroke-width="2" stroke-linecap="round"/>'
        '<rect x="28" y="38" width="44" height="18" rx="7" fill="#0F172A" stroke="#2563EB" stroke-width="1.5"/>'
        '<line x1="33" y1="47" x2="67" y2="47" stroke="#38BDF8" stroke-width="5" stroke-linecap="round"/>'
        '<circle cx="38" cy="47" r="2.5" fill="#FFFFFF"/>'
        '<circle cx="62" cy="47" r="2.5" fill="#FFFFFF"/>'
        '<line x1="38" y1="64" x2="42" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="46" y1="62" x2="54" y2="62" stroke="#38BDF8" stroke-width="3" stroke-linecap="round"/>'
        '<line x1="58" y1="64" x2="62" y2="64" stroke="#60A5FA" stroke-width="2.5" stroke-linecap="round"/>'
        '<rect x="14" y="42" width="6" height="18" rx="3" fill="#3B82F6"/>'
        '<rect x="80" y="42" width="6" height="18" rx="3" fill="#3B82F6"/>'
        '<path d="M 40 74 L 60 74 L 56 82 L 44 82 Z" fill="#334155"/>'
        '<circle cx="50" cy="78" r="2" fill="#38BDF8"/>'
        '</svg>'
    )

    if compact:
        banner_html = (
            '<div class="robot-banner" style="padding: 0.8rem 1rem; gap: 0.85rem; margin-bottom: 0.9rem; border-radius: 12px;">'
            f'<div class="robot-avatar-wrap" style="width: 46px; height: 46px; border-radius: 12px;">{robot_svg}</div>'
            '<div style="flex-grow: 1;">'
            '<div class="robot-info-title" style="font-size: 1.12rem;">Pulse AI Assistant 🤖</div>'
            '<div class="robot-info-sub" style="font-size: 0.78rem; margin-top: 0.15rem;">Multilingual Voice & Text Intelligence</div>'
            '<div class="robot-status-pill" style="font-size: 0.68rem; padding: 0.12rem 0.5rem; margin-top: 0.2rem;">'
            '<span class="robot-status-dot"></span> Online • Connected to Live Dataset'
            '</div>'
            '</div>'
            '</div>'
        )
    else:
        banner_html = (
            '<div class="robot-banner">'
            f'<div class="robot-avatar-wrap">{robot_svg}</div>'
            '<div style="flex-grow: 1;">'
            '<div class="robot-info-title">Pulse AI Assistant 🤖</div>'
            '<div class="robot-info-sub">Multilingual Voice & Text Intelligence</div>'
            '<div class="robot-status-pill">'
            '<span class="robot-status-dot"></span> Online • Connected to Live Dataset'
            '</div>'
            '</div>'
            '</div>'
        )
    st.markdown(banner_html, unsafe_allow_html=True)

def _clean_text_for_speech(text: str) -> str:
    """Format AI answer text for crisp, natural text-to-speech synthesis."""
    if not text:
        return ""
    import re
    # Strip mode badges, markdown, URLs, and emojis
    t = re.sub(r'\*+⚡[^*]+\*+', '', text)
    t = re.sub(r'[*_~`#>]', '', t)
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'[^\w\s,.\-?!:;\'"()/\u0900-\u097F]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
    return t[:480]

def render_voice_conversation_card(
    theme: str = "dark",
    latest_ai_answer: str = "",
    autoplay: bool = False,
    last_recognized: str = ""
):
    """
    Renders an interactive Voice Conversation Console:
    - 🎤 Speak Question (User Voice Input via Web Speech API).
    - Real-time speech transcription display before/while processing.
    - Automatic submission to Pulse AI.
    - Automatic AI Voice Read Aloud (Text-to-Speech) when triggered from voice.
    - Manual 'Listen to AI Answer' button to replay anytime.
    """
    card_bg = "#151D2E" if theme == "dark" else "#FFFFFF"
    text_c = "#F8FAFC" if theme == "dark" else "#0F172A"
    sub_c = "#94A3B8" if theme == "dark" else "#64748B"
    border_c = "rgba(255,255,255,0.16)" if theme == "dark" else "#CBD5E1"
    btn_bg = "#1E293B" if theme == "dark" else "#F1F5F9"
    
    clean_speech = _clean_text_for_speech(latest_ai_answer)
    clean_last_rec = last_recognized.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 2px;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: transparent;
            color: __TEXT_C__;
            overflow: hidden;
        }
        .voice-console {
            background: __CARD_BG__;
            border: 1px solid __BORDER_C__;
            border-radius: 14px;
            padding: 12px 18px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .console-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid __BORDER_C__;
            padding-bottom: 6px;
        }
        .console-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: __TEXT_C__;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .console-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }
        .mic-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .mic-btn {
            background: #2563EB;
            color: white;
            border: none;
            border-radius: 30px;
            padding: 8px 16px;
            font-size: 0.86rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
            transition: all 0.2s ease;
            outline: none;
        }
        .mic-btn:hover {
            background: #1D4ED8;
            transform: translateY(-1px);
        }
        .mic-btn.listening {
            background: #EF4444;
            animation: pulse-ring 1.5s infinite;
        }
        @keyframes pulse-ring {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .speaker-btn {
            background: #10B981;
            color: white;
            border: none;
            border-radius: 30px;
            padding: 8px 16px;
            font-size: 0.86rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);
            transition: all 0.2s ease;
            outline: none;
        }
        .speaker-btn:hover {
            background: #059669;
            transform: translateY(-1px);
        }
        .speaker-btn.speaking {
            background: #D97706;
            animation: pulse-speaker 1.5s infinite;
        }
        @keyframes pulse-speaker {
            0% { box-shadow: 0 0 0 0 rgba(217, 119, 6, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(217, 119, 6, 0); }
            100% { box-shadow: 0 0 0 0 rgba(217, 119, 6, 0); }
        }
        .speaker-btn:disabled {
            background: #64748B;
            opacity: 0.6;
            cursor: not-allowed;
        }
        .lang-select {
            background: __BTN_BG__;
            color: __TEXT_C__;
            border: 1px solid __BORDER_C__;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 0.82rem;
            outline: none;
            cursor: pointer;
            font-weight: 500;
        }
        .lang-select option {
            background: __BTN_BG__;
            color: __TEXT_C__;
        }
        .status-badge {
            font-size: 0.82rem;
            color: __TEXT_C__;
            padding: 6px 12px;
            border-radius: 8px;
            background: __BTN_BG__;
            border: 1px solid __BORDER_C__;
            width: 100%;
            min-height: 32px;
            line-height: 1.35;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
    </head>
    <body>
        <div class="voice-console">
            <div class="console-header">
                <div class="console-title">
                    <span>🎙️ Voice Conversation Center</span>
                </div>
                <div style="font-size: 0.75rem; color: __SUB_C__;">
                    Web Speech Recognition (Input) & Synthesis (Output)
                </div>
            </div>

            <div class="console-row">
                <!-- User Voice Input Group -->
                <div class="mic-group">
                    <button id="micBtn" class="mic-btn" title="Click to speak (User Voice Input)">
                        <span>🎤</span> <span id="micText">Speak Question</span>
                    </button>
                    <select id="langSelect" class="lang-select">
                        <option value="en-US">English (en-US)</option>
                        <option value="hi-IN">हिन्दी / Hinglish (hi-IN)</option>
                    </select>
                </div>

                <!-- AI Voice Output Group -->
                <div>
                    <button id="speakerBtn" class="speaker-btn" onclick="listenToAI(false)" title="Click to hear AI Response (Voice Output)">
                        <span>🔊</span> <span id="speakerText">Listen to AI Answer</span>
                    </button>
                </div>
            </div>

            <!-- Live Status & Transcript Display -->
            <div id="statusBox" class="status-badge">
                __INITIAL_STATUS__
            </div>
        </div>

        <script>
            function escapeHtml(text) {
                var div = document.createElement('div');
                div.appendChild(document.createTextNode(text));
                return div.innerHTML;
            }

            const micBtn = document.getElementById('micBtn');
            const micText = document.getElementById('micText');
            const speakerBtn = document.getElementById('speakerBtn');
            const statusBox = document.getElementById('statusBox');
            const langSelect = document.getElementById('langSelect');

            const aiSpeechText = "__AI_SPEECH_TEXT__";
            const shouldAutoplay = __SHOULD_AUTOPLAY__;
            const lastRecognized = "__LAST_RECOGNIZED__";

            if (!aiSpeechText || aiSpeechText.trim() === '') {
                speakerBtn.disabled = true;
                speakerBtn.title = 'No AI answer yet. Ask a question first!';
            }

            // Speech Recognition (USER VOICE INPUT)
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            let recognition = null;
            let isListening = false;

            if (SpeechRecognition) {
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = true;

                recognition.onstart = function() {
                    isListening = true;
                    micBtn.classList.add('listening');
                    micText.innerText = 'Listening...';
                    statusBox.innerHTML = '🔴 <strong>Listening...</strong> Speak your question clearly into your microphone.';
                };

                recognition.onresult = function(event) {
                    let interim = '';
                    let final = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            final = event.results[i][0].transcript;
                        } else {
                            interim += event.results[i][0].transcript;
                        }
                    }
                    const text = final || interim;
                    statusBox.innerHTML = '🎤 <strong>Captured:</strong> "' + escapeHtml(text) + '"';

                    if (final && final.trim().length > 1) {
                        statusBox.innerHTML = '⚡ <strong>Recognized:</strong> "' + escapeHtml(final.trim()) + '"<br><small style="color: #60A5FA;">Submitting to Pulse AI Assistant...</small>';
                        setTimeout(() => {
                            try {
                                const url = new URL(window.parent.location.href);
                                url.searchParams.set("voice_prompt", final.trim());
                                url.searchParams.set("auto_read", "1");
                                window.parent.location.href = url.href;
                            } catch (e) {
                                navigator.clipboard.writeText(final.trim());
                                statusBox.innerHTML = '📋 Question copied! Press Ctrl+V in the box below to ask.';
                            }
                        }, 600);
                    }
                };

                recognition.onerror = function(event) {
                    isListening = false;
                    micBtn.classList.remove('listening');
                    micText.innerText = 'Speak Question';
                    if (event.error === 'not-allowed' || event.error === 'permission-denied') {
                        statusBox.innerHTML = '⚠️ <strong>Microphone Permission Denied:</strong> Please click the camera/mic icon in your address bar to allow access, or type your question below.';
                    } else if (event.error === 'no-speech') {
                        statusBox.innerHTML = 'ℹ️ No speech detected. Click <strong>🎤 Speak Question</strong> to try again or type below.';
                    } else {
                        statusBox.innerHTML = 'ℹ️ Microphone status (' + event.error + '). You can freely type your question below.';
                    }
                };

                recognition.onend = function() {
                    isListening = false;
                    micBtn.classList.remove('listening');
                    micText.innerText = 'Speak Question';
                };

                micBtn.onclick = function() {
                    if (isListening) {
                        recognition.stop();
                    } else {
                        recognition.lang = langSelect.value;
                        try {
                            recognition.start();
                        } catch (e) {
                            recognition.stop();
                            setTimeout(() => recognition.start(), 200);
                        }
                    }
                };
            } else {
                statusBox.innerHTML = 'ℹ️ Web Speech API ready in Chrome, Edge, and Safari. You can freely type in the chat below.';
                micBtn.disabled = true;
                micBtn.style.opacity = '0.5';
            }

            // Speech Synthesis (AI VOICE OUTPUT)
            let isSpeaking = false;
            function listenToAI(isAuto) {
                if ('speechSynthesis' in window) {
                    if (isSpeaking) {
                        window.speechSynthesis.cancel();
                        isSpeaking = false;
                        speakerBtn.classList.remove('speaking');
                        document.getElementById('speakerText').innerText = 'Listen to AI Answer';
                        statusBox.innerHTML = 'Voice playback paused. Click <strong>🔊 Listen to AI Answer</strong> to replay.';
                        return;
                    }
                    if (!aiSpeechText || aiSpeechText.trim() === '') {
                        if (!isAuto) alert('No AI answer to read yet. Ask a question first!');
                        return;
                    }

                    window.speechSynthesis.cancel();
                    const utterance = new SpeechSynthesisUtterance(aiSpeechText);
                    utterance.rate = 1.0;
                    utterance.pitch = 1.0;

                    // Auto-detect Hindi script
                    if (/[\u0900-\u097F]/.test(aiSpeechText)) {
                        utterance.lang = 'hi-IN';
                    } else {
                        utterance.lang = 'en-US';
                    }

                    utterance.onstart = function() {
                        isSpeaking = true;
                        speakerBtn.classList.add('speaking');
                        document.getElementById('speakerText').innerText = '⏹️ Stop Voice';
                        statusBox.innerHTML = '🔊 <strong>Reading AI Answer Aloud...</strong> (Click Stop Voice to pause)';
                    };
                    utterance.onend = function() {
                        isSpeaking = false;
                        speakerBtn.classList.remove('speaking');
                        document.getElementById('speakerText').innerText = 'Listen to AI Answer';
                        statusBox.innerHTML = '✅ Answer read complete. Click <strong>🔊 Listen to AI Answer</strong> to replay.';
                    };
                    utterance.onerror = function() {
                        isSpeaking = false;
                        speakerBtn.classList.remove('speaking');
                        document.getElementById('speakerText').innerText = 'Listen to AI Answer';
                    };

                    window.speechSynthesis.speak(utterance);
                } else {
                    if (!isAuto) alert('Text-to-speech is not supported on this browser.');
                }
            }

            // Automatic Voice Output for Spoken Inquiries
            if (shouldAutoplay && aiSpeechText && aiSpeechText.trim() !== '') {
                setTimeout(function() {
                    listenToAI(true);
                }, 500);
            }
        </script>
    </body>
    </html>
    """
    
    if last_recognized:
        initial_status = f"🎤 <strong>Recognized Voice Question:</strong> \"{escape_html(last_recognized)}\" &bull; <span style='color: #34D399;'>Answer ready below!</span>"
    elif latest_ai_answer:
        initial_status = "✅ AI Answer Ready. Click <strong>🔊 Listen to AI Answer</strong> to hear it read aloud."
    else:
        initial_status = "Ready. Click <strong>🎤 Speak Question</strong> to talk to Pulse AI, or type in the box below."

    html_code = (
        html_code
        .replace("__CARD_BG__", card_bg)
        .replace("__TEXT_C__", text_c)
        .replace("__SUB_C__", sub_c)
        .replace("__BORDER_C__", border_c)
        .replace("__BTN_BG__", btn_bg)
        .replace("__AI_SPEECH_TEXT__", clean_speech)
        .replace("__SHOULD_AUTOPLAY__", "true" if autoplay else "false")
        .replace("__LAST_RECOGNIZED__", clean_last_rec)
        .replace("__INITIAL_STATUS__", initial_status)
    )
    
    components.html(html_code, height=160)

def escape_html(text: str) -> str:
    """Escape special HTML characters for safe template rendering."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

_VOICE_COMP_DIR = os.path.join(os.path.dirname(__file__), "voice_input_component")
_voice_input_component = components.declare_component("voice_input_widget", path=_VOICE_COMP_DIR)

def render_voice_input_widget(theme: str = "dark", key: str = "pulse_voice_widget"):
    """
    Renders the browser Web Speech Recognition component.
    Returns: dict with {"action": "voice_submit", "text": "...", "timestamp": int} when spoken.
    """
    try:
        return _voice_input_component(theme=theme, key=key, default=None)
    except Exception as e:
        return None

def trigger_browser_text_to_speech(text: str):
    """
    Executes browser Web Speech Synthesis for the given text.
    Safely cleans markdown/emojis and speaks using en-US or hi-IN.
    """
    if not text:
        return
    clean_audio = _clean_text_for_speech(text)
    if not clean_audio:
        return
    import json
    json_audio = json.dumps(clean_audio)
    tts_html = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="background:transparent; margin:0; padding:0; overflow:hidden;">
    <script>
    (function() {
        try {
            const toSpeak = __AUDIO_TEXT__;
            // Check top-level window controller first
            if (window.parent && window.parent.__pulseVoiceController && window.parent.__pulseVoiceController.speak) {
                window.parent.__pulseVoiceController.speak(toSpeak);
                return;
            }

            let synth = null;
            try {
                if (window.parent && window.parent.speechSynthesis) {
                    synth = window.parent.speechSynthesis;
                }
            } catch (frameErr) {}
            if (!synth) {
                synth = window.speechSynthesis;
            }
            if (synth) {
                const utterance = new SpeechSynthesisUtterance(toSpeak);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                if (/[\\u0900-\\u097F]/.test(toSpeak)) {
                    utterance.lang = "hi-IN";
                } else {
                    utterance.lang = "en-US";
                }
                const executeSpeak = function() {
                    try {
                        synth.cancel();
                        if (synth.paused) synth.resume();
                        synth.speak(utterance);
                    } catch (speakErr) {
                        console.error("SpeechSynthesis speak error:", speakErr);
                    }
                };
                if (synth.getVoices && synth.getVoices().length > 0) {
                    executeSpeak();
                } else {
                    synth.onvoiceschanged = executeSpeak;
                    setTimeout(executeSpeak, 150);
                }
            }
        } catch (e) {
            console.error("SpeechSynthesis error:", e);
        }
    })();
    </script>
    </body>
    </html>
    """.replace("__AUDIO_TEXT__", json_audio)
    components.html(tts_html, height=1)

def render_kpi_cards(
    total_reviews: int,
    avg_rating: float,
    pos_pct: float,
    neu_pct: float,
    neg_pct: float,
    total_products: int,
    theme: str = "dark",
    cols: int = 3
):
    """Render the 6 responsive executive KPI cards."""
    if cols == 3:
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        col1, col2, col3, col4, col5, col6 = r1_c1, r1_c2, r1_c3, r2_c1, r2_c2, r2_c3
    else:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Reviews</div><div class="kpi-value">{total_reviews:,}</div><div class="kpi-sub">Filtered Volume</div></div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Rating</div><div class="kpi-value">{avg_rating:.2f} <span style="font-size: 1.15rem; color: #f59e0b;">⭐</span></div><div class="kpi-sub">Scale 1.0 - 5.0</div></div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Positive %</div><div class="kpi-value" style="color: #10b981;">{pos_pct:.1f}%</div><div class="kpi-sub">Favorable Sentiment</div></div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Neutral %</div><div class="kpi-value" style="color: #f59e0b;">{neu_pct:.1f}%</div><div class="kpi-sub">Balanced Feedback</div></div>', unsafe_allow_html=True)
        
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Negative %</div><div class="kpi-value" style="color: #ef4444;">{neg_pct:.1f}%</div><div class="kpi-sub">Customer Friction</div></div>', unsafe_allow_html=True)
        
    with col6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Products</div><div class="kpi-value">{total_products}</div><div class="kpi-sub">Active Catalogs</div></div>', unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Dynamic Theme-Aware Plotly Visualization Charts
# -------------------------------------------------------------

def _get_plotly_layout(theme: str = "dark") -> Dict[str, Any]:
    """Helper to get high-contrast theme-aware colors for Plotly charts."""
    if theme == "dark":
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "#151D2E",
            "plot_bgcolor": "#151D2E",
            "font": {"color": "#F8FAFC", "family": "Inter, sans-serif"},
            "xaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "tickfont": {"color": "#CBD5E1"}, "title": {"font": {"color": "#F8FAFC"}}},
            "yaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "tickfont": {"color": "#CBD5E1"}, "title": {"font": {"color": "#F8FAFC"}}},
            "legend": {"font": {"color": "#F8FAFC"}}
        }
    else:
        return {
            "template": "plotly_white",
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "font": {"color": "#0F172A", "family": "Inter, sans-serif"},
            "xaxis": {"gridcolor": "#E2E8F0", "tickfont": {"color": "#475569"}, "title": {"font": {"color": "#0F172A"}}},
            "yaxis": {"gridcolor": "#E2E8F0", "tickfont": {"color": "#475569"}, "title": {"font": {"color": "#0F172A"}}},
            "legend": {"font": {"color": "#0F172A"}}
        }

def create_sentiment_donut(df: pd.DataFrame, theme: str = "dark"):
    """Modern donut chart with percentage labels, clear legend, and theme adaptation."""
    counts = df['sentiment'].value_counts()
    layout_cfg = _get_plotly_layout(theme)
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        title="<b>Sentiment Distribution</b>",
        color=counts.index,
        color_discrete_map=COLOR_MAP
    )
    fig.update_traces(
        textinfo='percent+label',
        textfont_size=13,
        marker=dict(line=dict(color='#151D2E' if theme == 'dark' else '#FFFFFF', width=2))
    )
    fig.update_layout(
        template=layout_cfg['template'],
        paper_bgcolor=layout_cfg['paper_bgcolor'],
        plot_bgcolor=layout_cfg['plot_bgcolor'],
        font=layout_cfg['font'],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=layout_cfg['legend']['font']),
        margin=dict(t=40, b=30, l=10, r=10),
        height=330
    )
    return fig

def create_rating_distribution(df: pd.DataFrame, theme: str = "dark"):
    """Bar chart showing count of reviews across ratings 1 to 5 with theme adaptation."""
    rating_counts = df['Rating'].value_counts().sort_index()
    layout_cfg = _get_plotly_layout(theme)
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="<b>Rating Distribution</b>",
        labels={'x': 'Rating (Stars)', 'y': 'Review Count'},
        color=rating_counts.values,
        color_continuous_scale='Blues'
    )
    avg_r = df['Rating'].mean()
    fig.add_vline(
        x=avg_r,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {avg_r:.2f}⭐",
        annotation_position="top right"
    )
    fig.update_layout(
        template=layout_cfg['template'],
        paper_bgcolor=layout_cfg['paper_bgcolor'],
        plot_bgcolor=layout_cfg['plot_bgcolor'],
        font=layout_cfg['font'],
        xaxis=layout_cfg['xaxis'],
        yaxis=layout_cfg['yaxis'],
        coloraxis_showscale=False,
        margin=dict(t=40, b=30, l=10, r=10),
        height=330
    )
    return fig

def create_sentiment_trend_chart(df: pd.DataFrame, theme: str = "dark"):
    """Timeline chart visualizing weekly sentiment progression."""
    if 'Date' not in df.columns or df['Date'].isna().all():
        return None
        
    df_trend = df.copy()
    df_trend['Date'] = pd.to_datetime(df_trend['Date'], errors='coerce')
    df_trend = df_trend.dropna(subset=['Date'])
    
    if df_trend.empty:
        return None
        
    layout_cfg = _get_plotly_layout(theme)
    df_trend['Period'] = df_trend['Date'].dt.to_period('W').dt.start_time
    trend_group = df_trend.groupby(['Period', 'sentiment']).size().reset_index(name='count')
    
    fig = px.line(
        trend_group,
        x='Period',
        y='count',
        color='sentiment',
        title="<b>Sentiment Volume Trend Over Time</b>",
        color_discrete_map=COLOR_MAP,
        markers=True
    )
    fig.update_layout(
        template=layout_cfg['template'],
        paper_bgcolor=layout_cfg['paper_bgcolor'],
        plot_bgcolor=layout_cfg['plot_bgcolor'],
        font=layout_cfg['font'],
        xaxis=layout_cfg['xaxis'],
        yaxis=layout_cfg['yaxis'],
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=layout_cfg['legend']['font']),
        margin=dict(t=40, b=30, l=10, r=10),
        height=330
    )
    return fig

def create_product_sentiment_chart(df: pd.DataFrame, top_n: int = 10, theme: str = "dark"):
    """Horizontal stacked bar chart showing sentiment breakdown by top products."""
    top_products = df['Product Name'].value_counts().head(top_n).index
    df_top = df[df['Product Name'].isin(top_products)]
    layout_cfg = _get_plotly_layout(theme)
    
    prod_sent = df_top.groupby(['Product Name', 'sentiment']).size().reset_index(name='count')
    
    fig = px.bar(
        prod_sent,
        y='Product Name',
        x='count',
        color='sentiment',
        title=f"<b>Sentiment by Top {len(top_products)} Products</b>",
        color_discrete_map=COLOR_MAP,
        orientation='h',
        barmode='stack'
    )
    fig.update_layout(
        template=layout_cfg['template'],
        paper_bgcolor=layout_cfg['paper_bgcolor'],
        plot_bgcolor=layout_cfg['plot_bgcolor'],
        font=layout_cfg['font'],
        xaxis=layout_cfg['xaxis'],
        yaxis=dict(categoryorder='total ascending', **layout_cfg['yaxis']),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=layout_cfg['legend']['font']),
        margin=dict(t=40, b=30, l=10, r=10),
        height=360
    )
    return fig

def create_rating_vs_sentiment(df: pd.DataFrame, theme: str = "dark"):
    """Grouped bar chart showing sentiment counts across each rating tier."""
    sent_rating = df.groupby(['Rating', 'sentiment']).size().reset_index(name='count')
    layout_cfg = _get_plotly_layout(theme)
    
    fig = px.bar(
        sent_rating,
        x='Rating',
        y='count',
        color='sentiment',
        title="<b>Rating vs. Sentiment Alignment</b>",
        color_discrete_map=COLOR_MAP,
        barmode='group'
    )
    fig.update_layout(
        template=layout_cfg['template'],
        paper_bgcolor=layout_cfg['paper_bgcolor'],
        plot_bgcolor=layout_cfg['plot_bgcolor'],
        font=layout_cfg['font'],
        xaxis=layout_cfg['xaxis'],
        yaxis=layout_cfg['yaxis'],
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=layout_cfg['legend']['font']),
        margin=dict(t=40, b=30, l=10, r=10),
        height=330
    )
    return fig

def create_keyword_bars(pos_keywords: List[Tuple[str, int]], neg_keywords: List[Tuple[str, int]], theme: str = "dark"):
    """Side-by-side keyword frequency bar charts with theme adaptation."""
    pos_df = pd.DataFrame(pos_keywords, columns=['Keyword', 'Frequency'])
    neg_df = pd.DataFrame(neg_keywords, columns=['Keyword', 'Frequency'])
    layout_cfg = _get_plotly_layout(theme)
    
    fig_pos = None
    if not pos_df.empty:
        fig_pos = px.bar(
            pos_df.head(10),
            x='Frequency',
            y='Keyword',
            orientation='h',
            title="<b>Top Positive Praise Keywords</b>",
            color_discrete_sequence=[COLOR_POS]
        )
        fig_pos.update_layout(
            template=layout_cfg['template'],
            paper_bgcolor=layout_cfg['paper_bgcolor'],
            plot_bgcolor=layout_cfg['plot_bgcolor'],
            font=layout_cfg['font'],
            xaxis=layout_cfg['xaxis'],
            yaxis=dict(categoryorder='total ascending', **layout_cfg['yaxis']),
            height=300,
            margin=dict(t=40, b=20, l=10, r=10)
        )
        
    fig_neg = None
    if not neg_df.empty:
        fig_neg = px.bar(
            neg_df.head(10),
            x='Frequency',
            y='Keyword',
            orientation='h',
            title="<b>Top Negative Pain-Point Keywords</b>",
            color_discrete_sequence=[COLOR_NEG]
        )
        fig_neg.update_layout(
            template=layout_cfg['template'],
            paper_bgcolor=layout_cfg['paper_bgcolor'],
            plot_bgcolor=layout_cfg['plot_bgcolor'],
            font=layout_cfg['font'],
            xaxis=layout_cfg['xaxis'],
            yaxis=dict(categoryorder='total ascending', **layout_cfg['yaxis']),
            height=300,
            margin=dict(t=40, b=20, l=10, r=10)
        )
        
    return fig_pos, fig_neg

def create_journey_stage_chart(df: pd.DataFrame, theme: str = "dark"):
    """Bar chart of customer touchpoint/journey stage sentiment breakdown."""
    if 'Journey Stage' not in df.columns:
        return None
        
    stage_sent = df.groupby(['Journey Stage', 'sentiment']).size().reset_index(name='count')
    layout_cfg = _get_plotly_layout(theme)
    
    fig = px.bar(
        stage_sent,
        x='Journey Stage',
        y='count',
        color='sentiment',
        title="<b>Sentiment Breakdown by Customer Journey Stage</b>",
        color_discrete_map=COLOR_MAP,
        barmode='stack'
    )
    fig.update_layout(
        template=layout_cfg['template'],
        paper_bgcolor=layout_cfg['paper_bgcolor'],
        plot_bgcolor=layout_cfg['plot_bgcolor'],
        font=layout_cfg['font'],
        xaxis=layout_cfg['xaxis'],
        yaxis=layout_cfg['yaxis'],
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=layout_cfg['legend']['font']),
        margin=dict(t=40, b=30, l=10, r=10),
        height=330
    )
    return fig

def render_footer(theme: str = "dark"):
    """Render professional enterprise footer."""
    st.markdown(
        '<div class="footer-container">'
        '<strong>InsightPulse AI</strong> — AI-Powered Customer Sentiment & Experience Analytics<br>'
        '<span style="font-size: 0.8rem;">Enterprise Data Science & NLP Architecture • Built with Streamlit, Plotly & Multilingual NLP Intelligence</span>'
        '</div>',
        unsafe_allow_html=True
    )
