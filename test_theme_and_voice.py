# test_theme_and_voice.py
"""
InsightPulse AI - Theme Contrast & Full Voice Conversation Test Suite
Validates Dark Mode high-contrast styling, Light Mode preservation,
Voice Conversation Center rendering, and Multilingual Voice Query handling.
"""

import sys
import os
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

from utils.data_loader import load_default_data
from utils.nlp_engine import analyze_dataset_sentiment
from utils.chatbot_engine import query_pulse_assistant
from utils.ui_components import (
    _get_plotly_layout,
    _clean_text_for_speech,
    create_sentiment_donut,
    create_rating_distribution,
    create_sentiment_trend_chart,
    create_product_sentiment_chart,
    create_rating_vs_sentiment,
    create_keyword_bars,
    create_journey_stage_chart
)

def run_tests():
    safe_print("=" * 65)
    safe_print("[TEST] Theme Contrast, Typing & Full Voice Verification...")
    safe_print("=" * 65)

    # 1. Test Theme Layouts
    safe_print("\n[STEP 1] Validating Plotly Layouts for Dark and Light Modes...")
    dark_cfg = _get_plotly_layout("dark")
    light_cfg = _get_plotly_layout("light")

    assert dark_cfg["template"] == "plotly_dark"
    assert dark_cfg["paper_bgcolor"] == "#151D2E"
    assert dark_cfg["font"]["color"] == "#F8FAFC"
    safe_print("  -> Dark Mode layout verified: High contrast (#F8FAFC on #151D2E)")

    assert light_cfg["template"] == "plotly_white"
    assert light_cfg["paper_bgcolor"] == "#FFFFFF"
    assert light_cfg["font"]["color"] == "#0F172A"
    safe_print("  -> Light Mode layout verified: Crisp typography (#0F172A on #FFFFFF)")

    # 2. Test All 7 Charts in Both Themes
    safe_print("\n[STEP 2] Verifying All 7 Plotly Charts in Both Themes...")
    df = analyze_dataset_sentiment(load_default_data())
    
    for theme in ["dark", "light"]:
        donut = create_sentiment_donut(df, theme=theme)
        assert donut is not None
        rating = create_rating_distribution(df, theme=theme)
        assert rating is not None
        trend = create_sentiment_trend_chart(df, theme=theme)
        assert trend is not None
        prod = create_product_sentiment_chart(df, theme=theme)
        assert prod is not None
        r_vs_s = create_rating_vs_sentiment(df, theme=theme)
        assert r_vs_s is not None
        k_pos, k_neg = create_keyword_bars([('quality', 10)], [('delay', 5)], theme=theme)
        assert k_pos is not None and k_neg is not None
        stage = create_journey_stage_chart(df, theme=theme)
        assert stage is not None
        safe_print(f"  -> All 7 charts generated successfully for theme: {theme.upper()}")

    # 3. Test Typed Question Processing (Requirement 1 & 3)
    safe_print("\n[STEP 3] Testing Typed Question Processing ('Why are customers unhappy?')...")
    typed_q = "Why are customers unhappy?"
    res_typed = query_pulse_assistant(typed_q, df)
    assert len(res_typed["answer"]) > 20
    assert "dissatisfaction" in res_typed["answer"].lower() or "unhappy" in res_typed["answer"].lower() or "negative" in res_typed["answer"].lower() or "issue" in res_typed["answer"].lower()
    safe_print(f"  -> Typed Question: \"{typed_q}\"")
    safe_print(f"     Pulse AI Answer: \"{res_typed['answer'][:85]}...\"")
    safe_print("  -> Typed question submitted and answered normally: PASS")

    # 4. Test Suggested Question Processing
    safe_print("\n[STEP 4] Testing Suggested Questions...")
    suggested_q = "Which product has the highest rating?"
    res_sugg = query_pulse_assistant(suggested_q, df)
    assert len(res_sugg["answer"]) > 20
    safe_print(f"  -> Suggested Question: \"{suggested_q}\"")
    safe_print(f"     Pulse AI Answer: \"{res_sugg['answer'][:85]}...\"")
    safe_print("  -> Suggested question processed: PASS")

    # 5. Test Full Voice Conversation Flow & Text-to-Speech Cleaning
    safe_print("\n[STEP 5] Simulating Voice Questions & Audio Synthesis Preparation...")
    voice_prompts = [
        ("English Voice Query", "Which product has the most negative reviews?", "en"),
        ("Hindi Voice Query", "सबसे ज्यादा negative reviews किस product के हैं?", "hi"),
        ("Hinglish Voice Query", "Sabse zyada negative reviews kis product ke hain?", "hinglish")
    ]

    for label, query, expected_lang in voice_prompts:
        res = query_pulse_assistant(query, df)
        assert res["language"] == expected_lang
        assert len(res["answer"]) > 30
        clean_audio_txt = _clean_text_for_speech(res["answer"])
        assert len(clean_audio_txt) > 0
        assert "*" not in clean_audio_txt  # Markdown bold removed
        assert "⚡" not in clean_audio_txt  # Emoji/symbols removed
        safe_print(f"  -> {label}: \"{query}\"")
        safe_print(f"     AI Voice Output Text: \"{clean_audio_txt[:70]}...\"")

    safe_print("\n" + "=" * 65)
    safe_print("ALL THEME, TYPING & VOICE TESTS PASSED SUCCESSFULLY! (5/5)")
    safe_print("=" * 65)

if __name__ == "__main__":
    run_tests()
