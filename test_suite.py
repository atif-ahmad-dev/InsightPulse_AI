# test_suite.py
"""
InsightPulse AI - Full Integration Test Suite
Verifies data loading, NLP engine, single review analyzer, CX insights,
multilingual Pulse Assistant, and theme rendering compatibility.
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

from utils.data_loader import load_default_data, process_uploaded_file
from utils.nlp_engine import analyze_single_review, analyze_dataset_sentiment
from utils.insights_engine import compute_cx_insights
from utils.chatbot_engine import query_pulse_assistant
from utils.ui_components import (
    create_sentiment_donut,
    create_rating_distribution,
    create_product_sentiment_chart
)

def test_full_pipeline():
    safe_print("\n[TEST 1] Loading benchmark dataset...")
    df = load_default_data()
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    assert 'Review Text' in df.columns
    assert 'Rating' in df.columns
    assert 'Product Name' in df.columns
    safe_print("  -> Benchmark dataset loaded successfully: 500 rows.")

    safe_print("\n[TEST 2] Running batch NLP sentiment analysis...")
    scored_df = analyze_dataset_sentiment(df)
    assert 'sentiment' in scored_df.columns
    assert 'polarity' in scored_df.columns
    assert 'confidence' in scored_df.columns
    assert 'detected_aspect' in scored_df.columns
    assert 'detected_emotion' in scored_df.columns
    safe_print(f"  -> Batch analysis completed: {scored_df['sentiment'].value_counts().to_dict()}")

    safe_print("\n[TEST 3] Testing AI Single Review Analyzer...")
    test_cases = [
        "The product quality is good but delivery was very late.",
        "Customer support was amazing! Resolved my query in minutes.",
        "Complete waste of money. Product quality is extremely poor.",
        "Standard delivery service. Product arrived intact."
    ]
    for text in test_cases:
        res = analyze_single_review(text)
        assert res['sentiment'] in ['Positive', 'Neutral', 'Negative']
        assert 0 <= res['confidence'] <= 100
        assert res['insight'] is not None and len(res['insight']) > 10
        safe_print(f"  -> Input: \"{text[:32]}...\" -> {res['sentiment']} ({res['confidence']}%) | Issue: {res['detected_customer_issue']}")

    safe_print("\n[TEST 4] Computing CX Insights & Executive Narratives...")
    cx = compute_cx_insights(scored_df)
    assert cx['total_reviews'] == 500
    assert 'net_sentiment_score' in cx
    assert len(cx['executive_narratives']) >= 3
    safe_print(f"  -> Net Sentiment Score: {cx['net_sentiment_score']:+.1f}")
    safe_print(f"  -> Generated {len(cx['executive_narratives'])} executive narrative takeaways.")

    safe_print("\n[TEST 5] Verifying Pulse Chatbot in Multilingual Modes...")
    # English
    en_res = query_pulse_assistant("What is the average rating?", scored_df)
    assert "Overall Average Rating" in en_res['answer']
    safe_print("  -> English Query: PASSED")

    # Hindi
    hi_res = query_pulse_assistant("सबसे ज्यादा negative reviews किस product के हैं?", scored_df)
    assert hi_res['language'] == 'hi'
    assert len(hi_res['answer']) > 20
    safe_print("  -> Hindi Query: PASSED")

    # Hinglish
    hing_res = query_pulse_assistant("Customers unhappy kyun hain?", scored_df)
    assert hing_res['language'] == 'hinglish'
    assert len(hing_res['answer']) > 20
    safe_print("  -> Hinglish Query: PASSED")

    safe_print("\n[TEST 6] Verifying Theme-Aware Chart Generation (Dark & Light)...")
    fig_dark = create_sentiment_donut(scored_df, theme="dark")
    fig_light = create_sentiment_donut(scored_df, theme="light")
    assert fig_dark is not None
    assert fig_light is not None
    safe_print("  -> Chart theme switching: PASSED")

    safe_print("\n" + "=" * 65)
    safe_print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY! (6/6)")
    safe_print("=" * 65)

if __name__ == "__main__":
    test_full_pipeline()
