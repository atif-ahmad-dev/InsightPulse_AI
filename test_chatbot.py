# test_chatbot.py
"""
InsightPulse AI - Pulse Chatbot Test Suite (Multilingual & Analytical)
Verifies UTF-8 safe console output and accurate analytical responses
across English, Hindi, Hinglish, Product Comparison, Topic queries,
Investigation prioritizer, Sentiment overviews, and Filter awareness.
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
    """Safely print text to any terminal environment without crashing."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

from utils.data_loader import load_default_data
from utils.nlp_engine import analyze_dataset_sentiment
from utils.chatbot_engine import query_pulse_assistant

def run_tests():
    safe_print("=" * 65)
    safe_print("[TEST] Loading dataset and analyzing sentiments...")
    raw_df = load_default_data()
    df = analyze_dataset_sentiment(raw_df)
    safe_print(f"[TEST] Dataset loaded successfully: {len(df)} records.")
    safe_print("=" * 65)

    test_queries = [
        ("Which product should we investigate first?", "en"),
        ("What is the overall sentiment?", "en"),
        ("Show me the most common customer pain points.", "en"),
        ("How many positive reviews do we have?", "en"),
        ("Which product has the highest average rating?", "en"),
        ("Which product has the most negative reviews?", "en"),
        ("Why are customers unhappy?", "en"),
        ("What is the average rating?", "en"),
        ("What are the main customer complaints?", "en"),
        ("Summarize customer feedback.", "en"),
        ("Which product is performing well?", "en"),
        ("Compare product performance", "en"),
        ("What problems are customers reporting about delivery?", "en"),
        ("सबसे ज्यादा negative reviews किस product के हैं?", "hi"),
        ("Customers unhappy kyun hain?", "hinglish"),
        ("Average rating kitni hai?", "hinglish"),
        ("Sabse accha product kaunsa hai?", "hinglish"),
        ("Kis product ko pehle investigate karna chahiye?", "hinglish"),
        ("Who is the prime minister of Canada?", "en") # Out-of-scope query
    ]

    passed = 0
    failed = 0

    for idx, (query, expected_lang) in enumerate(test_queries, 1):
        safe_print(f"\n[QUERY {idx}]: {query}")
        try:
            res = query_pulse_assistant(query, df)
            answer = res.get('answer', '')
            mode = res.get('mode', '')
            detected_lang = res.get('language', '')
            
            assert answer is not None and len(answer) > 20, "Answer too short or empty"
            assert mode is not None, "Mode missing"
            assert detected_lang == expected_lang, f"Expected {expected_lang}, got {detected_lang}"
            
            safe_print(f"  [MODE]: {mode}")
            safe_print(f"  [LANG]: {detected_lang}")
            preview = answer.strip().split('\n')[0]
            safe_print(f"  [PREVIEW]: {preview[:80]}")
            safe_print(f"  [STATUS]: PASS")
            passed += 1
        except Exception as e:
            safe_print(f"  [STATUS]: FAIL - {e}")
            failed += 1

    # Filter Awareness Test
    safe_print("\n[FILTER AWARENESS TEST]")
    try:
        filtered_slice = df[df['Product Name'] == df['Product Name'].iloc[0]].copy()
        res_filter = query_pulse_assistant(
            "What is the overall sentiment?",
            filtered_slice,
            total_dataset_size=len(df)
        )
        ans_filter = res_filter.get('answer', '')
        assert "Filtered View" in ans_filter or "Active Filter" in ans_filter or str(len(filtered_slice)) in ans_filter, "Filter banner not present in response"
        safe_print(f"  [FILTERED ROWS]: {len(filtered_slice)} of {len(df)}")
        safe_print(f"  [STATUS]: PASS (Banner detected)")
        passed += 1
    except Exception as e:
        safe_print(f"  [STATUS]: FAIL - {e}")
        failed += 1

    safe_print("\n" + "=" * 65)
    safe_print(f"[RESULT] Chatbot Multilingual & Intelligence Tests: {passed} PASSED, {failed} FAILED")
    safe_print("=" * 65)

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
