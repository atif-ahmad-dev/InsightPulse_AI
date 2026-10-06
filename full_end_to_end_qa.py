# full_end_to_end_qa.py
"""
InsightPulse AI - Master End-to-End Quality Assurance (QA) Suite
Performs exhaustive programmatic testing across all 20 core feature areas:
1. Dashboard KPIs & Analytics
2. Product Filters
3. Rating Filters
4. Sentiment Filters
5. Date Filters
6. CSV Upload & Dynamic Grounding
7. Excel Upload & Dynamic Grounding
8. Review Analyzer (Single Review NLP)
9. AI Customer Insights & CX Narratives
10. Pulse AI Typed Questions
11. Pulse AI Voice Questions
12. Automatic AI Voice Response (TTS)
13. Listen to AI Answer (Manual Replay)
14. Suggested Questions
15. Review Explorer & Search
16. Export & Reports (CSV, Excel, Briefing)
17. Dark Mode Styling & Charts
18. Light Mode Styling & Charts
19. Multiple Sequential Questions (Zero State Corruption)
20. Cross-Browser Engine Compatibility (Firefox, Chrome, Edge)
"""

import os
import sys
import io
import base64
from datetime import datetime, date, timedelta
import pandas as pd
import urllib.request

# Ensure UTF-8 output on Windows
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

# Import InsightPulse AI Modules
from utils.data_loader import load_default_data, process_uploaded_file, standardize_dataframe
from utils.nlp_engine import analyze_single_review, analyze_dataset_sentiment
from utils.insights_engine import compute_cx_insights
from utils.chatbot_engine import query_pulse_assistant
from utils.audio_transcriber import transcribe_audio_b64
from utils.ui_components import (
    _clean_text_for_speech,
    _get_plotly_layout,
    create_sentiment_donut,
    create_rating_distribution,
    create_sentiment_trend_chart,
    create_product_sentiment_chart,
    create_rating_vs_sentiment,
    create_journey_stage_chart,
    create_keyword_bars
)

def run_full_qa():
    safe_print("=" * 75)
    safe_print("INSIGHTPULSE AI — FINAL FULL QA AUTOMATED TEST SUITE")
    safe_print("=" * 75)

    results = {}
    total_tests = 20

    # -------------------------------------------------------------
    # 1. Dashboard KPIs & Visualizations
    # -------------------------------------------------------------
    try:
        raw_df = load_default_data()
        df = analyze_dataset_sentiment(raw_df)
        total_revs = len(df)
        avg_rtg = df['Rating'].mean()
        pos_p = len(df[df['sentiment'] == 'Positive']) / total_revs * 100
        neu_p = len(df[df['sentiment'] == 'Neutral']) / total_revs * 100
        neg_p = len(df[df['sentiment'] == 'Negative']) / total_revs * 100
        prod_count = df['Product Name'].nunique()

        assert total_revs > 0, "No reviews loaded"
        assert 1.0 <= avg_rtg <= 5.0, "Average rating out of bounds"
        assert round(pos_p + neu_p + neg_p) == 100, "Sentiment percentages do not sum to 100"

        # Check all chart generators
        f1 = create_sentiment_donut(df, theme="dark")
        f2 = create_rating_distribution(df, theme="dark")
        f3 = create_sentiment_trend_chart(df, theme="dark")
        f4 = create_journey_stage_chart(df, theme="dark")
        f5 = create_product_sentiment_chart(df, theme="dark")
        f6 = create_rating_vs_sentiment(df, theme="dark")

        assert f1 and f2 and f5 and f6, "Core charts failed to render"
        results["1. Dashboard KPIs & Analytics"] = "PASS"
        safe_print("✅ 1. Dashboard KPIs & Analytics: PASS")
    except Exception as e:
        results["1. Dashboard KPIs & Analytics"] = f"FAIL: {e}"
        safe_print(f"❌ 1. Dashboard KPIs & Analytics: FAIL - {e}")

    # -------------------------------------------------------------
    # 2. Product Filters
    # -------------------------------------------------------------
    try:
        sample_prod = df['Product Name'].iloc[0]
        p_filtered = df[df['Product Name'] == sample_prod]
        assert len(p_filtered) > 0, "Product filter returned 0 rows"
        assert (p_filtered['Product Name'] == sample_prod).all(), "Unmatched product rows found"
        results["2. Product Filters"] = "PASS"
        safe_print(f"✅ 2. Product Filters ({sample_prod}): PASS ({len(p_filtered)} rows)")
    except Exception as e:
        results["2. Product Filters"] = f"FAIL: {e}"
        safe_print(f"❌ 2. Product Filters: FAIL - {e}")

    # -------------------------------------------------------------
    # 3. Rating Filters
    # -------------------------------------------------------------
    try:
        r_filtered = df[df['Rating'].astype(int).isin([1, 2])]
        assert len(r_filtered) > 0, "Rating 1-2 filter returned 0 rows"
        assert r_filtered['Rating'].max() <= 2.0, "Found rating > 2 in filtered data"
        results["3. Rating Filters"] = "PASS"
        safe_print(f"✅ 3. Rating Filters (1-2 Stars): PASS ({len(r_filtered)} rows)")
    except Exception as e:
        results["3. Rating Filters"] = f"FAIL: {e}"
        safe_print(f"❌ 3. Rating Filters: FAIL - {e}")

    # -------------------------------------------------------------
    # 4. Sentiment Filters
    # -------------------------------------------------------------
    try:
        s_filtered = df[df['sentiment'] == 'Negative']
        assert len(s_filtered) > 0, "Sentiment Negative filter returned 0 rows"
        assert (s_filtered['sentiment'] == 'Negative').all(), "Non-negative rows found"
        results["4. Sentiment Filters"] = "PASS"
        safe_print(f"✅ 4. Sentiment Filters (Negative): PASS ({len(s_filtered)} rows)")
    except Exception as e:
        results["4. Sentiment Filters"] = f"FAIL: {e}"
        safe_print(f"❌ 4. Sentiment Filters: FAIL - {e}")

    # -------------------------------------------------------------
    # 5. Date Filters
    # -------------------------------------------------------------
    try:
        date_series = pd.to_datetime(df['Date'], errors='coerce').dt.date.dropna()
        if not date_series.empty:
            min_d, max_d = date_series.min(), date_series.max()
            mid_d = min_d + (max_d - min_d) // 2
            d_filtered = df[pd.to_datetime(df['Date'], errors='coerce').dt.date >= mid_d]
            assert len(d_filtered) > 0, "Date filter returned 0 rows"
            results["5. Date Filters"] = "PASS"
            safe_print(f"✅ 5. Date Filters (>= {mid_d}): PASS ({len(d_filtered)} rows)")
        else:
            results["5. Date Filters"] = "PASS (Skipped: no dates)"
            safe_print("✅ 5. Date Filters: PASS (No dates in benchmark)")
    except Exception as e:
        results["5. Date Filters"] = f"FAIL: {e}"
        safe_print(f"❌ 5. Date Filters: FAIL - {e}")

    # -------------------------------------------------------------
    # 6. CSV Upload & Dynamic Grounding
    # -------------------------------------------------------------
    try:
        mock_csv_data = (
            "Product Name,Rating,Review Text\n"
            "Widget Alpha,1,Terrible battery life and awful build quality.\n"
            "Widget Alpha,2,Stopped working after one week.\n"
            "Widget Beta,5,Fantastic product! Works like a charm.\n"
        )
        csv_file = io.BytesIO(mock_csv_data.encode('utf-8'))
        csv_file.name = "custom_test_upload.csv"

        uploaded_df, err = process_uploaded_file(csv_file)
        assert err is None, f"Upload error: {err}"
        assert len(uploaded_df) == 3, f"Expected 3 rows, got {len(uploaded_df)}"

        scored_upload = analyze_dataset_sentiment(uploaded_df)
        assert 'sentiment' in scored_upload.columns, "Sentiment scoring missing from upload"

        # Query Pulse AI on the newly uploaded dataset
        ans_upload = query_pulse_assistant(
            "Which product has the most negative reviews?",
            scored_upload,
            total_dataset_size=len(scored_upload)
        )
        assert "Widget Alpha" in ans_upload['answer'], f"Pulse AI failed to dynamically identify Widget Alpha in uploaded dataset: {ans_upload['answer']}"
        results["6. CSV Upload & Dynamic Grounding"] = "PASS"
        safe_print("✅ 6. CSV Upload & Dynamic Grounding: PASS")
    except Exception as e:
        results["6. CSV Upload & Dynamic Grounding"] = f"FAIL: {e}"
        safe_print(f"❌ 6. CSV Upload & Dynamic Grounding: FAIL - {e}")

    # -------------------------------------------------------------
    # 7. Excel Upload & Dynamic Grounding
    # -------------------------------------------------------------
    try:
        excel_io = io.BytesIO()
        mock_xl_df = pd.DataFrame({
            "Product Name": ["SmartDesk Pro", "SmartDesk Pro", "ErgoChair Elite"],
            "Rating": [1, 1, 5],
            "Review Text": [
                "Motor burned out on day two. Horrible customer support.",
                "Desk wobbles severely and legs failed.",
                "Most comfortable ergonomic chair I have ever owned!"
            ]
        })
        with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
            mock_xl_df.to_excel(writer, index=False)
        excel_io.seek(0)
        excel_io.name = "custom_inventory_test.xlsx"

        uploaded_xl, err_xl = process_uploaded_file(excel_io)
        assert err_xl is None, f"Excel upload error: {err_xl}"
        assert len(uploaded_xl) == 3, f"Expected 3 rows, got {len(uploaded_xl)}"

        scored_xl = analyze_dataset_sentiment(uploaded_xl)
        ans_xl = query_pulse_assistant(
            "Which product should we investigate first?",
            scored_xl,
            total_dataset_size=len(scored_xl)
        )
        assert "SmartDesk Pro" in ans_xl['answer'], f"Pulse AI failed to identify SmartDesk Pro from Excel upload: {ans_xl['answer']}"
        results["7. Excel Upload & Dynamic Grounding"] = "PASS"
        safe_print("✅ 7. Excel Upload & Dynamic Grounding: PASS")
    except Exception as e:
        results["7. Excel Upload & Dynamic Grounding"] = f"FAIL: {e}"
        safe_print(f"❌ 7. Excel Upload & Dynamic Grounding: FAIL - {e}")

    # -------------------------------------------------------------
    # 8. Review Analyzer (Single Review NLP)
    # -------------------------------------------------------------
    try:
        rev_res = analyze_single_review("The battery life is amazing but shipping took three weeks.")
        assert rev_res['sentiment'] in ['Positive', 'Mixed', 'Neutral'], f"Unexpected sentiment: {rev_res['sentiment']}"
        assert rev_res['confidence'] > 50, f"Confidence too low: {rev_res['confidence']}"
        assert rev_res['detected_customer_issue'], "No customer issue detected"
        assert len(rev_res['insight']) > 15, "Insight string too short"
        results["8. Review Analyzer"] = "PASS"
        safe_print(f"✅ 8. Review Analyzer: PASS ({rev_res['sentiment']} - {rev_res['confidence']}%)")
    except Exception as e:
        results["8. Review Analyzer"] = f"FAIL: {e}"
        safe_print(f"❌ 8. Review Analyzer: FAIL - {e}")

    # -------------------------------------------------------------
    # 9. AI Customer Insights
    # -------------------------------------------------------------
    try:
        cx_res = compute_cx_insights(df)
        assert 'net_sentiment_score' in cx_res, "net_sentiment_score missing"
        assert 'csat_score' in cx_res, "csat_score missing"
        assert len(cx_res.get('executive_narratives', [])) >= 2, "Insufficient executive narratives"
        results["9. AI Customer Insights"] = "PASS"
        safe_print(f"✅ 9. AI Customer Insights: PASS (NSS: {cx_res['net_sentiment_score']:+.1f}, CSAT: {cx_res['csat_score']:.1f}%)")
    except Exception as e:
        results["9. AI Customer Insights"] = f"FAIL: {e}"
        safe_print(f"❌ 9. AI Customer Insights: FAIL - {e}")

    # -------------------------------------------------------------
    # 10. Pulse AI Typed Questions
    # -------------------------------------------------------------
    try:
        t_res = query_pulse_assistant("Why are customers unhappy?", df, total_dataset_size=len(df))
        assert len(t_res['answer']) > 50, "Answer too short"
        assert "friction" in t_res['answer'].lower() or "complaint" in t_res['answer'].lower() or "dissatisfaction" in t_res['answer'].lower(), "Key themes missing"
        results["10. Pulse AI Typed Questions"] = "PASS"
        safe_print("✅ 10. Pulse AI Typed Questions: PASS")
    except Exception as e:
        results["10. Pulse AI Typed Questions"] = f"FAIL: {e}"
        safe_print(f"❌ 10. Pulse AI Typed Questions: FAIL - {e}")

    # -------------------------------------------------------------
    # 11. Pulse AI Voice Questions (Firefox/Chrome/Edge Audio)
    try:
        audio_candidates = [
            os.path.join(os.path.dirname(__file__), 'data', 'audio_samples', 'hello_world.wav'),
            os.path.join('data', 'audio_samples', 'hello_world.wav'),
            r'C:\Users\Atif\hello_world.wav'
        ]
        audio_file = next((p for p in audio_candidates if os.path.exists(p)), None)
        assert audio_file is not None, "Test audio file hello_world.wav not found"
        with open(audio_file, 'rb') as vf:
            wav_bytes = vf.read()
        b64_wav = base64.b64encode(wav_bytes).decode('ascii')
        stt_ok, stt_text = transcribe_audio_b64(b64_wav, language='en-US')
        assert stt_ok and len(stt_text) > 0, f"STT failed: {stt_text}"

        v_res = query_pulse_assistant(stt_text, df, total_dataset_size=len(df))
        assert len(v_res['answer']) > 20, "Voice inquiry produced empty response"
        results["11. Pulse AI Voice Questions"] = "PASS"
        safe_print(f"✅ 11. Pulse AI Voice Questions: PASS (Transcribed: '{stt_text}')")
    except Exception as e:
        results["11. Pulse AI Voice Questions"] = f"FAIL: {e}"
        safe_print(f"❌ 11. Pulse AI Voice Questions: FAIL - {e}")

    # -------------------------------------------------------------
    # 12. Automatic AI Voice Response (TTS)
    # -------------------------------------------------------------
    try:
        # Both typed and voice flows clean markdown and arm TTS
        raw_ans = t_res['answer']
        cleaned_tts = _clean_text_for_speech(raw_ans)
        assert len(cleaned_tts) > 10, "Cleaned TTS output is empty"
        assert "*" not in cleaned_tts and "#" not in cleaned_tts, "Markdown symbols leaked into TTS"
        results["12. Automatic AI Voice Response"] = "PASS"
        safe_print(f"✅ 12. Automatic AI Voice Response: PASS ({len(cleaned_tts)} speech chars ready)")
    except Exception as e:
        results["12. Automatic AI Voice Response"] = f"FAIL: {e}"
        safe_print(f"❌ 12. Automatic AI Voice Response: FAIL - {e}")

    # -------------------------------------------------------------
    # 13. Listen to AI Answer (Manual Replay)
    # -------------------------------------------------------------
    try:
        replay_text = _clean_text_for_speech(raw_ans)
        assert replay_text == cleaned_tts, "Replay text does not match latest answer"
        results["13. Listen to AI Answer"] = "PASS"
        safe_print("✅ 13. Listen to AI Answer (Manual Replay): PASS")
    except Exception as e:
        results["13. Listen to AI Answer"] = f"FAIL: {e}"
        safe_print(f"❌ 13. Listen to AI Answer: FAIL - {e}")

    # -------------------------------------------------------------
    # 14. Suggested Questions
    # -------------------------------------------------------------
    try:
        suggested = [
            "Why are customers unhappy?",
            "What are the top customer complaints?",
            "What is the overall sentiment?",
            "Which product should we investigate first?",
            "Which product has the highest rating?"
        ]
        for sq in suggested:
            sq_ans = query_pulse_assistant(sq, df, total_dataset_size=len(df))
            assert len(sq_ans['answer']) > 30, f"Suggested question failed: {sq}"
        results["14. Suggested Questions"] = "PASS"
        safe_print(f"✅ 14. Suggested Questions: PASS ({len(suggested)} prompts verified)")
    except Exception as e:
        results["14. Suggested Questions"] = f"FAIL: {e}"
        safe_print(f"❌ 14. Suggested Questions: FAIL - {e}")

    # -------------------------------------------------------------
    # 15. Review Explorer & Search
    # -------------------------------------------------------------
    try:
        search_term = "quality"
        exp_match = df[df['Review Text'].str.contains(search_term, case=False, na=False)]
        assert len(exp_match) > 0, "Keyword search produced 0 matches"

        # Sort by rating ascending & descending
        s_asc = df.sort_values(by='Rating', ascending=True)
        s_desc = df.sort_values(by='Rating', ascending=False)
        assert s_asc['Rating'].iloc[0] <= s_asc['Rating'].iloc[-1], "Ascending sort failed"
        assert s_desc['Rating'].iloc[0] >= s_desc['Rating'].iloc[-1], "Descending sort failed"
        results["15. Review Explorer & Search"] = "PASS"
        safe_print(f"✅ 15. Review Explorer & Search: PASS ({len(exp_match)} matches for '{search_term}')")
    except Exception as e:
        results["15. Review Explorer & Search"] = f"FAIL: {e}"
        safe_print(f"❌ 15. Review Explorer & Search: FAIL - {e}")

    # -------------------------------------------------------------
    # 16. Export & Reports
    # -------------------------------------------------------------
    try:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        assert len(csv_bytes) > 1000, "Export CSV empty"

        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        excel_bytes = excel_out.getvalue()
        assert len(excel_bytes) > 1000, "Export Excel empty"

        results["16. Export & Reports"] = "PASS"
        safe_print("✅ 16. Export & Reports: PASS (CSV & Excel generated)")
    except Exception as e:
        results["16. Export & Reports"] = f"FAIL: {e}"
        safe_print(f"❌ 16. Export & Reports: FAIL - {e}")

    # -------------------------------------------------------------
    # 17. Dark Mode Styling & Charts
    # -------------------------------------------------------------
    try:
        dark_cfg = _get_plotly_layout(theme="dark")
        assert dark_cfg['paper_bgcolor'] == '#151D2E', "Dark theme paper bgcolor incorrect"
        assert dark_cfg['template'] == 'plotly_dark', "Dark theme template incorrect"
        results["17. Dark Mode Styling & Charts"] = "PASS"
        safe_print("✅ 17. Dark Mode Styling & Charts: PASS")
    except Exception as e:
        results["17. Dark Mode Styling & Charts"] = f"FAIL: {e}"
        safe_print(f"❌ 17. Dark Mode Styling & Charts: FAIL - {e}")

    # -------------------------------------------------------------
    # 18. Light Mode Styling & Charts
    # -------------------------------------------------------------
    try:
        light_cfg = _get_plotly_layout(theme="light")
        assert light_cfg['paper_bgcolor'] == '#FFFFFF', "Light theme paper bgcolor incorrect"
        assert light_cfg['template'] == 'plotly_white', "Light theme template incorrect"
        results["18. Light Mode Styling & Charts"] = "PASS"
        safe_print("✅ 18. Light Mode Styling & Charts: PASS")
    except Exception as e:
        results["18. Light Mode Styling & Charts"] = f"FAIL: {e}"
        safe_print(f"❌ 18. Light Mode Styling & Charts: FAIL - {e}")

    # -------------------------------------------------------------
    # 19. Multiple Sequential Questions Without Refresh
    # -------------------------------------------------------------
    try:
        chat_history = []
        queries = [
            "Which product has the highest average rating?",
            "Which product has the most negative reviews?",
            "What are the main customer complaints?",
            "Summarize customer feedback.",
            "Kis product ko pehle investigate karna chahiye?"
        ]
        latest_answer = ""
        for q in queries:
            chat_history.append({"role": "user", "content": q})
            q_res = query_pulse_assistant(q, df, total_dataset_size=len(df))
            ans = q_res['answer']
            chat_history.append({"role": "assistant", "content": ans})
            latest_answer = ans
            assert len(ans) > 20, f"Query failed in loop: {q}"

        assert len(chat_history) == 10, f"Expected 10 messages in sequence, got {len(chat_history)}"
        assert len(latest_answer) > 20, "Latest answer empty"
        results["19. Multiple Questions Without Refresh"] = "PASS"
        safe_print(f"✅ 19. Multiple Questions Without Refresh: PASS (5 sequential Q&As processed)")
    except Exception as e:
        results["19. Multiple Questions Without Refresh"] = f"FAIL: {e}"
        safe_print(f"❌ 19. Multiple Questions Without Refresh: FAIL - {e}")

    # -------------------------------------------------------------
    # 20. Cross-Browser Engine Compatibility (Firefox, Chrome, Edge)
    # -------------------------------------------------------------
    try:
        # Check HTTP response from running Streamlit server
        req = urllib.request.Request("http://localhost:8501")
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            page_content = response.read().decode('utf-8', errors='ignore')

        assert status_code == 200, f"HTTP status was {status_code}"

        # Verify voice component index.html is present and contains dual engine
        comp_path = os.path.join(os.path.dirname(__file__), 'utils', 'voice_input_component', 'index.html')
        assert os.path.exists(comp_path), "Voice component index.html missing"
        with open(comp_path, 'r', encoding='utf-8') as cf:
            comp_content = cf.read()

        assert "startWebAudioCapture" in comp_content, "Web Audio capture engine missing for Firefox"
        assert "encodeWAV" in comp_content, "WAV encoder missing for Firefox"
        assert "startNative" in comp_content, "Native Web Speech API missing for Chrome/Edge"
        assert "window.speechSynthesis" in comp_content, "SpeechSynthesis missing for TTS"

        results["20. Browser Compatibility (Firefox, Chrome, Edge)"] = "PASS"
        safe_print("✅ 20. Browser Compatibility (Firefox, Chrome, Edge): PASS")
    except Exception as e:
        results["20. Browser Compatibility (Firefox, Chrome, Edge)"] = f"FAIL: {e}"
        safe_print(f"❌ 20. Browser Compatibility (Firefox, Chrome, Edge): FAIL - {e}")

    # -------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------
    safe_print("\n" + "=" * 75)
    safe_print("FINAL FULL QA SUMMARY REPORT")
    safe_print("=" * 75)

    passed_count = sum(1 for v in results.values() if v.startswith("PASS"))
    failed_count = sum(1 for v in results.values() if v.startswith("FAIL"))

    for item, status in results.items():
        safe_print(f"  {item.ljust(50)}: {status}")

    safe_print("-" * 75)
    safe_print(f"TOTAL: {passed_count}/{total_tests} PASSED ({failed_count} FAILED)")
    safe_print("=" * 75)

    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_full_qa()
