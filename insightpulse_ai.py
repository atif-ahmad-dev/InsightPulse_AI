# insightpulse_ai.py
"""
====================================================================
InsightPulse AI - AI-Powered Customer Sentiment & Experience Analytics
Enterprise Portfolio Edition - Final UI & Multilingual AI Upgrade
====================================================================
"""

import os
import sys
from datetime import datetime, date
import pandas as pd
import streamlit as st

# Ensure terminal stdout/stderr are UTF-8 safe on Windows
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

# Import InsightPulse AI Utility Engines
from utils.data_loader import load_default_data, process_uploaded_file
from utils.nlp_engine import (
    analyze_single_review,
    analyze_dataset_sentiment,
    load_hf_pipeline
)
from utils.insights_engine import compute_cx_insights
from utils.chatbot_engine import query_pulse_assistant
from utils.ui_components import (
    inject_custom_css,
    render_header,
    render_robot_assistant_header,
    render_voice_input_widget,
    trigger_browser_text_to_speech,
    render_voice_conversation_card,
    render_kpi_cards,
    create_sentiment_donut,
    create_rating_distribution,
    create_sentiment_trend_chart,
    create_product_sentiment_chart,
    create_rating_vs_sentiment,
    create_keyword_bars,
    create_journey_stage_chart,
    render_footer,
    COLOR_POS, COLOR_NEU, COLOR_NEG
)

# Set page configuration
st.set_page_config(
    page_title="InsightPulse AI | Customer Experience Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize persistent session state variables."""
    if 'app_theme' not in st.session_state:
        st.session_state.app_theme = "dark"
        
    if 'raw_df' not in st.session_state:
        try:
            st.session_state.raw_df = load_default_data()
            st.session_state.data_source_name = "Built-in Benchmark (customer-reviews-1000.csv)"
        except Exception:
            st.session_state.raw_df = pd.DataFrame()
            st.session_state.data_source_name = "No dataset loaded"
            
    if 'scored_df' not in st.session_state or st.session_state.scored_df is None:
        if not st.session_state.raw_df.empty:
            st.session_state.scored_df = analyze_dataset_sentiment(st.session_state.raw_df)
        else:
            st.session_state.scored_df = pd.DataFrame()

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I am **Pulse 🤖**, your AI Customer Experience Assistant.\n\n"
                    "I understand natural questions in **English**, **हिन्दी (Hindi)**, and **Hinglish**! "
                    "You can speak by clicking **🎤 Speak Question** in the Voice Center or type below."
                )
            }
        ]

    if 'latest_ai_answer' not in st.session_state:
        st.session_state.latest_ai_answer = ""
    if 'autoplay_voice' not in st.session_state:
        st.session_state.autoplay_voice = False
    if 'last_voice_recognized' not in st.session_state:
        st.session_state.last_voice_recognized = ""
    if 'last_processed_voice_ts' not in st.session_state:
        st.session_state.last_processed_voice_ts = 0
    if 'play_tts_trigger' not in st.session_state:
        st.session_state.play_tts_trigger = False
    if 'tts_text_to_play' not in st.session_state:
        st.session_state.tts_text_to_play = ""
    if 'show_assistant_panel' not in st.session_state:
        st.session_state.show_assistant_panel = True
    if 'last_auto_spoken_answer' not in st.session_state:
        st.session_state.last_auto_spoken_answer = ""

def render_pulse_assistant_panel(
    active_dataset: pd.DataFrame,
    theme: str = "dark",
    custom_gemini_key: str = None,
    custom_openai_key: str = None
):
    """
    Renders the Pulse AI Assistant as a dedicated, compact assistant panel on the right side.
    Contains:
    - Robot Avatar & Online Status
    - Voice Interaction Controls (🎤 Speak Question + language selector)
    - Audio Read-Aloud button (🔊 Listen to AI Answer)
    - Real-time voice recognition feedback
    - Quick Suggested Questions
    - Scrollable Chat Conversation Container (height=360)
    - In-panel Typing Form (st.text_input + Send 🚀 button)
    - Full speech-to-text -> automatic AI answering -> automatic text-to-speech
    - Clear chat history control
    """
    # 1. Robot Header (Compact Identity)
    render_robot_assistant_header(theme=theme, compact=True)

    # 2. Voice Controls Bar & Audio Read-Aloud Button
    voice_data = render_voice_input_widget(theme=theme, key="pulse_voice_widget_input")

    listen_clicked = st.button("🔊 Listen to AI Answer", key="btn_listen_ai_answer", use_container_width=True, help="Read aloud the latest AI response")
    if listen_clicked:
        ans_to_speak = st.session_state.get('latest_ai_answer')
        if not ans_to_speak:
            ans_to_speak = "Hello! I am Pulse, your AI Customer Experience Assistant. You can speak to me by clicking Speak Question or type below."
        st.session_state.tts_text_to_play = ans_to_speak
        st.session_state.play_tts_trigger = False  # Avoid duplicate trigger in step 8
        trigger_browser_text_to_speech(ans_to_speak)
        st.toast("🔊 Playing AI audio response...", icon="🗣️")

    # Check for voice query from custom component
    voice_query = None
    if voice_data and isinstance(voice_data, dict):
        v_act = voice_data.get("action")
        v_ts = voice_data.get("timestamp", 0)
        if v_ts > st.session_state.get("last_processed_voice_ts", 0):
            if v_act == "audio_submit":
                st.session_state.last_processed_voice_ts = v_ts
                audio_b64 = voice_data.get("audio_b64", "")
                audio_lang = voice_data.get("lang", "en-US")
                if audio_b64:
                    with st.spinner("Pulse 🤖 is transcribing speech..."):
                        from utils.audio_transcriber import transcribe_audio_b64
                        stt_ok, stt_res = transcribe_audio_b64(
                            audio_b64,
                            language=audio_lang,
                            custom_gemini_key=custom_gemini_key,
                            custom_openai_key=custom_openai_key
                        )
                    if stt_ok and stt_res.strip():
                        st.session_state.last_voice_recognized = stt_res.strip()
                        voice_query = stt_res.strip()
                    else:
                        st.warning(f"🎙️ Speech Recognition: {stt_res}")
                        st.session_state.last_voice_recognized = ""
            else:
                v_txt = str(voice_data.get("text", "")).strip()
                if v_txt:
                    st.session_state.last_processed_voice_ts = v_ts
                    st.session_state.last_voice_recognized = v_txt
                    voice_query = v_txt

    if st.session_state.get('last_voice_recognized'):
        rec_q = st.session_state.last_voice_recognized
        b_bg = 'rgba(59, 130, 246, 0.15)' if theme == 'dark' else '#EFF6FF'
        b_border = '#3B82F6' if theme == 'dark' else '#93C5FD'
        b_txt = '#F8FAFC' if theme == 'dark' else '#1E3A8A'
        st.markdown(
            f'<div style="background: {b_bg}; border: 1px solid {b_border}; border-radius: 8px; padding: 6px 10px; margin: 6px 0; color: {b_txt}; font-size: 0.8rem;">'
            f'🎙️ <strong>Voice Recognized:</strong> &ldquo;{rec_q}&rdquo;'
            f'</div>',
            unsafe_allow_html=True
        )

    # 3. Quick Suggested Questions (Compact)
    with st.expander("💡 Suggested Questions", expanded=False):
        sq_col1, sq_col2 = st.columns(2)
        selected_prompt = None
        with sq_col1:
            if st.button("❓ Why unhappy?", key="sq_unhappy", use_container_width=True):
                selected_prompt = "Why are customers unhappy?"
            if st.button("🚨 Top complaints?", key="sq_complaints", use_container_width=True):
                selected_prompt = "What are the top customer complaints?"
            if st.button("📊 Overall sentiment", key="sq_sentiment", use_container_width=True):
                selected_prompt = "What is the overall sentiment?"
            if st.button("🔍 Investigate first?", key="sq_investigate", use_container_width=True):
                selected_prompt = "Which product should we investigate first?"
        with sq_col2:
            if st.button("👎 Most negative?", key="sq_neg", use_container_width=True):
                selected_prompt = "Which product has the most negative reviews?"
            if st.button("⭐ Highest rated?", key="sq_best", use_container_width=True):
                selected_prompt = "Which product has the highest rating?"
            if st.button("📈 Feedback summary", key="sq_summary", use_container_width=True):
                selected_prompt = "Summarize all customer feedback."
            if st.button("🇮🇳 Scorpio N issues?", key="sq_hinglish", use_container_width=True):
                selected_prompt = "Customers Scorpio N se unhappy kyun hain?"

    # 4. Scrollable Chat History Container
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; margin: 6px 0 3px 0; color: var(--text-color);'>💬 Assistant Chat:</div>", unsafe_allow_html=True)
    chat_box = st.container(height=360)
    with chat_box:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # 5. Permanent In-Panel Typing Form with Send Button
    with st.form(key="pulse_chat_form", clear_on_submit=True):
        t_col, s_col = st.columns([3.4, 1.3])
        with t_col:
            typed_query = st.text_input(
                "Ask Pulse AI",
                placeholder="Ask Pulse AI anything about your customer reviews...",
                label_visibility="collapsed",
                key="pulse_typed_input_field"
            )
        with s_col:
            send_clicked = st.form_submit_button("Send 🚀", use_container_width=True)

    # 6. Clear History Button
    if len(st.session_state.chat_messages) > 1:
        if st.button("🗑️ Clear History", key="btn_clear_chat", use_container_width=True):
            st.session_state.chat_messages = [st.session_state.chat_messages[0]]
            st.session_state.latest_ai_answer = ""
            st.session_state.play_tts_trigger = False
            st.session_state.tts_text_to_play = ""
            st.session_state.last_voice_recognized = ""
            st.session_state.last_auto_spoken_answer = ""
            st.rerun()

    # 7. Determine prompt to process
    prompt_to_process = None
    is_voice_submit = False
    if voice_query:
        prompt_to_process = voice_query
        is_voice_submit = True
    elif selected_prompt:
        prompt_to_process = selected_prompt
    elif send_clicked and typed_query.strip():
        prompt_to_process = typed_query.strip()

    if prompt_to_process:
        user_msg = prompt_to_process.strip()
        msg_content = f"🎤 **Spoken Question**: {user_msg}" if is_voice_submit else user_msg
        st.session_state.chat_messages.append({"role": "user", "content": msg_content})

        total_ds_len = len(st.session_state.scored_df) if ('scored_df' in st.session_state and isinstance(st.session_state.scored_df, pd.DataFrame)) else None

        with st.spinner("Pulse 🤖 is analyzing reviews..."):
            response_dict = query_pulse_assistant(
                user_msg,
                active_dataset,
                custom_gemini_key=custom_gemini_key,
                custom_openai_key=custom_openai_key,
                total_dataset_size=total_ds_len
            )
            answer = response_dict['answer']
            mode_badge = f"\n\n*⚡ {response_dict['mode']}*"
            full_reply = answer + mode_badge

        st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})
        st.session_state.latest_ai_answer = answer

        # Automatically read the AI answer aloud for BOTH voice and typed inquiries!
        st.session_state.play_tts_trigger = True
        st.session_state.tts_text_to_play = answer
        st.session_state.last_auto_spoken_answer = answer

        st.rerun()

    # 8. Browser Text-to-Speech Execution (if triggered)
    if st.session_state.get('play_tts_trigger') and st.session_state.get('tts_text_to_play'):
        trigger_browser_text_to_speech(st.session_state['tts_text_to_play'])
        st.session_state.play_tts_trigger = False

def render_dashboard_tab(filtered_df: pd.DataFrame, current_theme: str, show_assistant: bool):
    """Render Tab 1: Dashboard Analytics & Charts."""
    if filtered_df.empty:
        st.info("No reviews match the current filter selection. Please adjust the sidebar filters.")
        return

    total_revs = len(filtered_df)
    avg_rtg = filtered_df['Rating'].mean()
    pos_p = (len(filtered_df[filtered_df['sentiment'] == 'Positive']) / total_revs * 100) if total_revs else 0
    neu_p = (len(filtered_df[filtered_df['sentiment'] == 'Neutral']) / total_revs * 100) if total_revs else 0
    neg_p = (len(filtered_df[filtered_df['sentiment'] == 'Negative']) / total_revs * 100) if total_revs else 0
    total_prods = filtered_df['Product Name'].nunique()

    # KPI Cards (responsive 3 cols if side-by-side, 6 cols if full width)
    render_kpi_cards(
        total_revs, avg_rtg, pos_p, neu_p, neg_p, total_prods,
        theme=current_theme,
        cols=3 if show_assistant else 6
    )

    # Row 1 of Charts
    c1, c2 = st.columns(2)
    with c1:
        donut_fig = create_sentiment_donut(filtered_df, theme=current_theme)
        st.plotly_chart(donut_fig, use_container_width=True)
    with c2:
        rating_fig = create_rating_distribution(filtered_df, theme=current_theme)
        st.plotly_chart(rating_fig, use_container_width=True)

    # Row 2 of Charts
    c3, c4 = st.columns(2)
    with c3:
        trend_fig = create_sentiment_trend_chart(filtered_df, theme=current_theme)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Timeline trend requires valid dates in the dataset.")
    with c4:
        stage_fig = create_journey_stage_chart(filtered_df, theme=current_theme)
        if stage_fig:
            st.plotly_chart(stage_fig, use_container_width=True)
        else:
            st.info("Journey stage touchpoints not present in this dataset.")

    # Row 3 of Charts
    c5, c6 = st.columns(2)
    with c5:
        prod_fig = create_product_sentiment_chart(filtered_df, top_n=10, theme=current_theme)
        st.plotly_chart(prod_fig, use_container_width=True)
    with c6:
        align_fig = create_rating_vs_sentiment(filtered_df, theme=current_theme)
        st.plotly_chart(align_fig, use_container_width=True)

    # Row 4: Keyword Signals
    cx_data = compute_cx_insights(filtered_df)
    if cx_data.get('pos_keywords') or cx_data.get('neg_keywords'):
        st.markdown("### 🔑 Keyword & Aspect Signals")
        k1, k2 = st.columns(2)
        fig_pos_k, fig_neg_k = create_keyword_bars(
            cx_data.get('pos_keywords', []),
            cx_data.get('neg_keywords', []),
            theme=current_theme
        )
        with k1:
            if fig_pos_k:
                st.plotly_chart(fig_pos_k, use_container_width=True)
            else:
                st.info("No positive keywords found.")
        with k2:
            if fig_neg_k:
                st.plotly_chart(fig_neg_k, use_container_width=True)
            else:
                st.info("No negative keywords found.")

def render_analyzer_tab(current_theme: str):
    """Render Tab 2: Single Review Analyzer."""
    st.markdown("### 🔍 AI Review Analyzer")
    st.write(
        "Analyze individual customer reviews in real-time. InsightPulse extracts sentiment polarity, "
        "calibrated confidence, friction points, positive highlights, and emotional undertone."
    )

    st.markdown("##### 💡 Try an Instant Preset Review:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    sample_review_text = "The product quality is good but delivery was very late."

    with col_p1:
        if st.button("Mixed: Quality vs Delivery", use_container_width=True):
            st.session_state['analyzer_input'] = "The product quality is good but delivery was very late."
    with col_p2:
        if st.button("Praise: Amazing Support", use_container_width=True):
            st.session_state['analyzer_input'] = "Customer support was amazing! Resolved my query in minutes."
    with col_p3:
        if st.button("Negative: Poor Quality", use_container_width=True):
            st.session_state['analyzer_input'] = "Complete waste of money. Product quality is extremely poor."
    with col_p4:
        if st.button("Neutral: Basic Delivery", use_container_width=True):
            st.session_state['analyzer_input'] = "Standard delivery service. Product arrived intact."

    default_text = st.session_state.get('analyzer_input', sample_review_text)
    user_review = st.text_area(
        "Customer Review Text",
        value=default_text,
        height=110,
        placeholder="Type or paste any customer review here..."
    )

    if st.button("⚡ Run AI Analysis", type="primary"):
        if user_review.strip():
            with st.spinner("Analyzing review with NLP..."):
                result = analyze_single_review(user_review)

            st.markdown("---")
            st.markdown("#### 📋 Analysis Intelligence Report")

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Sentiment", result['sentiment'])
            with m2:
                st.metric("Confidence Score", f"{result['confidence']}%")
            with m3:
                st.metric("Polarity Score", f"{result['polarity']:.2f}")
            with m4:
                st.metric("Emotion Tone", result['emotion'])

            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

            # Breakdown
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("**🎯 Detected Operational Touchpoint / Issue:**")
                st.info(f"**{result['detected_customer_issue']}**")

                st.markdown("**🟢 Positive Aspects Identified:**")
                if result['positive_aspects']:
                    for p in result['positive_aspects']:
                        st.markdown(f"- ✅ *\"{p}\"*")
                else:
                    st.caption("No distinctly positive aspect detected.")

            with r_col2:
                st.markdown("**🔴 Negative Aspects / Friction:**")
                if result['negative_aspects']:
                    for n in result['negative_aspects']:
                        st.markdown(f"- ⚠️ *\"{n}\"*")
                else:
                    st.caption("No friction or negative issue detected.")

                st.markdown("**🧠 Active NLP Model:**")
                st.caption(f"`{result['model_used']}`")

            st.markdown("**💡 AI-Generated Insight & Action Item:**")
            st.markdown(f'<div class="insight-card">{result["insight"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter review text to analyze.")

def render_insights_tab(filtered_df: pd.DataFrame):
    """Render Tab 3: Customer Experience Insights."""
    st.markdown("### 🧠 AI Customer Experience Insights")
    st.write("Aggregated business intelligence derived directly from your customer review corpus.")

    if filtered_df.empty:
        st.info("No data available for customer experience insights.")
        return

    cx = compute_cx_insights(filtered_df)

    # High-level indicators
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("Net Sentiment Score", f"{'+' if cx['net_sentiment_score'] >= 0 else ''}{cx['net_sentiment_score']:.1f}")
    with h2:
        st.metric("Estimated CSAT (% 4-5⭐)", f"{cx['csat_score']:.1f}%")
    with h3:
        st.metric("Customer Friction Rate", f"{cx['neg_pct']:.1f}%")
    with h4:
        st.metric("Customer Delight Rate", f"{cx['pos_pct']:.1f}%")

    st.markdown("---")

    # Strategic Narratives
    st.markdown("#### 📌 Strategic Executive Takeaways")
    for narr in cx.get('executive_narratives', []):
        st.markdown(f'<div class="insight-card">{narr}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Star Performers vs Problem Products
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("#### 🌟 Top Performing Star Products")
        star_df = cx.get('star_products', pd.DataFrame())
        if not star_df.empty:
            st.dataframe(
                star_df[['Product Name', 'Positive %', 'Avg Rating', 'Reviews']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No product performance data available.")

    with p_col2:
        st.markdown("#### 🚨 High-Risk Problem Products")
        prob_df = cx.get('problem_products', pd.DataFrame())
        if not prob_df.empty:
            st.dataframe(
                prob_df[['Product Name', 'Negative %', 'Avg Rating', 'Reviews']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No problem products identified.")

    # Journey Stage Touchpoints
    st.markdown("---")
    st.markdown("#### 🔄 Customer Journey Stage Performance")
    stage_table = cx.get('stage_df', pd.DataFrame())
    if not stage_table.empty:
        st.dataframe(
            stage_table,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Journey stage touchpoint data not present in dataset.")

def render_explorer_tab(filtered_df: pd.DataFrame):
    """Render Tab 5: Review Explorer & Search."""
    st.markdown("### 📋 Customer Review Explorer")

    if filtered_df.empty:
        st.info("No reviews to display.")
        return

    sr_col1, sr_col2 = st.columns([3, 1])
    with sr_col1:
        search_query = st.text_input("🔍 Search Reviews by Keyword", placeholder="Search for keywords like 'delivery', 'broken', 'amazing'...")
    with sr_col2:
        sort_option = st.selectbox("Sort Reviews By", ["Rating: High to Low", "Rating: Low to High", "Most Recent", "Sentiment Score"])

    exp_df = filtered_df.copy()
    if search_query:
        exp_df = exp_df[exp_df['Review Text'].str.contains(search_query, case=False, na=False)]

    if sort_option == "Rating: High to Low":
        exp_df = exp_df.sort_values(by='Rating', ascending=False)
    elif sort_option == "Rating: Low to High":
        exp_df = exp_df.sort_values(by='Rating', ascending=True)
    elif sort_option == "Sentiment Score":
        exp_df = exp_df.sort_values(by='polarity', ascending=False)
    else:
        if 'Date' in exp_df.columns:
            exp_df = exp_df.sort_values(by='Date', ascending=False)

    st.markdown(f"Displaying **{len(exp_df):,} reviews**:")

    display_columns = ['Review ID', 'Product Name', 'Rating', 'sentiment', 'Review Text']
    if 'Journey Stage' in exp_df.columns:
        display_columns.insert(4, 'Journey Stage')
    if 'Date' in exp_df.columns:
        display_columns.insert(3, 'Date')

    valid_display_cols = [c for c in display_columns if c in exp_df.columns]

    st.dataframe(
        exp_df[valid_display_cols].head(200),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rating": st.column_config.NumberColumn("Rating ⭐", format="%.1f"),
            "sentiment": st.column_config.TextColumn("Sentiment"),
            "Review Text": st.column_config.TextColumn("Customer Review", width="large")
        }
    )

def render_export_tab(filtered_df: pd.DataFrame, current_theme: str):
    """Render Tab 6: Export & Reports."""
    st.markdown("### 📊 Export Intelligence & Reports")
    st.write("Download filtered datasets, enriched sentiment calculations, and an executive briefing.")

    if filtered_df.empty:
        st.warning("No data available to export.")
        return

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    with exp_col1:
        st.markdown("##### 📄 Filtered Reviews CSV")
        csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered CSV",
            data=csv_filtered,
            file_name=f"insightpulse_filtered_{timestamp_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"Contains {len(filtered_df):,} currently filtered records.")

    with exp_col2:
        st.markdown("##### 📊 Full Enriched Dataset")
        full_enriched_csv = st.session_state.scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download All Enriched Data",
            data=full_enriched_csv,
            file_name=f"insightpulse_full_enriched_{timestamp_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"Contains all {len(st.session_state.scored_df):,} records with NLP scores.")

    with exp_col3:
        st.markdown("##### 📝 Executive Briefing")
        cx_brief = compute_cx_insights(filtered_df)
        narratives_text = "\n".join([f"- {n.replace('**', '').replace('🟢 ', '').replace('🔴 ', '').replace('🟡 ', '')}" for n in cx_brief.get('executive_narratives', [])])

        report_content = f"""# InsightPulse AI - Executive Customer Experience Briefing
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset Source: {st.session_state.data_source_name}

## 1. Executive KPIs
- Total Reviews Analyzed: {len(filtered_df):,}
- Average Rating: {filtered_df['Rating'].mean():.2f} / 5.0
- Positive Sentiment: {len(filtered_df[filtered_df['sentiment'] == 'Positive']):,} ({(len(filtered_df[filtered_df['sentiment'] == 'Positive'])/len(filtered_df)*100):.1f}%)
- Neutral Sentiment: {len(filtered_df[filtered_df['sentiment'] == 'Neutral']):,} ({(len(filtered_df[filtered_df['sentiment'] == 'Neutral'])/len(filtered_df)*100):.1f}%)
- Negative Sentiment: {len(filtered_df[filtered_df['sentiment'] == 'Negative']):,} ({(len(filtered_df[filtered_df['sentiment'] == 'Negative'])/len(filtered_df)*100):.1f}%)
- Net Sentiment Score: {cx_brief.get('net_sentiment_score', 0.0):+.1f}

## 2. Key Executive Takeaways
{narratives_text}

## 3. Recommended Management Priorities
1. Focus packaging and handling quality on shipping partners to mitigate damage complaints.
2. Maintain training on customer support teams to preserve high positive feedback in response efficiency.
3. Review product specifications and quality checks for problem products identified in the report.

====================================================================
InsightPulse AI Portfolio Edition
"""
        st.download_button(
            label="⬇️ Download Executive Brief (.txt)",
            data=report_content.encode('utf-8'),
            file_name=f"insightpulse_briefing_{timestamp_str}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.caption("Executive markdown summary of findings.")

    st.markdown("---")
    st.markdown("#### 📈 Summary Statistics Table")
    summary_stats = {
        "Metric": [
            "Total Reviews",
            "Average Rating",
            "Average Polarity",
            "Average Subjectivity",
            "Positive Reviews %",
            "Neutral Reviews %",
            "Negative Reviews %",
            "Unique Products Evaluated"
        ],
        "Value": [
            f"{len(filtered_df):,}",
            f"{filtered_df['Rating'].mean():.2f} ⭐",
            f"{filtered_df['polarity'].mean():.2f}",
            f"{filtered_df['subjectivity'].mean():.2f}",
            f"{(len(filtered_df[filtered_df['sentiment'] == 'Positive']) / len(filtered_df) * 100):.1f}%",
            f"{(len(filtered_df[filtered_df['sentiment'] == 'Neutral']) / len(filtered_df) * 100):.1f}%",
            f"{(len(filtered_df[filtered_df['sentiment'] == 'Negative']) / len(filtered_df) * 100):.1f}%",
            f"{filtered_df['Product Name'].nunique()}"
        ]
    }
    st.table(pd.DataFrame(summary_stats))

def main():
    initialize_session_state()

    # -------------------------------------------------------------
    # Sidebar: Theme Selector, Data Source, Configuration & Filters
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        theme_selection = st.radio(
            "Select Theme",
            ["🌙 Dark Mode", "☀️ Light Mode"],
            index=0 if st.session_state.app_theme == "dark" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.app_theme = "dark" if "Dark" in theme_selection else "light"
        current_theme = st.session_state.app_theme
        
        # Inject theme-specific CSS
        inject_custom_css(theme=current_theme)
        
        st.markdown("### 🖥️ Workspace Layout")
        show_assistant_panel = st.checkbox(
            "🤖 Pulse AI Assistant Panel",
            value=st.session_state.get('show_assistant_panel', True),
            help="Show the compact Pulse AI Assistant panel on the right side of the dashboard."
        )
        st.session_state.show_assistant_panel = show_assistant_panel
        st.markdown("---")

        st.markdown("### ⚙️ Data Source")
        data_source_option = st.radio(
            "Select Input Source",
            ["Sample Dataset", "Upload Reviews (CSV/Excel)"],
            index=0 if "Built-in" in st.session_state.data_source_name else 1,
            label_visibility="collapsed"
        )
        
        if data_source_option == "Upload Reviews (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "Upload Customer Reviews",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a CSV or Excel file containing customer reviews. Columns like 'Review Text', 'Rating', and 'Product Name' are auto-detected."
            )
            if uploaded_file is not None:
                new_df, error_msg = process_uploaded_file(uploaded_file)
                if error_msg:
                    st.error(f"❌ {error_msg}")
                else:
                    if st.session_state.data_source_name != uploaded_file.name:
                        with st.spinner("Analyzing uploaded customer reviews with NLP..."):
                            st.session_state.raw_df = new_df
                            st.session_state.scored_df = analyze_dataset_sentiment(new_df)
                            st.session_state.data_source_name = uploaded_file.name
                        st.success(f"Loaded {len(new_df):,} reviews from `{uploaded_file.name}`")
                        st.rerun()
        else:
            if "Built-in" not in st.session_state.data_source_name:
                with st.spinner("Resetting to sample dataset..."):
                    st.session_state.raw_df = load_default_data()
                    st.session_state.scored_df = analyze_dataset_sentiment(st.session_state.raw_df)
                    st.session_state.data_source_name = "Built-in Benchmark (customer-reviews-1000.csv)"
                    st.rerun()

        st.caption(f"Active Source: **{st.session_state.data_source_name}**")
        st.markdown("---")
        
        # Filters Section
        st.markdown("### 🔍 Filter Analytics")
        active_df = st.session_state.scored_df
        
        if active_df.empty:
            st.warning("No data available to filter.")
            filtered_df = pd.DataFrame()
        else:
            # 1. Product Filter
            all_products = sorted(active_df['Product Name'].dropna().unique().tolist())
            select_all_products = st.checkbox("All Products", value=True)
            if select_all_products:
                selected_products = all_products
            else:
                selected_products = st.multiselect(
                    "Products",
                    options=all_products,
                    default=all_products[:5] if len(all_products) >= 5 else all_products
                )
                
            # 2. Rating Filter
            ratings_available = sorted([int(r) for r in active_df['Rating'].dropna().unique()])
            selected_ratings = st.multiselect(
                "Ratings (Stars)",
                options=ratings_available,
                default=ratings_available
            )
            
            # 3. Sentiment Filter
            sentiments_available = ['Positive', 'Neutral', 'Negative']
            selected_sentiments = st.multiselect(
                "Sentiment",
                options=sentiments_available,
                default=sentiments_available
            )
            
            # 4. Date Range Filter
            valid_dates = pd.to_datetime(active_df['Date'], errors='coerce').dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
                
            # Apply Filters
            filtered_mask = (
                (active_df['Product Name'].isin(selected_products)) &
                (active_df['Rating'].astype(int).isin(selected_ratings)) &
                (active_df['sentiment'].isin(selected_sentiments))
            )
            
            if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_d, end_d = date_range
                date_series = pd.to_datetime(active_df['Date'], errors='coerce').dt.date
                filtered_mask = filtered_mask & (date_series >= start_d) & (date_series <= end_d)
                
            filtered_df = active_df[filtered_mask].copy()

        st.markdown("---")
        # AI Engine & API Key Configurations
        with st.expander("🤖 AI Engine Settings", expanded=False):
            hf_available = load_hf_pipeline() is not None
            if hf_available:
                st.success("✅ Hugging Face Transformer Active")
            else:
                st.info("⚡ Hybrid Multilingual Engine Active (NLTK + Aspect Mining)")
                
            user_gemini_key = st.text_input(
                "Gemini API Key (Optional)",
                type="password",
                placeholder="AIzaSy...",
                help="Optional: Enter a Google Gemini API key to enhance Pulse AI Chatbot responses."
            )
            user_openai_key = st.text_input(
                "OpenAI API Key (Optional)",
                type="password",
                placeholder="sk-...",
                help="Optional: Enter an OpenAI API key to enhance Pulse AI Chatbot responses."
            )

    # -------------------------------------------------------------
    # Check for Incoming Voice Query from Web Speech API
    # -------------------------------------------------------------
    if "voice_prompt" in st.query_params:
        raw_voice = st.query_params.get("voice_prompt")
        auto_read = str(st.query_params.get("auto_read", "1")).lower() in ["1", "true", "yes"]
        # Clear query params immediately
        del st.query_params["voice_prompt"]
        if "auto_read" in st.query_params:
            del st.query_params["auto_read"]

        if raw_voice and str(raw_voice).strip():
            spoken_query = str(raw_voice).strip()
            st.session_state.last_voice_recognized = spoken_query
            st.session_state.chat_messages.append({"role": "user", "content": f"🎤 **Spoken Question**: {spoken_query}"})

            # Query Assistant with active dataset (respecting filters)
            active_data = filtered_df if ('filtered_df' in locals() and not filtered_df.empty) else st.session_state.scored_df
            total_ds_len = len(st.session_state.scored_df) if ('scored_df' in st.session_state and isinstance(st.session_state.scored_df, pd.DataFrame)) else None
            response_dict = query_pulse_assistant(
                spoken_query,
                active_data,
                custom_gemini_key=user_gemini_key if 'user_gemini_key' in locals() else None,
                custom_openai_key=user_openai_key if 'user_openai_key' in locals() else None,
                total_dataset_size=total_ds_len
            )
            ans = response_dict['answer']
            full_reply = f"{ans}\n\n*⚡ {response_dict['mode']}*"
            st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})
            st.session_state.latest_ai_answer = ans
            if auto_read:
                st.session_state.autoplay_voice = True
                if ans != st.session_state.get('last_auto_spoken_answer'):
                    st.session_state.play_tts_trigger = True
                    st.session_state.tts_text_to_play = ans
                    st.session_state.last_auto_spoken_answer = ans
            st.rerun()

    # Inject CSS for current theme
    current_theme = st.session_state.app_theme
    inject_custom_css(theme=current_theme)

    # Render Opening Screen Hero Header
    render_header(theme=current_theme)

    # -------------------------------------------------------------
    # Main Application Layout: Workspace on Left, Assistant on Right
    # -------------------------------------------------------------
    show_assistant = st.session_state.get('show_assistant_panel', True)

    if show_assistant:
        col_main, col_pulse = st.columns([1.85, 1.15], gap="large")
    else:
        col_main = st.container()
        col_pulse = None

    with col_main:
        if show_assistant:
            tab_dash, tab_analyzer, tab_insights, tab_explorer, tab_export = st.tabs([
                "🏠 Dashboard",
                "🔍 Review Analyzer",
                "🧠 AI Customer Insights",
                "📋 Review Explorer",
                "📊 Export & Reports"
            ])
            with tab_dash:
                render_dashboard_tab(filtered_df, current_theme, show_assistant=True)
            with tab_analyzer:
                render_analyzer_tab(current_theme)
            with tab_insights:
                render_insights_tab(filtered_df)
            with tab_explorer:
                render_explorer_tab(filtered_df)
            with tab_export:
                render_export_tab(filtered_df, current_theme)
        else:
            tab_dash, tab_analyzer, tab_insights, tab_pulse_full, tab_explorer, tab_export = st.tabs([
                "🏠 Dashboard",
                "🔍 Review Analyzer",
                "🧠 AI Customer Insights",
                "🤖 Pulse AI Assistant",
                "📋 Review Explorer",
                "📊 Export & Reports"
            ])
            with tab_dash:
                render_dashboard_tab(filtered_df, current_theme, show_assistant=False)
            with tab_analyzer:
                render_analyzer_tab(current_theme)
            with tab_insights:
                render_insights_tab(filtered_df)
            with tab_pulse_full:
                render_pulse_assistant_panel(
                    active_dataset=filtered_df if not filtered_df.empty else st.session_state.scored_df,
                    theme=current_theme,
                    custom_gemini_key=user_gemini_key if 'user_gemini_key' in locals() else None,
                    custom_openai_key=user_openai_key if 'user_openai_key' in locals() else None
                )
            with tab_explorer:
                render_explorer_tab(filtered_df)
            with tab_export:
                render_export_tab(filtered_df, current_theme)

    # -------------------------------------------------------------
    # Right-Side Dedicated Pulse AI Assistant Panel
    # -------------------------------------------------------------
    if col_pulse is not None:
        with col_pulse:
            render_pulse_assistant_panel(
                active_dataset=filtered_df if not filtered_df.empty else st.session_state.scored_df,
                theme=current_theme,
                custom_gemini_key=user_gemini_key if 'user_gemini_key' in locals() else None,
                custom_openai_key=user_openai_key if 'user_openai_key' in locals() else None
            )

    # Render Professional Footer
    render_footer(theme=current_theme)

if __name__ == "__main__":
    main()