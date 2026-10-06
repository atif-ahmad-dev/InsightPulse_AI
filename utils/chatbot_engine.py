# utils/chatbot_engine.py
"""
InsightPulse AI - Pulse AI Assistant Engine
Provides hybrid intelligence: deterministic dataset analytics engine for instant,
zero-cost queries across English, Hindi, and Hinglish, plus optional LLM grounding via Google Gemini / OpenAI.
"""

import os
import sys
import re
from typing import Dict, Any, Optional, List, Tuple
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

def get_api_key_from_env() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve API keys safely from streamlit secrets or environment variables."""
    gemini_key = None
    openai_key = None
    
    # Try reading from Streamlit secrets safely
    try:
        if hasattr(st, "secrets"):
            gemini_key = st.secrets.get("GEMINI_API_KEY", None)
            openai_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass

    # Check environment variables
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not openai_key:
        openai_key = os.environ.get("OPENAI_API_KEY")

    return gemini_key, openai_key

def detect_language(query: str) -> str:
    """
    Detect language of the query:
    - 'hi' (Hindi in Devanagari script)
    - 'hinglish' (Hindi in Roman script)
    - 'en' (English default)
    """
    if not query:
        return 'en'
        
    # Check for Devanagari Unicode range
    if re.search(r'[\u0900-\u097F]', query):
        return 'hi'
        
    # Check for Romanized Hindi keywords (Hinglish)
    hinglish_markers = [
        r'\bkyun\b', r'\bkyu\b', r'\bkaun\b', r'\bkaunsa\b', r'\bkaunsi\b', r'\bkya\b',
        r'\bhai\b', r'\bhain\b', r'\bsabse\b', r'\bzyada\b', r'\bjyada\b', r'\bkharab\b',
        r'\baccha\b', r'\bacha\b', r'\bbura\b', r'\bbatao\b', r'\bbataiye\b', r'\bkisko\b',
        r'\bkaro\b', r'\brahi\b', r'\brahe\b', r'\bwajah\b', r'\bdikkat\b', r'\bpareshani\b',
        r'\bse\b', r'\bko\b', r'\bmein\b', r'\bmujhe\b', r'\bhumein\b', r'\bkitna\b', r'\bkitne\b',
        r'\bkitni\b', r'\bhona\b', r'\bchahiye\b', r'\bkarein\b', r'\bloag\b', r'\blog\b',
        r'\bnaraz\b', r'\bpasand\b', r'\bmuqabla\b', r'\bbare\b', r'\bpehle\b', r'\bkarni\b',
        r'\bkisae\b', r'\bchiz\b', r'\bcheez\b', r'\bkaren\b', r'\bkaisi\b', r'\bkaisa\b',
        r'\bdikhaye\b', r'\bdikhao\b', r'\bkripya\b'
    ]
    query_lower = query.lower()
    for marker in hinglish_markers:
        if re.search(marker, query_lower):
            return 'hinglish'
            
    return 'en'

def build_dataset_context_summary(df: pd.DataFrame) -> str:
    """Generate a compact contextual summary of the dataset for LLM grounding."""
    if df.empty:
        return "The dataset is empty."
        
    total = len(df)
    avg_rating = df['Rating'].mean() if 'Rating' in df.columns else 0.0
    pos_count = len(df[df['sentiment'] == 'Positive'])
    neg_count = len(df[df['sentiment'] == 'Negative'])
    neu_count = len(df[df['sentiment'] == 'Neutral'])
    
    # Top problem products
    problem_prods = ""
    if 'Product Name' in df.columns:
        prod_neg = df[df['sentiment'] == 'Negative']['Product Name'].value_counts().head(3)
        problem_prods = ", ".join([f"{p} ({c} negative)" for p, c in prod_neg.items()])
        
    # Top star products
    star_prods = ""
    if 'Product Name' in df.columns:
        prod_pos = df[df['sentiment'] == 'Positive']['Product Name'].value_counts().head(3)
        star_prods = ", ".join([f"{p} ({c} positive)" for p, c in prod_pos.items()])

    # Sample complaints
    sample_negs = df[df['sentiment'] == 'Negative']['Review Text'].dropna().head(3).tolist()
    sample_negs_str = "\n".join([f"- {s}" for s in sample_negs])

    return f"""
Dataset Analytics Summary:
- Total Reviews: {total}
- Average Rating: {avg_rating:.2f} / 5.0
- Positive: {pos_count} ({pos_count/total*100:.1f}%)
- Negative: {neg_count} ({neg_count/total*100:.1f}%)
- Neutral: {neu_count} ({neu_count/total*100:.1f}%)
- Products with Most Negative Reviews: {problem_prods if problem_prods else 'None'}
- Top Performing Products: {star_prods if star_prods else 'None'}
- Recent Sample Negative Complaints:
{sample_negs_str}
"""

def query_pulse_assistant(
    user_query: str,
    df: pd.DataFrame,
    custom_gemini_key: Optional[str] = None,
    custom_openai_key: Optional[str] = None,
    total_dataset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a user question regarding the dataset with automatic language detection (English, Hindi, Hinglish).
    Prioritizes Gemini/OpenAI if configured, with graceful fallback to the Multilingual Local Query Engine.
    Includes top-level exception handling ensuring zero application crashes.
    """
    if not user_query or not str(user_query).strip():
        return {
            "answer": "Please ask a question about your customer reviews. / कृपया अपने ग्राहक समीक्षाओं के बारे में प्रश्न पूछें।",
            "mode": "Local Engine",
            "source": "Local",
            "language": "en"
        }
        
    lang = detect_language(user_query)
    query_clean = str(user_query).strip()

    try:
        # Check for external LLM keys
        env_gemini, env_openai = get_api_key_from_env()
        gemini_key = custom_gemini_key or env_gemini
        openai_key = custom_openai_key or env_openai
        
        lang_name_map = {
            'hi': 'Hindi (Devanagari script)',
            'hinglish': 'Hinglish (Conversational Hindi in Roman English script)',
            'en': 'English'
        }
        lang_instruction = lang_name_map.get(lang, 'English')
        
        # If Gemini API key is available, try Gemini
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                context = build_dataset_context_summary(df)
                prompt = f"""
You are Pulse 🤖, an expert AI Customer Experience Analyst.
The user asked their question in: {lang_instruction}.
You MUST formulate your entire answer in that EXACT SAME LANGUAGE ({lang_instruction}).
Answer the user's question accurately and concisely based strictly on the provided customer reviews dataset summary below.
Do not fabricate information. Format your answer with clean markdown bullet points, percentages, product names, and concrete recommendations.

{context}

User Question: {user_query}
"""
                response = model.generate_content(prompt)
                if response and response.text:
                    return {
                        "answer": response.text.strip(),
                        "mode": f"Google Gemini 1.5 Flash ({lang.upper()})",
                        "source": "GenAI",
                        "language": lang
                    }
            except Exception:
                pass

        # If OpenAI API key is available, try OpenAI
        if openai_key:
            try:
                import openai
                openai.api_key = openai_key
                context = build_dataset_context_summary(df)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are Pulse 🤖, an expert AI Customer Experience Analyst. Answer questions strictly based on the provided dataset context. Respond in the user's language: {lang_instruction}."
                        },
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
                    ],
                    max_tokens=350,
                    temperature=0.2
                )
                ans = response.choices[0].message.content.strip()
                return {
                    "answer": ans,
                    "mode": f"OpenAI GPT ({lang.upper()})",
                    "source": "GenAI",
                    "language": lang
                }
            except Exception:
                pass

        # Execute Multilingual Local Analytics & Query Engine
        local_answer = execute_multilingual_query_engine(
            query_clean,
            df,
            lang=lang,
            total_dataset_size=total_dataset_size
        )
        return {
            "answer": local_answer,
            "mode": f"Pulse Multilingual Engine ({lang.capitalize()})",
            "source": "Local",
            "language": lang
        }

    except Exception as e:
        # Log internally without leaking tracebacks to UI
        print(f"[Pulse AI Exception] Error processing query '{user_query}': {e}", file=sys.stderr)
        
        if lang == 'hi':
            friendly_err = "क्षमा करें, मैं अभी इस प्रश्न को संसाधित नहीं कर सका। कृपया अपना प्रश्न पुनः पूछें या बदलकर देखें।"
        elif lang == 'hinglish':
            friendly_err = "Sorry, main abhi is question ko process nahi kar saka. Please thodi der baad try karein ya question rephrase karein."
        else:
            friendly_err = "I couldn't process that question right now. Please try again or rephrase your question."

        return {
            "answer": friendly_err,
            "mode": "Pulse AI Fallback",
            "source": "Local",
            "language": lang
        }

def execute_multilingual_query_engine(
    query: str,
    df: pd.DataFrame,
    lang: str = 'en',
    total_dataset_size: Optional[int] = None
) -> str:
    """
    Deterministic, high-accuracy analytics engine answering customer review questions
    fluently in English, Hindi (Devanagari), or Hinglish based on real dataset statistics.
    Outputs ChatGPT-style structured answers with bullet points, numbers, percentages,
    root cause friction drivers, and concrete management recommendations.
    """
    if df.empty:
        if lang == 'hi':
            return "⚠️ वर्तमान में डेटासेट खाली है। कृपया ग्राहक समीक्षा डेटा लोड या अपलोड करें।"
        elif lang == 'hinglish':
            return "⚠️ Dataset abhi empty hai. Please customer reviews load ya upload karein."
        else:
            return "⚠️ The dataset is currently empty. Please load or upload customer reviews."

    total = len(df)
    avg_rating = df['Rating'].mean() if 'Rating' in df.columns else 0.0
    pos_df = df[df['sentiment'] == 'Positive']
    neg_df = df[df['sentiment'] == 'Negative']
    neu_df = df[df['sentiment'] == 'Neutral']
    
    pos_count = len(pos_df)
    neg_count = len(neg_df)
    neu_count = len(neu_df)
    neg_avg_rating = neg_df['Rating'].mean() if not neg_df.empty else 0.0
    pos_avg_rating = pos_df['Rating'].mean() if not pos_df.empty else 0.0
    
    query_lower = query.lower()

    # Filter status banner (if filtered subset)
    filter_prefix = ""
    if total_dataset_size and total < total_dataset_size:
        if lang == 'hi':
            filter_prefix = f"*फ़िल्टर किए गए डेटा ({total:,} / {total_dataset_size:,} कुल समीक्षाएं) के आधार पर:*\n\n"
        elif lang == 'hinglish':
            filter_prefix = f"*Current filtered data ({total:,} of {total_dataset_size:,} reviews) ke mutabiq:*\n\n"
        else:
            filter_prefix = f"*Based on currently filtered reviews ({total:,} of {total_dataset_size:,} total)*\n\n"

    # Product Performance Breakdown Helper
    def get_product_stats():
        prod_stats = []
        if 'Product Name' in df.columns:
            for name, group in df.groupby('Product Name'):
                p_tot = len(group)
                if p_tot >= 1:
                    p_neg = len(group[group['sentiment'] == 'Negative'])
                    p_pos = len(group[group['sentiment'] == 'Positive'])
                    p_neu = len(group[group['sentiment'] == 'Neutral'])
                    p_rate = group['Rating'].mean()
                    prod_stats.append({
                        'Product': name,
                        'Total': p_tot,
                        'Negative': p_neg,
                        'Positive': p_pos,
                        'Neutral': p_neu,
                        'Neg_Pct': (p_neg / p_tot) * 100 if p_tot else 0,
                        'Pos_Pct': (p_pos / p_tot) * 100 if p_tot else 0,
                        'Avg_Rating': p_rate
                    })
        return pd.DataFrame(prod_stats)

    # Dynamic Theme & Friction Miner Helper
    def mine_friction_aspects(subset_df: pd.DataFrame) -> List[Tuple[str, int, float]]:
        if subset_df.empty:
            return []
        
        n_tot = len(subset_df)
        aspect_patterns = [
            ("Packaging & Transit Damage", r'\b(?:packag|box|damag|broken|crush|torn|dented|seal)\b'),
            ("Delivery & Transit Delays", r'\b(?:deliver|late|delay|ship|dispatch|courier|transit|wait)\b'),
            ("Support Response Latency", r'\b(?:support|service|reply|response|agent|ticket|customer care)\b'),
            ("Hardware & Build Quality", r'\b(?:qualit|defect|build|cheap|break|fail|battery|plastic|durab|sound)\b'),
            ("Pricing & Value Expectation", r'\b(?:price|cost|money|worth|expensive|waste)\b')
        ]
        
        found = []
        texts = subset_df['Review Text'].dropna().astype(str).tolist() if 'Review Text' in subset_df.columns else []
        for name, pattern in aspect_patterns:
            cnt = sum(1 for t in texts if re.search(pattern, t, re.IGNORECASE))
            if cnt > 0:
                found.append((name, cnt, (cnt / n_tot) * 100))
                
        found.sort(key=lambda x: x[1], reverse=True)
        return found[:3]

    def mine_praise_aspects(subset_df: pd.DataFrame) -> List[Tuple[str, int, float]]:
        if subset_df.empty:
            return []
        p_tot = len(subset_df)
        praise_patterns = [
            ("Ergonomic Comfort & Build Quality", r'\b(?:comfort|qualit|build|finish|solid|premium|sturdy)\b'),
            ("Smooth Performance & Usability", r'\b(?:smooth|fast|perform|easy|responsive|accurate|sound)\b'),
            ("Prompt Delivery & Packaging", r'\b(?:fast|quick|packag|safely|ontime|intact|prompt)\b'),
            ("Helpful Customer Service", r'\b(?:support|helpful|assist|polite|service|resolved)\b'),
            ("Great Value for Money", r'\b(?:value|worth|price|deal|budget|affordable)\b')
        ]
        found = []
        texts = subset_df['Review Text'].dropna().astype(str).tolist() if 'Review Text' in subset_df.columns else []
        for name, pattern in praise_patterns:
            cnt = sum(1 for t in texts if re.search(pattern, t, re.IGNORECASE))
            if cnt > 0:
                found.append((name, cnt, (cnt / p_tot) * 100))
        found.sort(key=lambda x: x[1], reverse=True)
        return found[:3]

    def get_sample_quote(subset_df: pd.DataFrame) -> str:
        if 'Review Text' in subset_df.columns:
            for text in subset_df['Review Text'].dropna():
                clean = str(text).strip().replace('\n', ' ')
                if 20 <= len(clean) <= 180:
                    return clean
        return ""

    # =========================================================================
    # INTENT 1: Which product should we investigate first? / Priority attention
    # =========================================================================
    investigate_triggers_en = ['investigate', 'priority', 'prioritize', 'attention first', 'investigate first', 'look into first', 'fix first', 'audit first', 'highest risk', 'critical issue']
    investigate_triggers_hi = ['जांच', 'सबसे पहले ध्यान', 'तत्काल ध्यान', 'प्राथमिकता', 'जोखिम']
    investigate_triggers_hinglish = ['investigate pehle', 'pehle kisko', 'kisko investigate', 'attention kisko', 'sabse pehle kis product', 'dhyan kis par', 'priority kisko', 'pehle check']

    if (any(k in query_lower for k in investigate_triggers_en) or
        any(k in query for k in investigate_triggers_hi) or
        any(k in query_lower for k in investigate_triggers_hinglish)):

        p_df = get_product_stats()
        if p_df.empty:
            return "No product review statistics are available to analyze."
            
        # Prioritize by negative percentage and absolute negative volume
        worst_candidates = p_df.sort_values(by=['Neg_Pct', 'Negative'], ascending=[False, False])
        top_problem = worst_candidates.iloc[0]
        
        # Mine friction points for this product
        prod_negs = neg_df[neg_df['Product Name'] == top_problem['Product']] if 'Product Name' in neg_df.columns else pd.DataFrame()
        prod_frictions = mine_friction_aspects(prod_negs)
        sample_q = get_sample_quote(prod_negs)

        if lang == 'hi':
            f_lines = [f"• **{name}**: {cnt} शिकायतें ({pct:.1f}%)" for name, cnt, pct in prod_frictions]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### 🚨 प्राथमिकता जांच: **{top_problem['Product']}**

सक्रिय डेटासेट के विश्लेषण के आधार पर, **{top_problem['Product']}** की सबसे पहले जांच की जानी चाहिए। इसमें ग्राहक असंतोष की दर सबसे अधिक दर्ज की गई है।

**मुख्य आंकड़े (Key Metrics):**
• **नकारात्मक समीक्षाएं:** **{top_problem['Negative']} / {top_problem['Total']}** ({top_problem['Neg_Pct']:.1f}% शिकायत दर)
• **औसत रेटिंग:** **{top_problem['Avg_Rating']:.2f} ⭐**

**शीर्ष ग्राहक समस्याएं (Customer Pain Points):**
{chr(10).join(f_lines) if f_lines else "• पैकेजिंग क्षति और डिलीवरी में देरी।"}
{quote_md}

**कार्यकारी अनुशंसा (Recommended Action):**
सप्लाई चेन और पारगमन पैकेजिंग की तत्काल ऑडिट करें तथा **{top_problem['Product']}** के लिए गुणवत्ता नियंत्रण प्रोटोकॉल लागू करें।
""".strip()
        elif lang == 'hinglish':
            f_lines = [f"• **{name}**: {cnt} complaints ({pct:.1f}%)" for name, cnt, pct in prod_frictions]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### 🚨 Priority Investigation: **{top_problem['Product']}**

Active dataset analysis ke mutabiq, **{top_problem['Product']}** ko sabse pehle investigate karna chahiye kyunki yahan customer dissatisfaction sabse zyada concentrated hai.

**Key Performance Stats:**
• **Negative Complaints:** **{top_problem['Negative']} out of {top_problem['Total']} reviews** ({top_problem['Neg_Pct']:.1f}% negative rate)
• **Average Rating:** **{top_problem['Avg_Rating']:.2f} ⭐**

**Main Issues Reported:**
{chr(10).join(f_lines) if f_lines else "• Packaging damage and shipment transit delays."}
{quote_md}

**Executive Action Item:**
**{top_problem['Product']}** ke supplier batches ki quality check karein aur shipping transit protection ko immediately upgrade karein.
""".strip()
        else:
            f_lines = [f"• **{name}**: {cnt} complaints ({pct:.1f}%)" for name, cnt, pct in prod_frictions]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### 🚨 Investigation Priority: **{top_problem['Product']}**

Based on the active review dataset, **{top_problem['Product']}** should be investigated first. It has the highest concentration of customer friction, accounting for **{top_problem['Negative']} negative reviews out of {top_problem['Total']} total** (**{top_problem['Neg_Pct']:.1f}% negative complaint rate**).

**Core Diagnostic Metrics:**
• **Dissatisfaction Share:** **{top_problem['Neg_Pct']:.1f}%** ({top_problem['Negative']} of {top_problem['Total']} reviews)
• **Average Rating:** **{top_problem['Avg_Rating']:.2f} / 5.0 ⭐**

**Primary Issues Driving Negative Feedback:**
{chr(10).join(f_lines) if f_lines else "• Transit packaging damage and fulfillment delays."}
{quote_md}

**Recommended Management Action:**
Conduct an immediate quality audit on **{top_problem['Product']}** inventory batches and enforce reinforced box packaging to prevent transit damage.
""".strip()

    # =========================================================================
    # INTENT 2: Which product has the most negative reviews? / Worst product
    # =========================================================================
    neg_triggers_en = ['most negative', 'worst product', 'highest complaint', 'lowest rating', 'problem product', 'most complaints', 'least liked']
    neg_triggers_hi = ['सबसे ज्यादा negative', 'सबसे खराब', 'सबसे ज्यादा शिकायत', 'नकारात्मक समीक्षा']
    neg_triggers_hinglish = ['sabse zyada negative', 'sabse jyada negative', 'sabse kharab', 'bura product', 'complaints kis']

    if (any(k in query_lower for k in neg_triggers_en) or
        any(k in query for k in neg_triggers_hi) or
        any(k in query_lower for k in neg_triggers_hinglish)):
        
        p_df = get_product_stats()
        if p_df.empty:
            return "No product review statistics available." if lang == 'en' else "उत्पाद समीक्षा के आंकड़े उपलब्ध नहीं हैं।"
            
        worst_sorted = p_df.sort_values(by=['Negative', 'Neg_Pct'], ascending=[False, False]).head(3)
        worst_prod = worst_sorted.iloc[0]
        
        # Mine friction themes for top problem product
        prod_negs = neg_df[neg_df['Product Name'] == worst_prod['Product']] if 'Product Name' in neg_df.columns else pd.DataFrame()
        frictions = mine_friction_aspects(prod_negs)
        sample_q = get_sample_quote(prod_negs)
        
        if lang == 'hi':
            lines = [f"• **{r['Product']}**: **{r['Negative']} नकारात्मक** / {r['Total']} कुल ({r['Neg_Pct']:.1f}%), रेटिंग: {r['Avg_Rating']:.2f} ⭐" for _, r in worst_sorted.iterrows()]
            f_lines = [f"• **{name}**: {cnt} शिकायतें" for name, cnt, _ in frictions]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### ⚠️ सबसे ज्यादा Negative Reviews वाला Product

डेटासेट के अनुसार, सबसे अधिक नकारात्मक समीक्षाएं **{worst_prod['Product']}** में दर्ज की गई हैं (**{worst_prod['Negative']} नकारात्मक समीक्षाएं**, {worst_prod['Neg_Pct']:.1f}%)।

**शीर्ष समस्याग्रस्त उत्पाद (Top Problem Products):**
{chr(10).join(lines)}

**प्रमुख समस्याएं:**
{chr(10).join(f_lines) if f_lines else "• पारगमन में होने वाली क्षति और देरी।"}
{quote_md}

**सुझाव:** **{worst_prod['Product']}** की पैकेजिंग और ग्राहक सेवा सहायता की तत्काल समीक्षा की जानी चाहिए।
""".strip()
        elif lang == 'hinglish':
            lines = [f"• **{r['Product']}**: **{r['Negative']} negative** out of {r['Total']} ({r['Neg_Pct']:.1f}%), Avg Rating: {r['Avg_Rating']:.2f} ⭐" for _, r in worst_sorted.iterrows()]
            f_lines = [f"• **{name}**: {cnt} complaints" for name, cnt, _ in frictions]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### ⚠️ Sabse Zyada Negative Reviews Wala Product

Dataset analysis ke mutabiq, sabse zyada complaints **{worst_prod['Product']}** ke liye darj hui hain (**{worst_prod['Negative']} negative reviews**, {worst_prod['Neg_Pct']:.1f}% complaint share).

**Top 3 Problem Products:**
{chr(10).join(lines)}

**Main Issues Mentioned:**
{chr(10).join(f_lines) if f_lines else "• Damaged packaging and delivery response delays."}
{quote_md}

**Recommendation:** **{worst_prod['Product']}** ke fulfillment aur customer feedback loop par foran action lein.
""".strip()
        else:
            lines = [f"• **{r['Product']}**: **{r['Negative']} negative reviews** out of {r['Total']} ({r['Neg_Pct']:.1f}%), Avg Rating: {r['Avg_Rating']:.2f} ⭐" for _, r in worst_sorted.iterrows()]
            f_lines = [f"• **{name}**: {cnt} complaints" for name, cnt, _ in frictions]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### ⚠️ Products with the Most Negative Reviews

Based on the current dataset, **{worst_prod['Product']}** has the highest volume of customer complaints with **{worst_prod['Negative']} negative reviews out of {worst_prod['Total']} total reviews** (**{worst_prod['Neg_Pct']:.1f}%**).

**Top 3 Problem Products:**
{chr(10).join(lines)}

**Main Issues Mentioned:**
{chr(10).join(f_lines) if f_lines else "• Protective packaging integrity and fulfillment delays."}
{quote_md}

**Management Recommendation:**
Conduct a focused review on **{worst_prod['Product']}** suppliers to address defect rates and audit transit packaging.
""".strip()

    # =========================================================================
    # INTENT 3: Highest rating / Best product / Best customer satisfaction
    # =========================================================================
    best_triggers_en = ['highest average rating', 'highest rating', 'best product', 'top rated', 'star product', 'best customer satisfaction', 'customer satisfaction', 'highest csat', 'csat', 'top performing', 'performing well', 'most liked', 'like most']
    best_triggers_hi = ['सबसे ज्यादा rating', 'सबसे अच्छा', 'सर्वश्रेष्ठ', 'सर्वोत्तम', 'सबसे बेहतरीन', 'पसंद', 'संतुष्टि']
    best_triggers_hinglish = ['highest rating', 'sabse accha', 'sabse best', 'achha product', 'top rating', 'sabse pasand', 'customer satisfaction', 'behtareen']

    if (any(k in query_lower for k in best_triggers_en) or
        any(k in query for k in best_triggers_hi) or
        any(k in query_lower for k in best_triggers_hinglish)):
        
        p_df = get_product_stats()
        if p_df.empty:
            return "No product review data available."
            
        best_sorted = p_df.sort_values(by=['Avg_Rating', 'Pos_Pct'], ascending=[False, False]).head(3)
        best_prod = best_sorted.iloc[0]
        
        prod_pos = pos_df[pos_df['Product Name'] == best_prod['Product']] if 'Product Name' in pos_df.columns else pd.DataFrame()
        praises = mine_praise_aspects(prod_pos)
        sample_q = get_sample_quote(prod_pos)
        
        if lang == 'hi':
            lines = [f"• **{r['Product']}**: **{r['Avg_Rating']:.2f} ⭐** ({r['Pos_Pct']:.1f}% सकारात्मक, {r['Positive']}/{r['Total']} समीक्षाएं)" for _, r in best_sorted.iterrows()]
            p_lines = [f"• **{name}**: {cnt} प्रशंसाएं" for name, cnt, _ in praises]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### 🌟 सर्वोच्च ग्राहक संतुष्टि: **{best_prod['Product']}**

डेटासेट में **{best_prod['Product']}** ग्राहकों की पहली पसंद है, जिसने **{best_prod['Avg_Rating']:.2f} / 5.0 ⭐** की औसत रेटिंग और **{best_prod['Pos_Pct']:.1f}% सकारात्मक दर** हासिल की है।

**शीर्ष 3 उत्कृष्ट उत्पाद:**
{chr(10).join(lines)}

**ग्राहकों को क्या पसंद आया:**
{chr(10).join(p_lines) if p_lines else "• प्रीमियम गुणवत्ता, उत्कृष्ट फिनिश और त्वरित डिलीवरी।"}
{quote_md}

**अनुशंसा:** **{best_prod['Product']}** के निर्माण मानकों को अन्य उत्पाद श्रेणियों के लिए एक बेंचमार्क के रूप में उपयोग करें।
""".strip()
        elif lang == 'hinglish':
            lines = [f"• **{r['Product']}**: **{r['Avg_Rating']:.2f} ⭐** ({r['Pos_Pct']:.1f}% positive, {r['Positive']}/{r['Total']} reviews)" for _, r in best_sorted.iterrows()]
            p_lines = [f"• **{name}**: {cnt} positive mentions" for name, cnt, _ in praises]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### 🌟 Best Customer Satisfaction: **{best_prod['Product']}**

Dataset analysis ke hisab se, **{best_prod['Product']}** customer satisfaction mein top par hai (**Avg Rating: {best_prod['Avg_Rating']:.2f} ⭐**, **{best_prod['Pos_Pct']:.1f}% positive reviews**).

**Top 3 Star Performers:**
{chr(10).join(lines)}

**Customer Delight Factors:**
{chr(10).join(p_lines) if p_lines else "• High build quality, comfortable design aur fast shipping."}
{quote_md}

**Takeaway:** **{best_prod['Product']}** ke best practices ko baki product catalog mein implement karein.
""".strip()
        else:
            lines = [f"• **{r['Product']}**: **{r['Avg_Rating']:.2f} / 5.0 ⭐** ({r['Pos_Pct']:.1f}% positive, {r['Positive']}/{r['Total']} reviews)" for _, r in best_sorted.iterrows()]
            p_lines = [f"• **{name}**: {cnt} positive mentions" for name, cnt, _ in praises]
            quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""
            return f"""{filter_prefix}### 🌟 Top Customer Satisfaction Leader: **{best_prod['Product']}**

Based on the active dataset, **{best_prod['Product']}** leads customer satisfaction with an outstanding average rating of **{best_prod['Avg_Rating']:.2f} / 5.0 ⭐** and a **{best_prod['Pos_Pct']:.1f}% positive sentiment rate** ({best_prod['Positive']} out of {best_prod['Total']} reviews).

**Top 3 Customer Favorites:**
{chr(10).join(lines)}

**Key Drivers of Customer Delight:**
{chr(10).join(p_lines) if p_lines else "• Premium build quality, ergonomic design, and prompt fulfillment."}
{quote_md}

**Strategic Recommendation:**
Benchmark **{best_prod['Product']}**'s product specifications and supplier quality standards across other catalog categories.
""".strip()

    # =========================================================================
    # INTENT 4: Product Comparison (Checked before generic keyword routing)
    # =========================================================================
    if any(k in query_lower for k in ['compare', ' vs ', ' vs. ', 'versus', 'muqabla', 'तुलना', 'difference between']):
        p_df = get_product_stats()
        if not p_df.empty and len(p_df) >= 2:
            matched_prods = [p for p in p_df['Product'].unique() if p.lower() in query_lower]
            if len(matched_prods) >= 2:
                r1 = p_df[p_df['Product'] == matched_prods[0]].iloc[0]
                r2 = p_df[p_df['Product'] == matched_prods[1]].iloc[0]
            else:
                r1 = p_df.sort_values(by=['Avg_Rating', 'Pos_Pct'], ascending=[False, False]).iloc[0]
                r2 = p_df.sort_values(by=['Neg_Pct', 'Negative'], ascending=[False, False]).iloc[0]

            if lang == 'hi':
                return f"""{filter_prefix}### ⚖️ उत्पाद तुलना विश्लेषण: **{r1['Product']}** बनाम **{r2['Product']}**

| मीट्रिक (Metric) | {r1['Product']} (लीडर) | {r2['Product']} (समीक्षाधीन) |
|---|---|---|
| **कुल समीक्षाएं** | {r1['Total']} | {r2['Total']} |
| **औसत रेटिंग** | {r1['Avg_Rating']:.2f} ⭐ | {r2['Avg_Rating']:.2f} ⭐ |
| **सकारात्मक दर** | {r1['Pos_Pct']:.1f}% | {r2['Pos_Pct']:.1f}% |
| **नकारात्मक दर** | {r1['Neg_Pct']:.1f}% | {r2['Neg_Pct']:.1f}% |

**विश्लेषणात्मक निष्कर्ष:** **{r1['Product']}** ग्राहक वफादारी में काफी आगे है, जबकि **{r2['Product']}** में सुधार की तत्काल आवश्यकता है।
""".strip()
            elif lang == 'hinglish':
                return f"""{filter_prefix}### ⚖️ Product Comparison: **{r1['Product']}** vs **{r2['Product']}**

| Metric | {r1['Product']} (Leader) | {r2['Product']} (Needs Review) |
|---|---|---|
| **Total Reviews** | {r1['Total']} | {r2['Total']} |
| **Average Rating** | {r1['Avg_Rating']:.2f} ⭐ | {r2['Avg_Rating']:.2f} ⭐ |
| **Positive Share** | {r1['Pos_Pct']:.1f}% | {r2['Pos_Pct']:.1f}% |
| **Negative Complaints** | {r1['Neg_Pct']:.1f}% | {r2['Neg_Pct']:.1f}% |

**Key Takeaway:** **{r1['Product']}** customer advocacy mein outperform kar raha hai, jabki **{r2['Product']}** par quality audit zaroori hai.
""".strip()
            else:
                return f"""{filter_prefix}### ⚖️ Product Performance Comparison

| Performance Metric | {r1['Product']} (Top Performer) | {r2['Product']} (High Friction) |
|---|---|---|
| **Total Reviews** | {r1['Total']} | {r2['Total']} |
| **Average Rating** | {r1['Avg_Rating']:.2f} ⭐ | {r2['Avg_Rating']:.2f} ⭐ |
| **Positive Share** | {r1['Pos_Pct']:.1f}% | {r2['Pos_Pct']:.1f}% |
| **Negative Share** | {r1['Neg_Pct']:.1f}% | {r2['Neg_Pct']:.1f}% |

**Analytical Summary:** **{r1['Product']}** demonstrates significantly higher customer advocacy, whereas **{r2['Product']}** requires targeted quality intervention.
""".strip()

    # =========================================================================
    # INTENT 5: Specific Topic / Feature inquiries (checked before generic friction)
    # =========================================================================
    topic_keywords = ['mileage', 'battery', 'delivery', 'packaging', 'price', 'quality', 'support', 'service', 'sound', 'screen', 'shipping', 'transit', 'keyboard', 'chair']
    matched_topic = next((t for t in topic_keywords if t in query_lower), None)
    if matched_topic:
        topic_reviews = df[df['Review Text'].str.contains(matched_topic, case=False, na=False)]
        if not topic_reviews.empty:
            t_cnt = len(topic_reviews)
            t_neg = len(topic_reviews[topic_reviews['sentiment'] == 'Negative'])
            t_pos = len(topic_reviews[topic_reviews['sentiment'] == 'Positive'])
            t_avg = topic_reviews['Rating'].mean()
            sample_t = get_sample_quote(topic_reviews)
            quote_md = f"\n> *\"{sample_t}\"*" if sample_t else ""

            if lang == 'hi':
                return f"""{filter_prefix}### 🔍 '{matched_topic.capitalize()}' पर ग्राहक प्रतिक्रिया विश्लेषण

• **उल्लेखों की संख्या:** **{t_cnt} समीक्षाएं** ({t_cnt/total*100:.1f}% डेटासेट)
• **औसत रेटिंग:** **{t_avg:.2f} / 5.0 ⭐**
• **सकारात्मक समीक्षाएं:** {t_pos} ({t_pos/t_cnt*100:.1f}%) 🟢
• **नकारात्मक समीक्षाएं:** {t_neg} ({t_neg/t_cnt*100:.1f}%) 🔴
{quote_md}

**निष्कर्ष:** ग्राहक प्रतिक्रिया से स्पष्ट है कि '{matched_topic}' अनुभव का एक प्रमुख निर्णय कारक है।
""".strip()
            elif lang == 'hinglish':
                return f"""{filter_prefix}### 🔍 Topic Analysis: '{matched_topic.capitalize()}'

• **Total Mentions:** **{t_cnt} reviews** ({t_cnt/total*100:.1f}% of dataset)
• **Average Rating:** **{t_avg:.2f} / 5.0 ⭐**
• **Positive Sentiment:** {t_pos} ({t_pos/t_cnt*100:.1f}%) 🟢
• **Negative Complaints:** {t_neg} ({t_neg/t_cnt*100:.1f}%) 🔴
{quote_md}

**Key Takeaway:** '{matched_topic.capitalize()}' mentions highlight important feedback for ongoing operations.
""".strip()
            else:
                return f"""{filter_prefix}### 🔍 Customer Feedback on '{matched_topic.capitalize()}'

• **Total Topic Mentions:** **{t_cnt} reviews** ({t_cnt/total*100:.1f}% of evaluated corpus)
• **Average Rating on Topic:** **{t_avg:.2f} / 5.0 ⭐**
• **Positive Sentiment:** {t_pos} ({t_pos/t_cnt*100:.1f}%) 🟢
• **Negative Complaints:** {t_neg} ({t_neg/t_cnt*100:.1f}%) 🔴
{quote_md}

**Operational Insight:** Sentiment for '{matched_topic}' directly impacts overall customer retention and return rates.
""".strip()
        else:
            # Topic not in this dataset
            if lang == 'hi':
                return f"{filter_prefix}ℹ️ वर्तमान डेटासेट में **'{matched_topic}'** से संबंधित कोई समीक्षा दर्ज नहीं है। हमारे डेटासेट में इलेक्ट्रॉनिक्स व ऑफिस एसेसरीज जैसे मॉनिटर, कीबोर्ड, ऑफिस चेयर और हेडफ़ोन शामिल हैं।"
            elif lang == 'hinglish':
                return f"{filter_prefix}ℹ️ Current dataset mein **'{matched_topic}'** ka koi review nahi mila. Catalog mein Monitor, Keyboard, Office Chair aur Headphones available hain."
            else:
                return f"{filter_prefix}ℹ️ No reviews in the current active dataset mention **'{matched_topic}'**. The catalog covers products such as Monitors, Keyboards, Chairs, and Headphones."

    # =========================================================================
    # INTENT 6: Why are customers unhappy? / Main complaints / Pain points
    # =========================================================================
    unhappy_triggers_en = ['why are customers unhappy', 'why unhappy', 'unhappy', 'complaint', 'complaints', 'pain point', 'pain points', 'frustrat', 'friction', 'dissatisfied', 'what is wrong', 'main issues', 'problem']
    unhappy_triggers_hi = ['ग्राहक क्यों नाराज', 'unhappy क्यों', 'शिकायतें क्या', 'समस्या', 'परेशानी', 'pain point', 'असंतोष']
    unhappy_triggers_hinglish = ['unhappy kyun', 'unhappy kyu', 'complaints kya', 'problem kya', 'naraz kyu', 'dikkat kya', 'pain points kya', 'pareshani kya']

    if (any(k in query_lower for k in unhappy_triggers_en) or
        any(k in query for k in unhappy_triggers_hi) or
        any(k in query_lower for k in unhappy_triggers_hinglish)):

        if neg_count == 0:
            if lang == 'hi':
                return f"{filter_prefix}🎉 **शानदार!** सक्रिय डेटासेट में कोई नकारात्मक समीक्षा नहीं है। सभी ग्राहक संतुष्ट हैं!"
            elif lang == 'hinglish':
                return f"{filter_prefix}🎉 **Bahut badiya!** Current dataset mein 0 negative reviews hain. Customers fully satisfied hain!"
            else:
                return f"{filter_prefix}🎉 **Great news!** There are currently 0 negative reviews in the active dataset. Customers are reporting high satisfaction!"

        frictions = mine_friction_aspects(neg_df)
        sample_q = get_sample_quote(neg_df)
        quote_md = f"\n> *\"{sample_q}\"*" if sample_q else ""

        if lang == 'hi':
            f_lines = [f"• **{name}** ({cnt} शिकायतें, {pct:.1f}%): पारगमन में क्षतिग्रस्त या विलंबित।" for name, cnt, pct in frictions]
            return f"""{filter_prefix}### 🔍 ग्राहक असंतोष और मुख्य समस्याएं (Customer Friction Analysis)

कुल **{total:,} समीक्षाओं** में से **{neg_count:,} ग्राहकों ({neg_count/total*100:.1f}%)** ने नकारात्मक अनुभव दर्ज किया है (औसत शिकायत रेटिंग: **{neg_avg_rating:.2f} ⭐**)।

**असंतोष के मुख्य कारण (Top Friction Drivers):**
{chr(10).join(f_lines) if f_lines else "• उत्पाद गुणवत्ता और शिपमेंट में सामान्य असंतोष।"}

**ग्राहकों की वास्तविक प्रतिक्रिया (Voice of Customer):**
{quote_md}

**रणनीतिक सुझाव:** वेयरहाउस पैकेजिंग सुरक्षा को अपग्रेड करें और शिपिंग पार्टनर के साथ डिलीवरी टाइमलाइन सुनिश्चित करें।
""".strip()
        elif lang == 'hinglish':
            f_lines = [f"• **{name}** ({cnt} complaints, {pct:.1f}%): Shipping damage aur delivery delays." for name, cnt, pct in frictions]
            return f"""{filter_prefix}### 🔍 Customers Unhappy Kyun Hain? (Main Complaints & Friction)

Total **{total:,} reviews** mein se **{neg_count:,} customers ({neg_count/total*100:.1f}%)** ne negative dissatisfaction feedback diya hai (Avg complaint rating: **{neg_avg_rating:.2f} ⭐**).

**Top Customer Pain Points:**
{chr(10).join(f_lines) if f_lines else "• Logistics fulfillment delays aur product packaging issues."}

**Sample Voice of Customer:**
{quote_md}

**Key Takeaway:** Logistics partner ke sath coordination improve karein taaki damaged goods aur delivery delays ko eliminate kiya ja sake.
""".strip()
        else:
            f_lines = [f"• **{name}** ({cnt} complaints, {pct:.1f}%): Issues reported in product transit or post-purchase support." for name, cnt, pct in frictions]
            return f"""{filter_prefix}### 🔍 Customer Friction & Dissatisfaction Analysis

Out of **{total:,} total reviews**, **{neg_count:,} customers ({neg_count/total*100:.1f}%)** reported dissatisfaction and friction, resulting in an average complaint rating of **{neg_avg_rating:.2f} / 5.0 ⭐**.

**Primary Customer Pain Points:**
{chr(10).join(f_lines) if f_lines else "• Logistics fulfillment delays and packaging damage."}

**Representative Voice of the Customer:**
{quote_md}

**Executive Action Plan:**
Management should prioritize courier packaging durability and support response triage to directly mitigate the primary sources of customer dissatisfaction.
""".strip()

    # =========================================================================
    # INTENT 7: Summarize customer feedback / Overall sentiment
    # =========================================================================
    summary_triggers_en = ['summarize', 'summary', 'overall sentiment', 'sentiment distribution', 'overview', 'feedback summary', 'executive summary', 'briefing', 'what is the sentiment', 'general sentiment']
    summary_triggers_hi = ['सारांश', 'समग्र सेंटिमेंट', 'summary बताएं', 'कैसा सेंटिमेंट है', 'समीक्षाओं का सारांश', 'सेंटिमेंट कैसा']
    summary_triggers_hinglish = ['summary batao', 'overall sentiment kya hai', 'summarize karo', 'sentiment kaisa hai', 'feedback ka overview', 'overall feedback']

    if (any(k in query_lower for k in summary_triggers_en) or
        any(k in query for k in summary_triggers_hi) or
        any(k in query_lower for k in summary_triggers_hinglish)):

        net_score = (pos_count/total*100) - (neg_count/total*100) if total else 0.0
        sentiment_label = "Predominantly Positive" if pos_count > (neg_count * 1.5) else ("Balanced / Mixed" if pos_count >= neg_count else "High Friction")

        if lang == 'hi':
            return f"""{filter_prefix}### 📊 ग्राहक प्रतिक्रिया का कार्यकारी सारांश (Executive Summary)

कुल **{total:,} समीक्षाओं** के आधार पर, समग्र ग्राहक अनुभव **{sentiment_label}** है, जिसकी औसत रेटिंग **{avg_rating:.2f} / 5.0 ⭐** और नेट सेंटिमेंट स्कोर **{'+' if net_score >= 0 else ''}{net_score:.1f}** है।

**सेंटिमेंट वितरण:**
• 🟢 **सकारात्मक (Positive):** {pos_count:,} समीक्षाएं ({pos_count/total*100:.1f}%) — उच्च उत्पाद गुणवत्ता और डिज़ाइन प्रशंसा।
• 🟡 **तटस्थ (Neutral):** {neu_count:,} समीक्षाएं ({neu_count/total*100:.1f}%) — मानक अपेक्षाएं और मूल्य मिलान।
• 🔴 **नकारात्मक (Negative):** {neg_count:,} समीक्षाएं ({neg_count/total*100:.1f}%) — डिलीवरी में देरी और पैकेजिंग क्षति।

**रणनीतिक निष्कर्ष:** उत्पाद के प्रति ग्राहक विश्वास मजबूत है, लेकिन लॉजिस्टिक्स में सुधार से ग्राहक संतुष्टि और बढ़ सकती है।
""".strip()
        elif lang == 'hinglish':
            return f"""{filter_prefix}### 📊 Executive Customer Feedback Summary

Total **{total:,} reviews** ke analysis ke mutabiq, overall customer sentiment **{sentiment_label}** hai (Avg Rating: **{avg_rating:.2f} ⭐**, Net Sentiment Score: **{'+' if net_score >= 0 else ''}{net_score:.1f}**).

**Sentiment Breakdown:**
• 🟢 **Positive Feedback:** {pos_count:,} reviews ({pos_count/total*100:.1f}%) — Build quality aur ergonomics customer favorites hain.
• 🟡 **Neutral Feedback:** {neu_count:,} reviews ({neu_count/total*100:.1f}%) — Standard expectation fulfillment.
• 🔴 **Negative Complaints:** {neg_count:,} reviews ({neg_count/total*100:.1f}%) — Courier delays aur transit packaging issues.

**Summary:** Brand advocacy strong hai, lekin supply-chain delivery packaging par foran dhyan dena zaroori hai.
""".strip()
        else:
            return f"""{filter_prefix}### 📊 Executive Customer Feedback Summary

Across the **{total:,} total reviews** analyzed, the overall customer sentiment is **{sentiment_label}** with an average rating of **{avg_rating:.2f} / 5.0 ⭐** and a Net Sentiment Score of **{'+' if net_score >= 0 else ''}{net_score:.1f}**.

**Sentiment Distribution:**
• 🟢 **Positive Sentiment:** {pos_count:,} reviews ({pos_count/total*100:.1f}%) — Driven by solid build quality, ergonomic comfort, and functional reliability.
• 🟡 **Neutral Sentiment:** {neu_count:,} reviews ({neu_count/total*100:.1f}%) — Reflects standard expectations and balanced pricing satisfaction.
• 🔴 **Negative Sentiment:** {neg_count:,} reviews ({neg_count/total*100:.1f}%) — Driven by fulfillment delivery delays and transit packaging damage.

**Executive Strategic Context:**
Customer advocacy is robust, but targeted improvements in logistics handling and packaging will protect brand reputation.
""".strip()

    # =========================================================================
    # INTENT 8: Review Counts (Positive, Negative, Rating)
    # =========================================================================
    if any(k in query_lower for k in ['how many positive', 'positive reviews', 'count of positive', 'number of positive', 'kitne positive', 'positive kitne']):
        if lang == 'hi':
            return f"""{filter_prefix}### 🟢 सकारात्मक समीक्षाओं का विवरण (Positive Reviews)
• **कुल सकारात्मक समीक्षाएं:** **{pos_count:,}**
• **डेटासेट में हिस्सा:** **{pos_count/total*100:.1f}%**
• **सकारात्मक समीक्षाओं की औसत रेटिंग:** **{pos_avg_rating:.2f} ⭐**
""".strip()
        elif lang == 'hinglish':
            return f"""{filter_prefix}### 🟢 Positive Reviews Breakdown
• **Total Positive Reviews:** **{pos_count:,}**
• **Dataset Percentage:** **{pos_count/total*100:.1f}%**
• **Average Rating (Positive):** **{pos_avg_rating:.2f} ⭐**
""".strip()
        else:
            return f"""{filter_prefix}### 🟢 Positive Reviews Breakdown
• **Total Positive Reviews:** **{pos_count:,}**
• **Percentage of Dataset:** **{pos_count/total*100:.1f}%**
• **Average Rating among Positive Reviews:** **{pos_avg_rating:.2f} / 5.0 ⭐**
""".strip()

    if any(k in query_lower for k in ['how many negative', 'negative reviews', 'count of negative', 'number of negative', 'kitne negative', 'negative kitne']):
        if lang == 'hi':
            return f"""{filter_prefix}### 🔴 नकारात्मक समीक्षाओं का विवरण (Negative Reviews)
• **कुल नकारात्मक समीक्षाएं:** **{neg_count:,}**
• **डेटासेट में हिस्सा:** **{neg_count/total*100:.1f}%**
• **नकारात्मक समीक्षाओं की औसत रेटिंग:** **{neg_avg_rating:.2f} ⭐**
""".strip()
        elif lang == 'hinglish':
            return f"""{filter_prefix}### 🔴 Negative Reviews Breakdown
• **Total Negative Reviews:** **{neg_count:,}**
• **Dataset Percentage:** **{neg_count/total*100:.1f}%**
• **Average Rating (Negative):** **{neg_avg_rating:.2f} ⭐**
""".strip()
        else:
            return f"""{filter_prefix}### 🔴 Negative Reviews Breakdown
• **Total Negative Complaints:** **{neg_count:,}**
• **Percentage of Dataset:** **{neg_count/total*100:.1f}%**
• **Average Rating among Negative Reviews:** **{neg_avg_rating:.2f} / 5.0 ⭐**
""".strip()

    if any(k in query_lower for k in ['average rating', 'avg rating', 'mean rating', 'rating score', 'rating kitni', 'औसत rating', 'औसत रेटिंग']):
        star_counts = df['Rating'].value_counts().sort_index(ascending=False).to_dict() if 'Rating' in df.columns else {}
        star_lines = [f"• **{int(star)} Stars:** {cnt} reviews ({cnt/total*100:.1f}%)" for star, cnt in star_counts.items()]
        if lang == 'hi':
            return f"""{filter_prefix}### ⭐ समग्र रेटिंग विश्लेषण (Rating Analysis)
• **औसत रेटिंग (Average Rating):** **{avg_rating:.2f} / 5.0 ⭐**
• **कुल विश्लेषित समीक्षाएं:** **{total:,}**

**स्टार ब्रेकडाउन:**
{chr(10).join(star_lines)}
""".strip()
        elif lang == 'hinglish':
            return f"""{filter_prefix}### ⭐ Average Rating Analysis
• **Overall Average Rating:** **{avg_rating:.2f} / 5.0 ⭐**
• **Total Reviews:** **{total:,}**

**Rating Breakdown:**
{chr(10).join(star_lines)}
""".strip()
        else:
            return f"""{filter_prefix}### ⭐ Rating Analysis
• **Overall Average Rating:** **{avg_rating:.2f} / 5.0 ⭐**
• **Total Evaluated Reviews:** **{total:,}**

**Star Breakdown:**
{chr(10).join(star_lines)}
""".strip()

    # =========================================================================
    # INTENT 9: Specific Product Dossier Inquiry
    # =========================================================================
    if 'Product Name' in df.columns:
        for prod_name in df['Product Name'].unique():
            if str(prod_name).lower() in query_lower:
                sub_df = df[df['Product Name'] == prod_name]
                sub_tot = len(sub_df)
                sub_pos = len(sub_df[sub_df['sentiment'] == 'Positive'])
                sub_neg = len(sub_df[sub_df['sentiment'] == 'Negative'])
                sub_avg = sub_df['Rating'].mean()
                sample_text = get_sample_quote(sub_df)
                quote_md = f"\n> *\"{sample_text}\"*" if sample_text else ""
                
                if lang == 'hi':
                    return f"""{filter_prefix}### 🏷️ उत्पाद विवरण: **{prod_name}**
• **कुल समीक्षाएं:** **{sub_tot}**
• **औसत रेटिंग:** **{sub_avg:.2f} ⭐**
• **सकारात्मक (Positive):** {sub_pos} ({sub_pos/sub_tot*100:.1f}%)
• **नकारात्मक (Negative):** {sub_neg} ({sub_neg/sub_tot*100:.1f}%)
{quote_md}
""".strip()
                elif lang == 'hinglish':
                    return f"""{filter_prefix}### 🏷️ Product Dossier: **{prod_name}**
• **Total Reviews:** **{sub_tot}**
• **Average Rating:** **{sub_avg:.2f} ⭐**
• **Positive Sentiment:** {sub_pos} ({sub_pos/sub_tot*100:.1f}%)
• **Negative Complaints:** {sub_neg} ({sub_neg/sub_tot*100:.1f}%)
{quote_md}
""".strip()
                else:
                    return f"""{filter_prefix}### 🏷️ Product Dossier: **{prod_name}**
• **Total Reviews:** **{sub_tot}**
• **Average Rating:** **{sub_avg:.2f} / 5.0 ⭐**
• **Positive Sentiment:** {sub_pos} ({sub_pos/sub_tot*100:.1f}%)
• **Negative Sentiment:** {sub_neg} ({sub_neg/sub_tot*100:.1f}%)
{quote_md}
""".strip()

    # =========================================================================
    # INTENT 10: Generic Query with Informative Keyword Matching
    # =========================================================================
    stop_words = {'what', 'which', 'where', 'when', 'would', 'could', 'should', 'about', 'there', 'their', 'with', 'from', 'this', 'that', 'have', 'been', 'tell', 'show', 'give', 'does'}
    tokens = [w for w in re.findall(r'\b\w+\b', query_lower) if len(w) >= 4 and w not in stop_words]
    if tokens:
        matched = df[df['Review Text'].str.contains('|'.join(tokens), case=False, na=False)]
        if not matched.empty and len(matched) >= 2:
            m_count = len(matched)
            m_avg = matched['Rating'].mean()
            m_pos = len(matched[matched['sentiment'] == 'Positive'])
            sample_m = get_sample_quote(matched)
            sample_md = f"\n> *\"{sample_m}\"*" if sample_m else ""
            
            if lang == 'hi':
                return f"""{filter_prefix}### 🔎 प्रश्न परिणाम (Search Results)
• आपके खोज शब्दों से संबंधित **{m_count} समीक्षाएं** मिलीं।
• **औसत रेटिंग:** **{m_avg:.2f} ⭐**
• **सकारात्मक दर:** **{m_pos/m_count*100:.1f}%**
{sample_md}
""".strip()
            elif lang == 'hinglish':
                return f"""{filter_prefix}### 🔎 Query Results for \"{query}\"
• Search keywords se match karti hui **{m_count} reviews** mili hain.
• **Average Rating:** **{m_avg:.2f} ⭐**
• **Positive Feedback:** **{m_pos/m_count*100:.1f}%**
{sample_md}
""".strip()
            else:
                return f"""{filter_prefix}### 🔎 Query Insights for \"{query}\"
• Found **{m_count} matching reviews** mentioning your search keywords.
• **Average Rating:** **{m_avg:.2f} / 5.0 ⭐**
• **Positive Proportion:** **{m_pos/m_count*100:.1f}%**
{sample_md}
""".strip()

    # =========================================================================
    # INTENT 11: Out-of-Scope Fallback with Dataset Boundaries Guidance
    # =========================================================================
    if lang == 'hi':
        return f"""{filter_prefix}### 🤖 Pulse AI सहायक (Pulse AI Assistant)

मैं **InsightPulse AI** का विश्लेषक सहायक हूँ, जो विशेष रूप से आपके सक्रिय **{total:,} समीक्षाओं** वाले ग्राहक डेटासेट से जुड़ा हूँ।

मेरे पास सामान्य वेब ज्ञान की जानकारी नहीं है, लेकिन आप मुझसे अपने समीक्षा डेटा के बारे में पूछ सकते हैं:
• *"सबसे ज्यादा negative reviews किस product के हैं?"*
• *"ग्राहक क्यों असंतुष्ट हैं?"*
• *"सबसे पहले किस उत्पाद की जांच करनी चाहिए?"*
• *"सर्वोच्च ग्राहक संतुष्टि वाला उत्पाद कौन सा है?"*
• *"समग्र सेंटिमेंट क्या है?"*
• *"कितनी सकारात्मक समीक्षाएं हैं?"*
""".strip()
    elif lang == 'hinglish':
        return f"""{filter_prefix}### 🤖 Pulse AI Assistant

Main **InsightPulse AI** ka analytical assistant hoon, jo aapke active review dataset (**{total:,} reviews**, Avg Rating: **{avg_rating:.2f} ⭐**) se live connected hoon.

Mere paas external general web info nahi hai, lekin aap mujhse dataset analytics pooch sakte hain:
• *"Sabse zyada negative reviews kis product ke hain?"*
• *"Customers unhappy kyun hain?"*
• *"Sabse pehle kis product ko investigate karna chahiye?"*
• *"Highest rating kis product ki hai?"*
• *"Overall sentiment kaisa hai?"*
• *"Positive reviews kitne hain?"*
""".strip()
    else:
        return f"""{filter_prefix}### 🤖 Pulse AI Assistant

I am **Pulse 🤖**, your AI Customer Experience Analyst connected to your active dataset of **{total:,} reviews** (Avg Rating: **{avg_rating:.2f} ⭐**).

I am dedicated exclusively to analyzing your customer review corpus. You can ask me analytical questions such as:
• *"Which product has the highest average rating?"*
• *"Which product has the most negative reviews?"*
• *"Which product should we investigate first?"*
• *"Why are customers unhappy?"*
• *"What is the overall sentiment?"*
• *"What are the most common customer pain points?"*
• *"How many positive reviews do we have?"*
""".strip()
