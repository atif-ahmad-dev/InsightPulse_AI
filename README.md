# InsightPulse AI

## AI-Powered Customer Sentiment & Experience Analytics

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.0+-3F4F75.svg?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

**InsightPulse AI** is an enterprise-grade AI analytics application built to transform raw, unstructured customer feedback into actionable customer experience (CX) intelligence. Combining modern Natural Language Processing (NLP), interactive Data Science visualization, and a dataset-grounded conversational AI assistant, InsightPulse AI empowers product managers, CX leaders, and operations teams to identify customer friction points, track sentiment shifts over time, and resolve product quality issues.

---

## 🎯 Problem Statement

Modern e-commerce and retail brands receive thousands of customer reviews across diverse touchpoints every day. Organizations encounter three fundamental challenges:
1. **Unstructured Data Overload**: Customer feedback is qualitative, messy, and time-consuming to aggregate manually.
2. **Delayed Response to Friction**: Traditional review audits fail to detect emerging defect patterns or shipping bottlenecks before ratings drop.
3. **Lack of Actionable Insights**: Traditional dashboards assign surface-level positive/negative labels without uncovering the operational root cause or journey stage.

---

## 💡 Solution

**InsightPulse AI** bridges this gap through a unified analytics platform:
- **Aspect & Friction Mining**: Automatically isolates specific operational concerns (e.g., *Delivery & Logistics*, *Packaging Quality*, *Product Durability*, *Customer Support*).
- **Grounded Conversational Intelligence**: An integrated **Pulse AI Assistant** capable of answering analytical questions directly from active dataset slices in natural language.
- **Cross-Browser Voice Interaction**: Hands-free voice querying with real-time speech recognition and automatic text-to-speech (TTS) responses.
- **Executive CX Analytics**: Real-time Net Sentiment Score (NSS), CSAT estimation, journey stage tracking, and executive takeaways.

---

## ✨ Key Features

### 1. 🏠 Executive Sentiment Dashboard
- **Real-Time KPIs**: Total Reviews, Average Star Rating, Positive %, Neutral %, Negative %, and Catalog Product Count.
- **7 Theme-Aware Interactive Plotly Visualizations**:
  - *Sentiment Distribution*: Interactive donut chart with category breakdowns.
  - *Rating Distribution*: Frequency bar chart with average rating reference line.
  - *Sentiment Trend Timeline*: Temporal sentiment tracking across dates.
  - *Product-Wise Sentiment Breakdown*: Comparative sentiment across catalog products.
  - *Rating vs. Sentiment Alignment*: Cross-tabulation of star ratings vs. NLP sentiment.
  - *Aspect & Keyword Signals*: Frequency distribution of top praise and pain-point keywords.
  - *Journey Stage Distribution*: Tracks feedback across touchpoints (*Delivery*, *Ordering*, *Support*, *After-sales*, *Browsing*).
- **Multi-Dimensional Sidebar Filters**: Filter simultaneously by Product, Star Rating (1–5), Sentiment (Positive/Neutral/Negative), and Date Range.

### 2. 🤖 Pulse AI Assistant (Multilingual & Cross-Browser Voice)
- **Dataset-Grounded Analytics**: Evaluates real dataset numbers, calculates proportions, and formats clean markdown tables and bullet points.
- **Multilingual Understanding**:
  - **English**: *"Which product has the most negative reviews?"*, *"Why are customers unhappy?"*
  - **Hindi (हिन्दी / Devanagari)**: *"सबसे ज्यादा negative reviews किस product के हैं?"*
  - **Hinglish**: *"Customers unhappy kyun hain?"*, *"Sabse accha product kaunsa hai?"*
- **Cross-Browser Voice Input**:
  - **Google Chrome & Microsoft Edge**: Native Web Speech API (`webkitSpeechRecognition`) with streaming transcript.
  - **Mozilla Firefox**: HTML5 `MediaRecorder` audio capture with server-side WAV transcription fallback (`SpeechRecognition`).
- **Automatic Voice Read-Aloud (TTS)**: AI answers are spoken automatically aloud with language-specific voice selection (`hi-IN` / `en-US`).
- **Listen to AI Answer**: Manual audio replay button to replay the latest answer at any time.
- **Instant Suggested Questions**: Clickable query pill buttons in English, Hindi, and Hinglish.

### 3. 🔍 AI Review Analyzer (Single Review Intelligence)
- Test custom or pre-set customer reviews with instant NLP inference.
- Outputs:
  - **Sentiment Classification**: Positive, Neutral, or Negative with confidence score (0–100%).
  - **Detected Operational Issue**: Classifies friction into operational domains.
  - **Clause Extraction**: Separates positive praise from negative friction clauses in mixed feedback.
  - **Emotion Classification**: Identifies underlying customer emotion (*Delighted*, *Satisfied*, *Neutral*, *Frustrated*, *Disappointed*, *Impatient*).
  - **Prescriptive Recommendation**: Actionable advice for operations and product teams.

### 4. 🧠 AI Customer Experience (CX) Insights
- **Net Sentiment Score (NSS)** and **Estimated CSAT %**.
- **Automated Strategic Takeaways**: Dynamically generated executive briefings based on statistical findings.
- **Star Performers vs. Problem Products**: Highlights highest-rated items and high-risk products needing investigation.

### 5. 📤 Dynamic File Upload & Validation
- Ingest custom **CSV** or **Excel (.xlsx, .xls)** datasets.
- Resilient column auto-detection (recognizes `Review Text`, `review`, `comment`, `Rating`, `score`, `Product Name`, `Date`, etc.).
- Instant recalculation of all metrics, charts, and Pulse AI knowledge base.
- One-click restore to built-in 500-review benchmark dataset.

### 6. 📋 Review Explorer & Multi-Format Export
- Full-text keyword search across review texts.
- Sort reviews by Rating, Date, or Sentiment polarity.
- Download Filtered Reviews CSV, Full Enriched Dataset CSV, or Executive Briefing text summary.

### 7. 🎨 SaaS Design & Theme Switcher
- **☀️ Light Mode & 🌙 Dark Mode**: High-contrast, theme-adapted colors across KPI cards, tables, widgets, and Plotly charts.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend Framework** | Streamlit (Python-native web app framework) |
| **Data Processing & Analytics** | Pandas, NumPy |
| **Interactive Visualizations** | Plotly Express, Plotly Graph Objects |
| **Natural Language Processing** | NLTK (VADER), TextBlob, Regex Aspect Mining |
| **Multilingual NLP** | Devanagari Unicode script detector, Romanized Hinglish normalizer |
| **Speech-to-Text (STT)** | Browser Web Speech API, Python `SpeechRecognition` (Firefox fallback) |
| **Text-to-Speech (TTS)** | Browser Web Speech Synthesis API (`SpeechSynthesisUtterance`) |
| **Optional LLM Enhancements** | Google Gemini (`google-generativeai`), OpenAI (`openai`) |
| **File I/O** | OpenPyXL (Excel), CSV |

---

## 🔬 System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │      Customer Review Data / Upload      │
                    │         (CSV / Excel / Built-in)        │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Resilient DataLoader & Schema      │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       ┌────────────────────────┐               ┌───────────────────────────┐
       │   Batch NLP Pipeline   │               │ Pulse AI Assistant Engine │
       │  - VADER Polarity      │               │  - Multilingual Routing   │
       │  - TextBlob Subjectiv. │               │  - Grounded Data Math     │
       │  - Aspect Extraction   │               │  - (Optional) Gemini/GPT  │
       └────────────┬───────────┘               └─────────────┬─────────────┘
                    │                                         │
                    ▼                                         ▼
       ┌────────────────────────┐               ┌───────────────────────────┐
       │   Executive Dashboard  │               │    Cross-Browser Audio    │
       │  - 7 Plotly Charts     │               │  - Chrome/Edge Web Speech │
       │  - CX Insights & KPIs  │               │  - Firefox MediaRecorder  │
       │  - Review Explorer     │               │  - Auto TTS Synthesis     │
       └────────────────────────┘               └───────────────────────────┘
```

---

## 📁 Project Structure

```
InsightPulse_AI/
├── insightpulse_ai.py           # Primary Streamlit Application Entrypoint
├── customer-reviews-1000.csv    # Benchmark Dataset (500 Real-World Reviews)
├── full_end_to_end_qa.py        # Master 20-Point Automated QA Test Suite
├── requirements.txt             # Production Dependencies
├── README.md                    # Comprehensive Portfolio Documentation
├── .gitignore                   # Git Exclusions (.venv, cache, secrets)
├── .streamlit/
│   └── config.toml              # Server & Layout Configuration
├── data/
│   ├── customer-reviews-1000.csv# Structured Dataset Copy
│   └── audio_samples/           # Test Audio Verification Samples
│       ├── hello_world.wav
│       └── atif.wav
├── utils/
│   ├── __init__.py
│   ├── audio_transcriber.py     # Cross-Browser Audio Transcription Fallback
│   ├── chatbot_engine.py        # Pulse AI Assistant & Grounded Query Engine
│   ├── data_loader.py           # File Ingestion & Schema Mapper
│   ├── insights_engine.py       # Executive CX Metrics & Strategic Narratives
│   ├── nlp_engine.py            # Sentiment, Aspect & Emotion Classification
│   ├── ui_components.py         # SaaS Layout, Themes, Charts & Audio Bridge
│   └── voice_input_component/   # Custom Streamlit Voice Component
│       └── index.html           # Cross-Browser Microphone & TTS Component
├── test_chatbot.py              # Chatbot Engine Test Suite (20/20 Passed)
├── test_cross_browser_voice.py  # Voice Engine Test Suite (Passed)
├── test_theme_and_voice.py      # Theme & Audio Test Suite (5/5 Passed)
└── test_suite.py                # Core Integration Test Suite (6/6 Passed)
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/InsightPulse_AI.git
cd InsightPulse_AI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (Command Prompt / PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Quality Assurance
```bash
python full_end_to_end_qa.py
```

### 5. Launch the Application
```bash
streamlit run insightpulse_ai.py
```

Open your browser at **`http://localhost:8501`**.

---

## 💡 Example Usage

1. **Exploring the Executive Dashboard**:
   - Use the sidebar filters to select a product (e.g., *"Gaming Mouse Wireless"*).
   - All 7 charts and KPI metrics dynamically recalculate to reflect only that slice.
2. **Asking Pulse AI Analytical Questions**:
   - Type or speak: *"Which product has the most negative reviews?"*
   - Pulse AI analyzes the active dataset and returns:
     > *"Based on the current dataset, USB-C Hub Multiport has the highest number of negative reviews (11 negative reviews, 4.0 average rating)."*
   - The AI voice engine automatically reads the answer aloud.
3. **Multilingual Queries**:
   - Type or speak: *"Customers unhappy kyun hain?"*
   - Pulse AI extracts common customer pain points across battery, delivery, and durability in natural Hinglish.
4. **Analyzing a Single Customer Review**:
   - Navigate to the **Review Analyzer** tab.
   - Enter: *"The sound quality is exceptional, but the charging cable broke after 3 days."*
   - Pulse AI classifies the mixed review, identifying positive praise for audio alongside operational friction in *Product Durability*.

---

## 🖼️ Application Screenshots

*(Screenshots captured from live InsightPulse AI application)*

| Executive Dashboard & KPIs | Pulse AI Assistant (Multilingual & Voice) |
|:---:|:---:|
| *(Light & Dark Theme Analytics with 7 Plotly Charts)* | *(Hands-free voice querying with automatic speech response)* |

| Single Review Intelligence | Customer Journey & Aspect Mining |
|:---:|:---:|
| *(Aspect extraction, confidence score & emotion detection)* | *(Touchpoint sentiment breakdown & friction silos)* |

---

## 🔑 Optional Generative AI Configuration

InsightPulse AI operates out-of-the-box using local NLP engines without requiring any external API keys or network calls.

If you wish to optionally enhance Pulse Assistant with Google Gemini or OpenAI:
1. **Via Application Sidebar**: Open the "AI Engine Settings" expander and enter your API key.
2. **Via `.streamlit/secrets.toml`** (locally ignored by git):
   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   OPENAI_API_KEY = "your-openai-key"
   ```
3. **Via Environment Variables**:
   ```bash
   export GEMINI_API_KEY="your-gemini-key"
   ```

---

## 📊 Dataset Information

The included benchmark dataset (`customer-reviews-1000.csv`) contains **500 real-world customer reviews** spanning 30 unique electronics and office accessory products from January 2026 to July 2026.

| Field | Type | Description |
|---|---|---|
| `Review ID` | String | Unique review identifier (e.g. `R000001`) |
| `Buyer ID` | String | Anonymized customer identifier |
| `Seller ID` | String | Marketplace merchant identifier |
| `Product Name` | String | Specific product catalog item |
| `Review Text` | String | Raw qualitative customer feedback |
| `Rating` | Integer | Star rating from 1 to 5 |
| `Journey Stage` | String | Touchpoint (*Delivery*, *Ordering*, *Support*, *After-sales*, *Browsing*) |
| `Date` | Date | Review timestamp (`YYYY-MM-DD`) |
| `Sentiment` | String | Ground-truth sentiment label |

---

## 🔮 Future Enhancements

- [ ] **Automated PDF Executive Report**: One-click generation of formatted slide summaries for board presentations.
- [ ] **Live Webhook Integration**: Direct ingestion from Zendesk, Shopify, and Amazon Seller Central APIs.
- [ ] **Fine-Tuned Domain SLM**: Lightweight quantized models (e.g., Gemma-2B) for zero-latency local enterprise deployment.

---

## 📄 License

This project is open-source and licensed under the **[MIT License](LICENSE)**.

---

## 👨‍💻 Author

**Data Science & AI Portfolio Project**  
*Built with Python, Streamlit, Plotly & Multilingual Natural Language Processing*
