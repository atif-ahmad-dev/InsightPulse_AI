# utils/nlp_engine.py
"""
InsightPulse AI - Natural Language Processing & Sentiment Intelligence Engine
Supports Hugging Face Transformers models with automated fallback to
a high-precision Hybrid Local NLP Engine (NLTK VADER + TextBlob + Aspect Mining).
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from textblob import TextBlob
import pandas as pd
import streamlit as st

# Setup NLTK VADER safely
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except (LookupError, IndexError):
        nltk.download('vader_lexicon', quiet=True)
    _sia = SentimentIntensityAnalyzer()
except Exception:
    _sia = None

# Aspect Category Taxonomy with weighted keyword patterns
ASPECT_CATEGORIES = {
    "Delivery & Logistics": [
        r'\bdeliv\w*\b', r'\bship\w*\b', r'\bcourier\b', r'\blate\b', r'\bdelay\w*\b',
        r'\barriv\w*\b', r'\btransit\b', r'\bdispatch\w*\b', r'\btracking\b'
    ],
    "Packaging & Handling": [
        r'\bpackag\w*\b', r'\bbox\b', r'\bdamag\w*\b', r'\bcrush\w*\b', r'\bdent\w*\b',
        r'\bseal\w*\b', r'\bprotect\w*\b', r'\bwrapped\b'
    ],
    "Product Quality & Build": [
        r'\bquality\b', r'\bdurab\w*\b', r'\bbuild\b', r'\bmaterial\b', r'\bcheap\b',
        r'\bflimsy\b', r'\bsturdy\b', r'\bdefect\w*\b', r'\bbroken\b', r'\bfail\w*\b',
        r'\bpremium\b', r'\bsolid\b', r'\bmalfunction\w*\b'
    ],
    "Customer Support & Service": [
        r'\bsupport\b', r'\bservice\b', r'\brepresentative\b', r'\bagent\b', r'\bhelpdesk\b',
        r'\bpatient\b', r'\brude\b', r'\bunhelpful\b', r'\bhelpful\b', r'\bquery\b',
        r'\bresolved\b', r'\bresponse time\b', r'\bassistance\b'
    ],
    "Pricing & Value": [
        r'\bprice\b', r'\bcost\b', r'\bexpensive\b', r'\boverpriced\b', r'\bworth\b',
        r'\bvalue\b', r'\bwaste of money\b', r'\bbargain\b', r'\baffordable\b'
    ],
    "Setup & Usability": [
        r'\bsetup\b', r'\binstall\w*\b', r'\binstruct\w*\b', r'\bmanual\b', r'\bcomplicat\w*\b',
        r'\beasy\b', r'\bsimple\b', r'\bconfus\w*\b', r'\bplug and play\b'
    ],
    "Ordering & Checkout": [
        r'\bcheckout\b', r'\border\w*\b', r'\bpayment\b', r'\bgateway\b', r'\bcart\b',
        r'\bwebsite\b', r'\bnavigat\w*\b', r'\bbrows\w*\b'
    ],
    "Battery & Charging": [
        r'\bbattery\b', r'\bcharg\w*\b', r'\bdrain\w*\b', r'\boverheat\w*\b', r'\blifespan\b'
    ]
}

# Emotion Keywords & Linguistic Signals
EMOTION_PATTERNS = {
    "Delighted / Enthusiastic": [
        r'\bamaz\w*\b', r'\bexcel\w*\b', r'\blove\b', r'\bsuperb\b', r'\bfantastic\b',
        r'\bexceeded\b', r'\bawesome\b', r'\bperfect\b', r'\bbrilliant\b', r'\b10/10\b'
    ],
    "Satisfied / Pleased": [
        r'\bgood\b', r'\bdecent\b', r'\bnice\b', r'\bpleased\b', r'\bhappy\b',
        r'\bsatisf\w*\b', r'\bworks well\b', r'\bdoes the job\b', r'\bintact\b'
    ],
    "Neutral / Factual": [
        r'\bokay\b', r'\bstandard\b', r'\baverage\b', r'\bacceptable\b', r'\badequate\b',
        r'\bexpected\b', r'\bmoderate\b', r'\bas described\b'
    ],
    "Frustrated / Annoyed": [
        r'\buseless\b', r'\bannoy\w*\b', r'\bterrible\b', r'\bhorrible\b', r'\bworst\b',
        r'\bunhelpful\b', r'\brude\b', r'\bwaste\b', r'\bnever again\b', r'\bfurious\b'
    ],
    "Disappointed / Let Down": [
        r'\bdisappoint\w*\b', r'\bpoor\b', r'\bbad\b', r'\bregret\b', r'\bfail\w*\b',
        r'\bnot worth\b', r'\bdamaged\b', r'\bflimsy\b', r'\bbroken\b'
    ],
    "Impatient / Anxious": [
        r'\blate\b', r'\bdelay\w*\b', r'\bwaiting\b', r'\bforever\b', r'\bslow\b',
        r'\bwhere is\b', r'\bstill waiting\b'
    ]
}

@st.cache_resource(show_spinner=False)
def load_hf_pipeline():
    """
    Attempt to load a Hugging Face Transformers sentiment model if available.
    Returns None if transformers or PyTorch are not installed.
    """
    try:
        from transformers import pipeline
        nlp = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        return nlp
    except Exception:
        return None

def detect_aspects(text: str) -> List[str]:
    """Detect customer experience aspects/topics mentioned in text."""
    if not text or not isinstance(text, str):
        return ["General Experience"]
    
    text_lower = text.lower()
    matched = []
    for category, patterns in ASPECT_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched.append(category)
                break
    return matched if matched else ["General Feedback"]

def detect_emotion(text: str, sentiment: str) -> str:
    """Classify user emotional undertone based on linguistic cues and polarity."""
    if not text or not isinstance(text, str):
        return "Neutral / Factual"
    
    text_lower = text.lower()
    for emotion, patterns in EMOTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return emotion
                
    if sentiment == "Positive":
        return "Satisfied / Pleased"
    elif sentiment == "Negative":
        return "Disappointed / Let Down"
    return "Neutral / Factual"

def extract_aspect_clauses(text: str) -> Tuple[List[str], List[str]]:
    """Extract positive and negative clauses/aspects using sentiment clause splitting."""
    if not text or not isinstance(text, str):
        return [], []
    
    # Split review by sentence or contrastive conjunctions (but, however, yet, although, while)
    clauses = re.split(r'[,;.\n]+|\b(?:but|however|although|yet|though|except)\b', text, flags=re.IGNORECASE)
    positive_aspects = []
    negative_aspects = []
    
    for clause in clauses:
        clause_clean = clause.strip()
        if len(clause_clean.split()) < 2:
            continue
        
        blob = TextBlob(clause_clean)
        pol = blob.sentiment.polarity
        
        # Also check with VADER if available
        if _sia:
            vader_score = _sia.polarity_scores(clause_clean)['compound']
            combined_pol = (pol * 0.4) + (vader_score * 0.6)
        else:
            combined_pol = pol
            
        if combined_pol >= 0.15:
            positive_aspects.append(clause_clean)
        elif combined_pol <= -0.15:
            negative_aspects.append(clause_clean)
            
    # Fallback to whole text categorization if clauses weren't distinct
    if not positive_aspects and not negative_aspects:
        blob = TextBlob(text)
        if blob.sentiment.polarity > 0.1:
            positive_aspects.append(text)
        elif blob.sentiment.polarity < -0.1:
            negative_aspects.append(text)
            
    return positive_aspects[:3], negative_aspects[:3]

def generate_ai_insight(
    review_text: str,
    sentiment: str,
    aspects: List[str],
    emotion: str,
    positive_aspects: List[str],
    negative_aspects: List[str]
) -> str:
    """Generate concise, actionable business insights from analysis results."""
    primary_aspect = aspects[0] if aspects else "General Experience"
    
    if sentiment == "Positive":
        pos_highlight = f" ('{positive_aspects[0]}')" if positive_aspects else ""
        return (
            f"Customer exhibits a positive tone with strong satisfaction in **{primary_aspect}**{pos_highlight}. "
            f"Recommended Action: Acknowledge positive advocacy and preserve standards in this operational area."
        )
    elif sentiment == "Negative":
        neg_highlight = f" ('{negative_aspects[0]}')" if negative_aspects else ""
        return (
            f"Customer expresses distress regarding **{primary_aspect}**{neg_highlight}, showing **{emotion}** emotion. "
            f"Recommended Action: Priority intervention required. Flag to {primary_aspect.lower()} operations team for remediation."
        )
    else:
        return (
            f"Customer sentiment is balanced with neutral expectations on **{primary_aspect}**. "
            f"Recommended Action: Opportunity to elevate experience by addressing minor friction points."
        )

def analyze_single_review(text: str, force_local: bool = False) -> Dict[str, Any]:
    """
    Comprehensive single review analysis returning sentiment, confidence,
    aspects, positive/negative components, emotion, and actionable insights.
    """
    if not text or not str(text).strip():
        return {
            "sentiment": "Neutral",
            "confidence": 50.0,
            "polarity": 0.0,
            "subjectivity": 0.0,
            "detected_issues": ["No content"],
            "positive_aspects": [],
            "negative_aspects": [],
            "emotion": "Neutral / Factual",
            "insight": "Empty or whitespace review provided.",
            "model_used": "None"
        }
    
    clean_text = str(text).strip()
    hf_pipe = None if force_local else load_hf_pipeline()
    
    # 1. Base Sentiment Scoring
    blob = TextBlob(clean_text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity
    
    if _sia:
        vader_scores = _sia.polarity_scores(clean_text)
        compound = vader_scores['compound']
        pos_score = vader_scores['pos']
        neg_score = vader_scores['neg']
        neu_score = vader_scores['neu']
    else:
        compound = tb_polarity
        pos_score = max(0.0, tb_polarity)
        neg_score = max(0.0, -tb_polarity)
        neu_score = 1.0 - (pos_score + neg_score)

    if hf_pipe is not None:
        try:
            hf_res = hf_pipe(clean_text[:512])[0]
            hf_label = hf_res['label'].upper()
            hf_score = float(hf_res['score'])
            
            if hf_label == 'POSITIVE' and hf_score > 0.6:
                sentiment = 'Positive'
                confidence = round(hf_score * 100, 1)
            elif hf_label == 'NEGATIVE' and hf_score > 0.6:
                sentiment = 'Negative'
                confidence = round(hf_score * 100, 1)
            else:
                sentiment = 'Neutral'
                confidence = round((1.0 - abs(hf_score - 0.5) * 2) * 100, 1)
            model_used = "Hugging Face DistilBERT (Transformer)"
        except Exception:
            sentiment, confidence, model_used = _compute_local_sentiment(compound, tb_polarity)
    else:
        sentiment, confidence, model_used = _compute_local_sentiment(compound, tb_polarity)

    # 2. Extract Aspects, Emotion, and Clauses
    aspects = detect_aspects(clean_text)
    emotion = detect_emotion(clean_text, sentiment)
    pos_aspects, neg_aspects = extract_aspect_clauses(clean_text)
    
    # Identify specific customer issue
    if neg_aspects:
        detected_issue = f"{aspects[0]} ({neg_aspects[0]})"
    elif aspects:
        detected_issue = aspects[0]
    else:
        detected_issue = "General Experience"

    # 3. Formulate Actionable Insight
    if neg_aspects and pos_aspects:
        insight = (
            f"Mixed experience detected: High praise for **{pos_aspects[0]}**, but friction caused by **{neg_aspects[0]}**. "
            f"Recommended Action: Protect core product quality while targeting operational fix for {aspects[0].lower()}."
        )
    else:
        insight = generate_ai_insight(clean_text, sentiment, aspects, emotion, pos_aspects, neg_aspects)
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "polarity": round(compound, 3),
        "subjectivity": round(tb_subjectivity, 3),
        "detected_issues": aspects,
        "detected_customer_issue": detected_issue,
        "positive_aspects": pos_aspects,
        "negative_aspects": neg_aspects,
        "emotion": emotion,
        "insight": insight,
        "model_used": model_used
    }

def _compute_local_sentiment(compound: float, tb_polarity: float) -> Tuple[str, float, str]:
    """Helper to compute sentiment label and calibrated confidence score."""
    # Blend VADER compound and TextBlob polarity
    blended = (compound * 0.7) + (tb_polarity * 0.3)
    
    if blended >= 0.12:
        sentiment = 'Positive'
        confidence = min(99.0, max(55.0, (abs(blended) * 45.0) + 55.0))
    elif blended <= -0.12:
        sentiment = 'Negative'
        confidence = min(99.0, max(55.0, (abs(blended) * 45.0) + 55.0))
    else:
        sentiment = 'Neutral'
        confidence = min(95.0, max(50.0, (1.0 - abs(blended)) * 80.0))
        
    model_used = "InsightPulse Hybrid NLP Engine (VADER + TextBlob + Aspect Mining)"
    return sentiment, round(confidence, 1), model_used

@st.cache_data(show_spinner=False)
def analyze_dataset_sentiment(df: pd.DataFrame, text_col: str = 'Review Text') -> pd.DataFrame:
    """
    Perform high-performance batch sentiment analysis on an entire dataframe.
    Adds 'sentiment', 'polarity', 'subjectivity', 'confidence', 'journey_aspect', 'emotion'.
    """
    df_result = df.copy()
    
    if text_col not in df_result.columns:
        return df_result
        
    sentiments = []
    polarities = []
    subjectivities = []
    confidences = []
    primary_aspects = []
    emotions = []
    
    for text in df_result[text_col]:
        if pd.isna(text) or str(text).strip() == '':
            sentiments.append('Neutral')
            polarities.append(0.0)
            subjectivities.append(0.0)
            confidences.append(50.0)
            primary_aspects.append('General')
            emotions.append('Neutral / Factual')
            continue
            
        str_text = str(text)
        blob = TextBlob(str_text)
        tb_pol = blob.sentiment.polarity
        tb_sub = blob.sentiment.subjectivity
        
        if _sia:
            compound = _sia.polarity_scores(str_text)['compound']
        else:
            compound = tb_pol
            
        sent, conf, _ = _compute_local_sentiment(compound, tb_pol)
        aspects = detect_aspects(str_text)
        emotion = detect_emotion(str_text, sent)
        
        sentiments.append(sent)
        polarities.append(round(compound, 3))
        subjectivities.append(round(tb_sub, 3))
        confidences.append(conf)
        primary_aspects.append(aspects[0] if aspects else 'General')
        emotions.append(emotion)
        
    df_result['sentiment'] = sentiments
    df_result['polarity'] = polarities
    df_result['subjectivity'] = subjectivities
    df_result['confidence'] = confidences
    df_result['detected_aspect'] = primary_aspects
    df_result['detected_emotion'] = emotions
    
    return df_result
