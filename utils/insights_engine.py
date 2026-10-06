# utils/insights_engine.py
"""
InsightPulse AI - Customer Experience & Business Insights Engine
Derives actionable intelligence, pain points, star performers, friction areas,
and automated executive summaries from customer review datasets.
"""

import re
from collections import Counter
from typing import Dict, List, Any, Tuple
import pandas as pd
import streamlit as st

# Comprehensive English Stopwords (Zero-dependency, avoids NLTK missing corpus errors)
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn', 'product', 'item', 'bought', 'one', 'get', 'got', 'even',
    'also', 'really', 'would', 'could', 'ordered', 'received', 'use', 'using', 'used'
}

def extract_top_keywords(texts: List[str], n: int = 15) -> List[Tuple[str, int]]:
    """Extract most frequent significant words filtering out noise and stopwords."""
    words = []
    for text in texts:
        if pd.isna(text):
            continue
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
        words.extend([w for w in tokens if w not in STOP_WORDS and len(w) > 2])
    return Counter(words).most_common(n)

def extract_top_phrases(texts: List[str], n: int = 8) -> List[Tuple[str, int]]:
    """Extract frequent bi-grams (2-word phrases) for deeper context."""
    phrases = []
    for text in texts:
        if pd.isna(text):
            continue
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
        clean_tokens = [w for w in tokens if w not in STOP_WORDS]
        for i in range(len(clean_tokens) - 1):
            phrase = f"{clean_tokens[i]} {clean_tokens[i+1]}"
            phrases.append(phrase)
    return Counter(phrases).most_common(n)

def compute_cx_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive Customer Experience analytics and friction points."""
    if df.empty:
        return {}
        
    total_reviews = len(df)
    avg_rating = df['Rating'].mean() if 'Rating' in df.columns else 0.0
    
    pos_df = df[df['sentiment'] == 'Positive']
    neg_df = df[df['sentiment'] == 'Negative']
    neu_df = df[df['sentiment'] == 'Neutral']
    
    pos_pct = (len(pos_df) / total_reviews * 100) if total_reviews else 0
    neg_pct = (len(neg_df) / total_reviews * 100) if total_reviews else 0
    neu_pct = (len(neu_df) / total_reviews * 100) if total_reviews else 0
    
    # Net Sentiment Score: Pos % - Neg % (Scale -100 to +100)
    net_sentiment_score = pos_pct - neg_pct
    
    # CSAT Estimate (% of reviews rating 4 or 5)
    high_rating_count = len(df[df['Rating'] >= 4]) if 'Rating' in df.columns else len(pos_df)
    csat_score = (high_rating_count / total_reviews * 100) if total_reviews else 0
    
    # Extract keywords
    pos_keywords = extract_top_keywords(pos_df['Review Text'].tolist(), n=12)
    neg_keywords = extract_top_keywords(neg_df['Review Text'].tolist(), n=12)
    neg_phrases = extract_top_phrases(neg_df['Review Text'].tolist(), n=8)
    pos_phrases = extract_top_phrases(pos_df['Review Text'].tolist(), n=8)
    
    # Journey Stage Breakdown
    stage_metrics = []
    if 'Journey Stage' in df.columns:
        for stage, group in df.groupby('Journey Stage'):
            count = len(group)
            stage_pos = (len(group[group['sentiment'] == 'Positive']) / count) * 100
            stage_neg = (len(group[group['sentiment'] == 'Negative']) / count) * 100
            stage_rating = group['Rating'].mean()
            stage_metrics.append({
                'Journey Stage': stage,
                'Reviews': count,
                'Avg Rating': round(stage_rating, 2),
                'Positive %': round(stage_pos, 1),
                'Negative %': round(stage_neg, 1)
            })
    stage_df = pd.DataFrame(stage_metrics)
    if not stage_df.empty:
        stage_df = stage_df.sort_values(by='Negative %', ascending=False)
        
    # Product Performance Breakdown
    prod_metrics = []
    if 'Product Name' in df.columns:
        for prod, group in df.groupby('Product Name'):
            count = len(group)
            if count >= 3:  # Only evaluate products with statistical significance
                p_pos = (len(group[group['sentiment'] == 'Positive']) / count) * 100
                p_neg = (len(group[group['sentiment'] == 'Negative']) / count) * 100
                p_rating = group['Rating'].mean()
                prod_metrics.append({
                    'Product Name': prod,
                    'Reviews': count,
                    'Avg Rating': round(p_rating, 2),
                    'Positive %': round(p_pos, 1),
                    'Negative %': round(p_neg, 1)
                })
    prod_df = pd.DataFrame(prod_metrics)
    
    star_products = prod_df.sort_values(by=['Positive %', 'Avg Rating'], ascending=False).head(5) if not prod_df.empty else pd.DataFrame()
    problem_products = prod_df.sort_values(by=['Negative %', 'Avg Rating'], ascending=[False, True]).head(5) if not prod_df.empty else pd.DataFrame()
    
    # Aspect Breakdown (Friction areas)
    aspect_counts = {}
    if 'detected_aspect' in df.columns:
        for aspect, group in df.groupby('detected_aspect'):
            total_aspect = len(group)
            neg_aspect = len(group[group['sentiment'] == 'Negative'])
            aspect_counts[aspect] = {
                'total': total_aspect,
                'negative': neg_aspect,
                'neg_rate': round((neg_aspect / total_aspect * 100), 1) if total_aspect else 0
            }
            
    # Automated Executive Summary Narratives
    executive_narratives = []
    
    # 1. Overall sentiment narrative
    if net_sentiment_score > 25:
        executive_narratives.append(
            f"🟢 **Healthy Customer Sentiment**: Net Sentiment Score is strongly positive at **+{net_sentiment_score:.1f}** "
            f"with **{pos_pct:.1f}%** positive feedback and an average rating of **{avg_rating:.2f}/5.0**."
        )
    elif net_sentiment_score >= 0:
        executive_narratives.append(
            f"🟡 **Moderate Sentiment**: Net Sentiment Score stands at **+{net_sentiment_score:.1f}**. "
            f"While **{pos_pct:.1f}%** of customers are satisfied, **{neg_pct:.1f}%** report notable friction."
        )
    else:
        executive_narratives.append(
            f"🔴 **Elevated Customer Friction**: Negative sentiment (**{neg_pct:.1f}%**) outpaces positive sentiment (**{pos_pct:.1f}%**). "
            f"Immediate intervention needed across top complaint categories."
        )
        
    # 2. Key Pain Point narrative
    if not stage_df.empty:
        worst_stage = stage_df.iloc[0]
        executive_narratives.append(
            f"⚠️ **Primary Friction Touchpoint**: The **{worst_stage['Journey Stage']}** stage experiences the highest negativity "
            f"(**{worst_stage['Negative %']}%** negative reviews, Avg Rating: {worst_stage['Avg Rating']})."
        )
        
    # 3. Product Pain Point narrative
    if not problem_products.empty:
        worst_prod = problem_products.iloc[0]
        executive_narratives.append(
            f"🚨 **Vulnerable Product**: **{worst_prod['Product Name']}** has the highest negative review share "
            f"(**{worst_prod['Negative %']}%** negative across {worst_prod['Reviews']} reviews)."
        )
        
    # 4. Delight Driver
    if not star_products.empty:
        best_prod = star_products.iloc[0]
        executive_narratives.append(
            f"⭐ **Customer Favorite**: **{best_prod['Product Name']}** leads with **{best_prod['Positive %']}%** positive sentiment "
            f"and an outstanding **{best_prod['Avg Rating']} ⭐** rating."
        )
        
    return {
        'total_reviews': total_reviews,
        'avg_rating': round(avg_rating, 2),
        'pos_pct': round(pos_pct, 1),
        'neg_pct': round(neg_pct, 1),
        'neu_pct': round(neu_pct, 1),
        'net_sentiment_score': round(net_sentiment_score, 1),
        'csat_score': round(csat_score, 1),
        'pos_keywords': pos_keywords,
        'neg_keywords': neg_keywords,
        'pos_phrases': pos_phrases,
        'neg_phrases': neg_phrases,
        'stage_df': stage_df,
        'star_products': star_products,
        'problem_products': problem_products,
        'aspect_counts': aspect_counts,
        'executive_narratives': executive_narratives
    }
