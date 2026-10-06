# utils/data_loader.py
"""
InsightPulse AI - Resilient Data Loader & Validator
Handles standard and custom CSV/Excel uploads with automated column detection,
data cleansing, and seamless fallback to sample datasets.
"""

import os
import io
import pandas as pd
import streamlit as st
from typing import Tuple, Optional

# Standard Column Schema
REQUIRED_TEXT_CANDIDATES = ['Review Text', 'review_text', 'review', 'comment', 'feedback', 'text', 'body', 'content']
RATING_CANDIDATES = ['Rating', 'rating', 'score', 'stars', 'rate', 'star_rating']
PRODUCT_CANDIDATES = ['Product Name', 'product_name', 'product', 'item', 'item_name', 'title']
DATE_CANDIDATES = ['Date', 'date', 'timestamp', 'created_at', 'review_date', 'time']
STAGE_CANDIDATES = ['Journey Stage', 'journey_stage', 'stage', 'category', 'touchpoint']
SENTIMENT_CANDIDATES = ['Sentiment', 'sentiment', 'sentiment_label']
ID_CANDIDATES = ['Review ID', 'review_id', 'id', 'id_review']

def find_matching_column(columns, candidates):
    """Find the best matching column name case-insensitively."""
    col_map = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None

@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Load the built-in customer reviews dataset with reliable path resolution."""
    candidate_paths = [
        'customer-reviews-1000.csv',
        os.path.join('data', 'customer-reviews-1000.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'customer-reviews-1000.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'customer-reviews-1000.csv')
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return standardize_dataframe(df)
            
    # If not found on disk, raise FileNotFoundError with context
    raise FileNotFoundError("Default dataset 'customer-reviews-1000.csv' could not be located.")

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, fill missing values, and parse dates."""
    standardized = pd.DataFrame()
    cols = df.columns
    
    # Text Column (Required)
    text_col = find_matching_column(cols, REQUIRED_TEXT_CANDIDATES)
    if not text_col:
        raise ValueError(
            f"Uploaded file missing customer review text. Looked for columns such as: {', '.join(REQUIRED_TEXT_CANDIDATES[:4])}"
        )
    standardized['Review Text'] = df[text_col].astype(str).fillna('').str.strip()
    # Filter out completely empty reviews
    standardized = standardized[standardized['Review Text'] != ''].reset_index(drop=True)
    
    # Review ID
    id_col = find_matching_column(cols, ID_CANDIDATES)
    if id_col:
        standardized['Review ID'] = df[id_col].astype(str)
    else:
        standardized['Review ID'] = [f"REV-{i+1:05d}" for i in range(len(standardized))]
        
    # Product Name
    prod_col = find_matching_column(cols, PRODUCT_CANDIDATES)
    if prod_col:
        standardized['Product Name'] = df[prod_col].fillna('General Product').astype(str)
    else:
        standardized['Product Name'] = 'General Product'
        
    # Rating
    rating_col = find_matching_column(cols, RATING_CANDIDATES)
    if rating_col:
        standardized['Rating'] = pd.to_numeric(df[rating_col], errors='coerce').fillna(3.0)
        # Cap between 1 and 5
        standardized['Rating'] = standardized['Rating'].clip(1, 5)
    else:
        standardized['Rating'] = 3.0
        
    # Journey Stage / Category
    stage_col = find_matching_column(cols, STAGE_CANDIDATES)
    if stage_col:
        standardized['Journey Stage'] = df[stage_col].fillna('General').astype(str)
    else:
        standardized['Journey Stage'] = 'General Experience'
        
    # Date
    date_col = find_matching_column(cols, DATE_CANDIDATES)
    if date_col:
        try:
            standardized['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            standardized['Date'] = standardized['Date'].dt.date
        except Exception:
            standardized['Date'] = pd.to_datetime('today').date()
    else:
        standardized['Date'] = pd.to_datetime('today').date()
        
    # Sentiment (Optional pre-computed)
    sent_col = find_matching_column(cols, SENTIMENT_CANDIDATES)
    if sent_col:
        standardized['Sentiment'] = df[sent_col].fillna('Neutral').astype(str)
        # Standardize capitalization
        standardized['Sentiment'] = standardized['Sentiment'].apply(
            lambda s: 'Positive' if 'pos' in str(s).lower() else ('Negative' if 'neg' in str(s).lower() else 'Neutral')
        )
    
    return standardized

def process_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Process an uploaded CSV or Excel file.
    Returns (standardized_df, error_message).
    """
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a .csv or .xlsx file."
            
        if df.empty:
            return None, "The uploaded file is empty. Please provide a file with review rows."
            
        standardized_df = standardize_dataframe(df)
        return standardized_df, None
        
    except ValueError as ve:
        return None, str(ve)
    except Exception as e:
        return None, f"Failed to parse file: {str(e)}"
