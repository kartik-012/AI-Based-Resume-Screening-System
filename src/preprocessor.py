# -*- coding: utf-8 -*-
"""
Text Preprocessing Pipeline for AI-Based Resume Screening & Candidate Analysis
Performs text cleaning, stopword removal, token normalization, and dataset curation.
"""

import os
import re
import pandas as pd
import nltk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'resumes_dataset.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_resumes.csv')
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

# Standard English stopwords set to guarantee zero dependency failures
ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t',
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
    'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if',
    'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most',
    'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d',
    'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d',
    'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what',
    'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why',
    'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve',
    'your', 'yours', 'yourself', 'yourselves'
}

def clean_text(text, remove_stopwords=True):
    """
    Standardizes and cleans text for NLP modeling:
    - Lowercases text
    - Strips URLs, email addresses, phone numbers
    - Normalizes punctuation while keeping technical hyphens/dots (.js, c++, etc.)
    - Removes stopwords and extraneous whitespace
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' ', text)
    # Protect common tech symbols before punctuation stripping
    text = text.replace('c++', 'cpp').replace('c#', 'csharp').replace('.net', 'dotnet')
    text = text.replace('node.js', 'nodejs').replace('react.js', 'reactjs').replace('vue.js', 'vuejs').replace('next.js', 'nextjs')
    
    # Remove special punctuation but keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize and filter stopwords
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
        
    return ' '.join(tokens)

def preprocess_dataset(input_csv=RAW_DATA_PATH, output_csv=PROCESSED_DATA_PATH):
    """Loads raw resume dataset, applies cleaning pipeline, and persists processed data."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Raw dataset not found at {input_csv}. Please run src/data_loader.py first.")
        
    print(f"[INFO] Loading raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print("[INFO] Cleaning resume texts and generating NLP tokens...")
    df['Clean_Resume_Text'] = df['Resume_Text'].apply(lambda x: clean_text(x, remove_stopwords=True))
    df['Word_Count'] = df['Clean_Resume_Text'].apply(lambda x: len(x.split()))
    df['Character_Count'] = df['Clean_Resume_Text'].apply(lambda x: len(x))
    
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"[SUCCESS] Cleaned dataset saved to: {output_csv} ({len(df)} records)")
    return df

if __name__ == '__main__':
    preprocess_dataset()