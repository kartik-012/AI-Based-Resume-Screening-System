# -*- coding: utf-8 -*-
"""
Resume Classification Models Definition & Factory
Implements TF-IDF vectorization and machine learning classifiers:
Linear SVM, Multinomial Naive Bayes, Random Forest, and Logistic Regression.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_vectorizer():
    """Configures TF-IDF vectorizer optimized for resume text vocabulary."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3500,
        sublinear_tf=True,
        min_df=2,
        max_df=0.92
    )

def create_model_pipelines():
    """Builds scikit-learn pipelines for multiple classification algorithms."""
    pipelines = {
        'Linear SVM (Calibrated)': Pipeline([
            ('tfidf', get_vectorizer()),
            ('clf', CalibratedClassifierCV(estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=3))
        ]),
        'Multinomial Naive Bayes': Pipeline([
            ('tfidf', get_vectorizer()),
            ('clf', MultinomialNB(alpha=0.1))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', get_vectorizer()),
            ('clf', RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1))
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', get_vectorizer()),
            ('clf', LogisticRegression(max_iter=1000, C=2.0, random_state=42))
        ])
    }
    return pipelines