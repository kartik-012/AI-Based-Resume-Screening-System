# -*- coding: utf-8 -*-
"""
Model Training, Benchmarking & Evaluation Pipeline
Orchestrates training across 4 candidate classifiers, computes standard metrics
(Accuracy, Precision, Recall, F1, Cross-Validation), serializes best model and saves CSV metrics.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from src.models.classifier import create_model_pipelines
from src.preprocessor import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_resumes.csv')
METRICS_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'model_comparison.csv')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'best_model_pipeline.joblib')
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'confusion_matrix.npy')
CLASSES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'classes.npy')

def train_and_evaluate_all():
    """Trains all candidate models, evaluates on holdout test set, and determines the champion model."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset missing at {PROCESSED_DATA_PATH}")
        
    print("[INFO] Loading preprocessed dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    X = df['Clean_Resume_Text'].fillna('')
    y = df['Category']
    
    print(f"[INFO] Dataset size: {len(df)} samples across {y.nunique()} categories.")
    
    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    pipelines = create_model_pipelines()
    results = []
    best_f1 = -1.0
    best_model_name = None
    best_pipeline = None
    best_cm = None
    
    print("\n" + "="*70)
    print("STARTING MODEL BENCHMARKING (Linear SVM, Naive Bayes, Random Forest, Logistic Regression)")
    print("="*70)
    
    for name, pipeline in pipelines.items():
        print(f"\n[TRAINING] Fitting {name}...")
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 5-fold cross validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"--> {name} Results:")
        print(f"    Test Accuracy : {acc:.4f}")
        print(f"    Macro F1-Score: {f1_macro:.4f}")
        print(f"    Weighted F1   : {f1_weighted:.4f}")
        print(f"    5-Fold CV Acc : {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Macro_Precision': round(prec_macro, 4),
            'Macro_Recall': round(rec_macro, 4),
            'Macro_F1': round(f1_macro, 4),
            'Weighted_F1': round(f1_weighted, 4),
            'CV_Mean_Accuracy': round(cv_mean, 4),
            'CV_Std': round(cv_std, 4)
        })
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_pipeline = pipeline
            best_cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
            
    # Save benchmark table
    metrics_df = pd.DataFrame(results)
    metrics_df.sort_values(by='Macro_F1', ascending=False, inplace=True)
    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Model comparison metrics saved to: {METRICS_OUTPUT_PATH}")
    
    # Save best model and metadata
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    np.save(CONFUSION_MATRIX_PATH, best_cm)
    np.save(CLASSES_PATH, np.array(sorted(y.unique())))
    print(f"[CHAMPION MODEL] Selected: {best_model_name} (Macro F1 = {best_f1:.4f})")
    print(f"[SUCCESS] Best pipeline serialized to: {BEST_MODEL_PATH}")
    
    return metrics_df, best_model_name

class ResumeClassifierInference:
    """Inference engine to classify single or batch resumes."""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(f"Model file missing. Train first using train_and_evaluate_all().")
        self.pipeline = joblib.load(BEST_MODEL_PATH)
        self.classes = np.load(CLASSES_PATH, allow_pickle=True)
        
    def predict(self, raw_text):
        """Returns top predicted category and probability distribution."""
        clean = clean_text(raw_text)
        probs = self.pipeline.predict_proba([clean])[0]
        top_idx = np.argmax(probs)
        predicted_category = self.classes[top_idx]
        confidence = float(probs[top_idx])
        
        # Sorted distribution
        distribution = []
        for cat, prob in sorted(zip(self.classes, probs), key=lambda x: x[1], reverse=True):
            distribution.append({'category': cat, 'probability': round(float(prob), 4)})
            
        return {
            'predicted_category': predicted_category,
            'confidence': round(confidence, 4),
            'top_3': distribution[:3],
            'full_distribution': distribution
        }

if __name__ == '__main__':
    train_and_evaluate_all()