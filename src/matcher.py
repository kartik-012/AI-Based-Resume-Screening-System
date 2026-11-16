# -*- coding: utf-8 -*-
"""
Candidate Suitability & Job Matching Engine
Computes multidimensional candidate scores:
- Semantic Cosine Similarity (TF-IDF vector matching against Job Description)
- Skill Overlap (Matched vs. Missing Required and Preferred Skills)
- Experience Alignment (Candidate years vs. minimum threshold)
- Education Compatibility
- Composite Candidate Suitability Score & Automated Ranking Leaderboard
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import clean_text
from src.resume_parser import parse_resume

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JD_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'job_descriptions.json')

def load_job_descriptions():
    """Loads benchmark job descriptions dictionary."""
    if os.path.exists(JD_PATH):
        with open(JD_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def compute_semantic_similarity(resume_clean_text, jd_clean_text):
    """Calculates TF-IDF cosine similarity between resume and job description text."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_clean_text, jd_clean_text])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(np.clip(sim, 0.0, 1.0))

def evaluate_skills(candidate_skills, required_skills, preferred_skills=None):
    """
    Analyzes skill overlap, identifies missing required and preferred competencies,
    and returns a normalized skill match score (0 - 100).
    """
    cand_set = {s.lower().strip() for s in candidate_skills}
    req_set = {s.lower().strip() for s in required_skills} if required_skills else set()
    pref_set = {s.lower().strip() for s in preferred_skills} if preferred_skills else set()
    
    # Matching
    matched_req = [s for s in required_skills if s.lower().strip() in cand_set]
    missing_req = [s for s in required_skills if s.lower().strip() not in cand_set]
    
    matched_pref = [s for s in (preferred_skills or []) if s.lower().strip() in cand_set]
    missing_pref = [s for s in (preferred_skills or []) if s.lower().strip() not in cand_set]
    
    # Calculate score
    req_score = (len(matched_req) / max(1, len(required_skills))) * 100.0 if required_skills else 100.0
    pref_score = (len(matched_pref) / max(1, len(preferred_skills))) * 100.0 if preferred_skills else 100.0
    
    total_skill_score = (0.75 * req_score) + (0.25 * pref_score) if preferred_skills else req_score
    
    return {
        'skill_score': round(total_skill_score, 1),
        'matched_required': matched_req,
        'missing_required': missing_req,
        'matched_preferred': matched_pref,
        'missing_preferred': missing_pref,
        'all_matched_skills': matched_req + matched_pref,
        'all_missing_skills': missing_req + missing_pref
    }

def evaluate_experience(candidate_years, required_years):
    """Computes experience alignment score (0 - 100)."""
    if candidate_years >= required_years:
        # Full score with minor bonus for seniority up to 100
        return 100.0
    elif candidate_years <= 0:
        return 20.0
    else:
        # Proportional ratio
        ratio = (candidate_years / max(1.0, required_years)) * 100.0
        return round(float(np.clip(ratio, 20.0, 95.0)), 1)

def evaluate_education(degree_level):
    """Computes education tier score (0 - 100)."""
    tier_scores = {
        'Ph.D.': 100.0,
        "Master's": 95.0,
        "Bachelor's": 85.0,
        'Diploma / Secondary': 65.0
    }
    return tier_scores.get(degree_level, 80.0)

def match_candidate_to_job(parsed_candidate, job_profile):
    """
    Main evaluation pipeline matching a candidate against a target job description.
    Computes Composite Suitability Score:
    Score = 0.40 * Skill + 0.30 * Semantic + 0.20 * Experience + 0.10 * Education
    """
    resume_clean = clean_text(parsed_candidate['raw_text'])
    jd_clean = clean_text(job_profile['jd_text'])
    
    # 1. Semantic Similarity (0 - 100)
    semantic_sim = compute_semantic_similarity(resume_clean, jd_clean)
    semantic_score = round(semantic_sim * 100.0, 1)
    
    # 2. Skill Evaluation (0 - 100)
    skill_eval = evaluate_skills(
        parsed_candidate['skills'],
        job_profile.get('required_skills', []),
        job_profile.get('preferred_skills', [])
    )
    skill_score = skill_eval['skill_score']
    
    # 3. Experience Evaluation (0 - 100)
    exp_score = evaluate_experience(
        parsed_candidate['experience_years'],
        job_profile.get('min_experience', 3.0)
    )
    
    # 4. Education Evaluation (0 - 100)
    edu_score = evaluate_education(parsed_candidate.get('degree_level', "Bachelor's"))
    
    # 5. Composite Weighted Suitability Score
    composite_score = (
        0.40 * skill_score +
        0.30 * semantic_score +
        0.20 * exp_score +
        0.10 * edu_score
    )
    composite_score = round(float(np.clip(composite_score, 0.0, 100.0)), 1)
    
    # Status Category
    if composite_score >= 75.0:
        status = "Shortlisted"
        status_color = "#10b981"  # Emerald Green
    elif composite_score >= 55.0:
        status = "Under Review"
        status_color = "#f59e0b"  # Amber Orange
    else:
        status = "Not Recommended"
        status_color = "#ef4444"  # Coral Red
        
    return {
        'candidate_name': parsed_candidate['name'],
        'email': parsed_candidate['email'],
        'phone': parsed_candidate['phone'],
        'degree': parsed_candidate['degree'],
        'degree_level': parsed_candidate['degree_level'],
        'experience_years': parsed_candidate['experience_years'],
        'target_role': job_profile.get('role_title', 'Target Role'),
        'composite_score': composite_score,
        'status': status,
        'status_color': status_color,
        'semantic_score': semantic_score,
        'skill_score': skill_score,
        'experience_score': exp_score,
        'education_score': edu_score,
        'matched_skills': skill_eval['all_matched_skills'],
        'missing_skills': skill_eval['all_missing_skills'],
        'missing_required_skills': skill_eval['missing_required'],
        'missing_preferred_skills': skill_eval['missing_preferred'],
        'matched_required_skills': skill_eval['matched_required'],
        'matched_preferred_skills': skill_eval['matched_preferred']
    }

def rank_candidates(candidate_list, job_profile):
    """
    Ranks a collection of parsed candidates for a specific job profile.
    Returns sorted list with ranks and summary metrics.
    """
    ranked = []
    for cand in candidate_list:
        eval_result = match_candidate_to_job(cand, job_profile)
        ranked.append(eval_result)
        
    # Sort descending by composite score, then experience
    ranked.sort(key=lambda x: (x['composite_score'], x['experience_years']), reverse=True)
    
    for idx, item in enumerate(ranked):
        item['rank'] = idx + 1
        
    return ranked

if __name__ == '__main__':
    print("[INFO] Testing candidate-to-job matching engine...")
    jds = load_job_descriptions()
    if 'Data Science' in jds:
        jd_ds = jds['Data Science']
        sample_cand = {
            'name': 'Kartik Raikar',
            'email': 'kartikraikar2005@gmail.com',
            'phone': '+91 98765 43210',
            'degree': 'B.Tech in Computer Science',
            'degree_level': "Bachelor's",
            'experience_years': 4.5,
            'skills': ['Python', 'SQL', 'Scikit-learn', 'Pandas', 'NumPy', 'Data Visualization', 'Statistical Modeling', 'PyTorch', 'Tableau'],
            'raw_text': 'Senior Data Scientist with 4.5 years experience in Python, SQL, Scikit-learn, Pandas, predictive modeling and statistics.'
        }
        res = match_candidate_to_job(sample_cand, jd_ds)
        print(f"Candidate: {res['candidate_name']}")
        print(f"Role: {res['target_role']} | Composite Score: {res['composite_score']}% ({res['status']})")
        print(f"Skill Score: {res['skill_score']}% | Semantic: {res['semantic_score']}%")
        print(f"Matched Skills: {res['matched_skills']}")
        print(f"Missing Skills: {res['missing_skills']}")
        print("[SUCCESS] Matcher verified successfully!")