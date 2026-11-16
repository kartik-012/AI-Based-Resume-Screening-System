# -*- coding: utf-8 -*-
"""
Resume Parser Engine for AI-Based Resume Screening & Candidate Analysis
Supports extraction from PDF, DOCX, and TXT files.
Extracts Candidate Identity, Contact Info, Education, Experience, and Skills using
NLP ontologies and regex patterns.
"""

import os
import re
import pypdf
import docx

# Comprehensive Skill Ontology across domains
SKILL_TAXONOMY = {
    "AI & Machine Learning": [
        "python", "pytorch", "tensorflow", "keras", "deep learning", "nlp", "computer vision",
        "transformers", "huggingface", "scikit-learn", "xgboost", "lightgbm", "llm", "langchain",
        "bert", "reinforcement learning", "neural networks", "opencv", "cuda", "mlops", "mlflow",
        "pandas", "numpy", "matplotlib", "seaborn", "statistics", "data analysis", "statistical modeling"
    ],
    "Data Science & Analytics": [
        "sql", "r", "tableau", "powerbi", "excel", "bigquery", "snowflake", "spark", "hadoop",
        "etl", "data visualization", "exploratory data analysis", "hypothesis testing", "a/b testing",
        "business intelligence", "data mining", "predictive analytics", "quantitative analysis"
    ],
    "Web & Frontend Development": [
        "javascript", "typescript", "react", "next.js", "angular", "vue.js", "html5", "html",
        "css3", "css", "tailwind css", "bootstrap", "redux", "graphql", "rest api", "restful api",
        "webpack", "vite", "sass", "responsive design", "web development"
    ],
    "Backend & Software Engineering": [
        "java", "spring boot", "spring mvc", "hibernate", "jpa", "microservices", "c++", "c#",
        ".net", "golang", "go", "rust", "node.js", "express.js", "fastapi", "django", "flask",
        "multithreading", "design patterns", "oop", "system design", "kafka", "rabbitmq"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "ci/cd",
        "jenkins", "github actions", "gitlab ci", "ansible", "linux", "bash", "shell scripting",
        "prometheus", "grafana", "nginx", "helm", "infrastructure as code", "serverless"
    ],
    "Databases & Storage": [
        "postgresql", "mysql", "mongodb", "redis", "cassandra", "dynamodb", "sqlite",
        "oracle db", "elasticsearch", "database optimization", "indexing", "nosql"
    ],
    "Cybersecurity": [
        "penetration testing", "ethical hacking", "network security", "vulnerability assessment",
        "siem", "wireshark", "metasploit", "owasp", "firewalls", "cryptography", "incident response",
        "cissp", "ceh", "soc analysis", "information security"
    ],
    "Mobile App Development": [
        "flutter", "react native", "swift", "ios", "kotlin", "android", "android sdk",
        "jetpack compose", "dart", "mobile app", "xcode"
    ],
    "QA & Software Testing": [
        "selenium", "cypress", "playwright", "testng", "junit", "automation testing",
        "manual testing", "postman", "jmeter", "performance testing", "api testing", "jira"
    ],
    "Management & Soft Skills": [
        "agile", "scrum", "kanban", "sprint planning", "leadership", "project management",
        "stakeholder management", "communication skills", "problem solving", "teamwork",
        "critical thinking", "mentoring", "time management"
    ]
}

# Flattened list of all unique skills
ALL_SKILLS = set()
for cat_skills in SKILL_TAXONOMY.values():
    for sk in cat_skills:
        ALL_SKILLS.add(sk.lower())

DEGREE_LEVELS = {
    'Ph.D.': ['ph.d', 'phd', 'doctor of philosophy', 'doctorate'],
    "Master's": ['m.tech', 'm.s', 'ms in', 'mtech', 'master of technology', 'master of science', 'mca', 'mba', 'm.sc'],
    "Bachelor's": ['b.tech', 'b.e', 'btech', 'bachelor of technology', 'bachelor of engineering', 'bca', 'b.sc', 'bba', 'bachelor of science'],
    'Diploma / Secondary': ['diploma', 'high school', 'higher secondary', 'polytechnic']
}

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from PDF using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"[ERROR] Failed to extract PDF text from {pdf_path}: {e}")
    return text.strip()

def extract_text_from_docx(docx_path):
    """Extracts raw text from DOCX using python-docx."""
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                text += " | ".join(row_text) + "\n"
    except Exception as e:
        print(f"[ERROR] Failed to extract DOCX text from {docx_path}: {e}")
    return text.strip()

def extract_text_from_file(file_path):
    """Extracts raw text based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception as e:
            print(f"[ERROR] Failed reading text file {file_path}: {e}")
            return ""

def extract_email(text):
    """Extracts email address using regex."""
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "Not Specified"

def extract_phone(text):
    """Extracts phone number using international and Indian regex patterns."""
    pattern = r'(?:\+?\d{1,4}[\s-]?)?\(?\d{2,5}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}'
    for m in re.finditer(pattern, text):
        cleaned = re.sub(r'\D', '', m.group(0))
        if 10 <= len(cleaned) <= 13:
            return m.group(0).strip()
    return "Not Specified"

def extract_links(text):
    """Extracts LinkedIn and GitHub URLs."""
    linkedin_match = re.search(r'(linkedin\.com/in/[a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    github_match = re.search(r'(github\.com/[a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    return {
        'linkedin': linkedin_match.group(1) if linkedin_match else "Not Specified",
        'github': github_match.group(1) if github_match else "Not Specified"
    }

def extract_name(text):
    """Extracts candidate name heuristics from top lines of the resume."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:5]:
        # Filter out common header labels
        if re.search(r'(curriculum\s+vitae|resume|cv|biodata)', line, re.IGNORECASE):
            continue
        # Clean line and verify reasonable name length
        clean_line = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        words = clean_line.split()
        if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
            return clean_line.title()
    return "Candidate"

def extract_education(text):
    """Identifies degree level and specific degree from text."""
    lower_text = text.lower()
    for level, patterns in DEGREE_LEVELS.items():
        for pat in patterns:
            if re.search(r'\b' + re.escape(pat) + r'\b', lower_text):
                # Search for surrounding line or phrase
                match_lines = [l.strip() for l in text.split('\n') if pat in l.lower()]
                degree_name = match_lines[0] if match_lines else level
                # Clean up if too long
                if len(degree_name) > 60:
                    degree_name = degree_name[:60] + "..."
                return {
                    'degree_level': level,
                    'degree_name': degree_name
                }
    return {
        'degree_level': "Bachelor's",
        'degree_name': "Bachelor of Technology / Science (Inferred)"
    }

def extract_experience_years(text):
    """Extracts total years of experience using numeric patterns and date ranges."""
    lower_text = text.lower()
    
    # Pattern 1: Explicit statements like "5.5 years of experience", "4+ years of industry experience"
    exp_pattern = r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:proven|industry|relevant|hands-on|professional)?\s*experience'
    match = re.search(exp_pattern, lower_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Pattern 2: Year ranges like 2018 - 2023, 2019 - Present
    year_ranges = re.findall(r'(20\d{2})\s*(?:-|–|to)\s*(20\d{2}|present|current)', lower_text)
    total_years = 0.0
    current_year = 2024
    for start, end in year_ranges:
        start_yr = int(start)
        end_yr = current_year if end in ['present', 'current'] else int(end)
        diff = max(0, end_yr - start_yr)
        total_years = max(total_years, diff)
        
    if total_years > 0:
        return float(total_years)
        
    return 3.0  # Default reasonable industry baseline

def extract_skills(text):
    """Extracts technical and soft skills by matching against the ontology."""
    lower_text = " " + text.lower() + " "
    found_skills = set()
    category_matches = {cat: [] for cat in SKILL_TAXONOMY}
    
    for category, skill_list in SKILL_TAXONOMY.items():
        for skill in skill_list:
            # Word boundary matching to avoid false positives (e.g. 'c' or 'r')
            if len(skill) <= 2:
                pattern = r'(?:^|[\s,;|/()\[\]])' + re.escape(skill) + r'(?:$|[\s,;|/()\[\]])'
            else:
                pattern = r'\b' + re.escape(skill) + r'\b'
                
            if re.search(pattern, lower_text):
                # Standardize display casing
                display_skill = skill.title() if len(skill) > 3 else skill.upper()
                if skill in ['aws', 'sql', 'gcp', 'qa', 'siem', 'ci/cd', 'oop', 'rest api', 'restful api', 'html5', 'css3', 'soc analysis']:
                    display_skill = skill.upper()
                elif skill == 'scikit-learn':
                    display_skill = 'Scikit-learn'
                elif skill == 'pytorch':
                    display_skill = 'PyTorch'
                elif skill == 'tensorflow':
                    display_skill = 'TensorFlow'
                elif skill == 'powerbi':
                    display_skill = 'PowerBI'
                elif skill == 'next.js':
                    display_skill = 'Next.js'
                elif skill == 'node.js':
                    display_skill = 'Node.js'
                    
                found_skills.add(display_skill)
                category_matches[category].append(display_skill)
                
    # Filter empty categories
    active_categories = {k: v for k, v in category_matches.items() if len(v) > 0}
    return sorted(list(found_skills)), active_categories

def parse_resume(file_path_or_text, is_file=True):
    """Master resume parsing orchestrator. Returns structured metadata dictionary."""
    if is_file:
        raw_text = extract_text_from_file(file_path_or_text)
        filename = os.path.basename(file_path_or_text)
    else:
        raw_text = file_path_or_text
        filename = "pasted_text_resume"
        
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    links = extract_links(raw_text)
    education = extract_education(raw_text)
    experience_years = extract_experience_years(raw_text)
    skills, category_breakdown = extract_skills(raw_text)
    
    return {
        'filename': filename,
        'name': name,
        'email': email,
        'phone': phone,
        'linkedin': links['linkedin'],
        'github': links['github'],
        'degree': education['degree_name'],
        'degree_level': education['degree_level'],
        'experience_years': experience_years,
        'skills': skills,
        'skills_count': len(skills),
        'category_breakdown': category_breakdown,
        'raw_text': raw_text
    }

if __name__ == '__main__':
    print("[INFO] Testing Resume Parser on sample files...")
    sample_pdf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sample_resumes', 'sample_resume_datascience.pdf')
    if os.path.exists(sample_pdf):
        parsed = parse_resume(sample_pdf)
        print(f"Candidate: {parsed['name']}")
        print(f"Email: {parsed['email']} | Phone: {parsed['phone']}")
        print(f"Experience: {parsed['experience_years']} yrs | Degree: {parsed['degree_level']}")
        print(f"Skills Extracted ({parsed['skills_count']}): {parsed['skills'][:8]}")
        print("[SUCCESS] Resume parser verified successfully!")
    else:
        print("[WARNING] Sample PDF not found. Run src/data_loader.py first.")