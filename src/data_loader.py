# -*- coding: utf-8 -*-
"""
Data Loader & Dataset Generator for AI-Based Resume Screening & Candidate Analysis
Constructs a comprehensive multi-domain resume dataset, standard job descriptions,
and generates sample candidate resumes (PDF, DOCX, TXT) for live application testing.
"""

import os
import json
import random
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import docx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
SAMPLE_RESUMES_DIR = os.path.join(BASE_DIR, 'data', 'sample_resumes')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_RESUMES_DIR, exist_ok=True)

CATEGORIES = [
    'Data Science',
    'Machine Learning',
    'Web Development',
    'Java Development',
    'DevOps & Cloud',
    'Cybersecurity',
    'Mobile App Development',
    'Database Engineering',
    'Software Testing / QA',
    'HR & Talent Acquisition',
    'Business Analytics',
    'Product Management',
    'UI/UX Design',
    'Blockchain Engineering',
    'Network Engineering'
]

SKILL_ONTOLOGY = {
    'Data Science': [
        'Python', 'R', 'Pandas', 'NumPy', 'Scikit-learn', 'SQL', 'Tableau', 'PowerBI',
        'Data Visualization', 'Exploratory Data Analysis', 'Statistical Modeling',
        'Hypothesis Testing', 'Matplotlib', 'Seaborn', 'A/B Testing', 'BigQuery'
    ],
    'Machine Learning': [
        'Python', 'PyTorch', 'TensorFlow', 'Keras', 'Deep Learning', 'NLP', 'Computer Vision',
        'Transformers', 'HuggingFace', 'Scikit-learn', 'XGBoost', 'LightGBM', 'Model Deployment',
        'MLflow', 'Docker', 'REST APIs', 'CUDA'
    ],
    'Web Development': [
        'JavaScript', 'TypeScript', 'React', 'Next.js', 'Node.js', 'Express.js', 'HTML5',
        'CSS3', 'Tailwind CSS', 'Redux', 'RESTful APIs', 'GraphQL', 'MongoDB', 'PostgreSQL',
        'Webpack', 'Git', 'Responsive Design'
    ],
    'Java Development': [
        'Java', 'Spring Boot', 'Spring MVC', 'Hibernate', 'JPA', 'Microservices', 'REST APIs',
        'Maven', 'Gradle', 'JUnit', 'Mockito', 'MySQL', 'Oracle DB', 'Kafka', 'Docker',
        'Multithreading', 'Design Patterns'
    ],
    'DevOps & Cloud': [
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'Jenkins',
        'GitHub Actions', 'Linux', 'Bash Scripting', 'Ansible', 'Prometheus', 'Grafana',
        'Helm', 'Nginx', 'Infrastructure as Code'
    ],
    'Cybersecurity': [
        'Penetration Testing', 'Ethical Hacking', 'Network Security', 'Vulnerability Assessment',
        'SIEM', 'Wireshark', 'Metasploit', 'OWASP Top 10', 'Firewalls', 'Cryptography',
        'Incident Response', 'CISSP', 'NIST Framework', 'SOC Analysis'
    ],
    'Mobile App Development': [
        'Flutter', 'Dart', 'React Native', 'Swift', 'iOS SDK', 'Kotlin', 'Android SDK',
        'Jetpack Compose', 'Mobile UI/UX', 'SQLite', 'Firebase', 'REST APIs', 'App Store Deployment'
    ],
    'Database Engineering': [
        'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Cassandra', 'Database Optimization',
        'Query Tuning', 'ETL Pipelines', 'Data Warehousing', 'Snowflake', 'DynamoDB',
        'Replication', 'Database Indexing'
    ],
    'Software Testing / QA': [
        'Selenium', 'Cypress', 'Playwright', 'TestNG', 'JUnit', 'Automation Testing',
        'Manual Testing', 'Postman', 'API Testing', 'Performance Testing', 'JMeter',
        'Bug Tracking', 'Jira', 'Regression Testing'
    ],
    'HR & Talent Acquisition': [
        'Technical Recruiting', 'Talent Sourcing', 'Interviewing', 'Candidate Screening',
        'HR Policies', 'Employee Engagement', 'Onboarding', 'Applicant Tracking Systems (ATS)',
        'Workday', 'Performance Management', 'Labor Laws'
    ],
    'Business Analytics': [
        'SQL', 'Excel', 'Tableau', 'PowerBI', 'Business Intelligence', 'Data Storytelling',
        'KPI Dashboards', 'Financial Modeling', 'Market Analysis', 'Requirements Gathering',
        'Process Optimization', 'Python'
    ],
    'Product Management': [
        'Product Strategy', 'Roadmapping', 'Agile / Scrum', 'User Stories', 'Sprint Planning',
        'Market Research', 'Competitor Analysis', 'Stakeholder Management', 'Jira',
        'Wireframing', 'Product Analytics', 'User Experience'
    ],
    'UI/UX Design': [
        'Figma', 'Adobe XD', 'Sketch', 'Wireframing', 'Prototyping', 'User Research',
        'Usability Testing', 'Information Architecture', 'Design Systems', 'Visual Design',
        'Interaction Design', 'CSS Basics'
    ],
    'Blockchain Engineering': [
        'Solidity', 'Ethereum', 'Smart Contracts', 'Web3.js', 'Ethers.js', 'Hardhat',
        'Truffle', 'Cryptography', 'DeFi', 'NFTs', 'Consensus Algorithms', 'Rust'
    ],
    'Network Engineering': [
        'Cisco Networking', 'CCNA', 'Routing & Switching', 'TCP/IP', 'BGP', 'OSPF',
        'VLANs', 'Network Troubleshooting', 'Firewalls', 'VPNs', 'VoIP', 'Wireshark', 'DNS/DHCP'
    ]
}

DEGREES = [
    'B.Tech in Computer Science', 'B.E. in Information Technology', 'M.Tech in Artificial Intelligence',
    'M.S. in Computer Science', 'B.Sc in Computer Science', 'Master of Computer Applications (MCA)',
    'Bachelor of Computer Applications (BCA)', 'MBA in Operations & HR', 'B.Tech in Electronics',
    'M.S. in Data Science', 'B.E. in Telecommunications', 'BBA in Information Systems'
]

COMPANIES = [
    'TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant', 'Amazon', 'Microsoft', 'Google',
    'Deloitte', 'Capgemini', 'IBM', 'HCL Tech', 'Oracle', 'Cisco', 'Razorpay', 'Swiggy'
]

UNIVERSITIES = [
    'Indian Institute of Technology (IIT)', 'National Institute of Technology (NIT)',
    'Birla Institute of Technology and Science (BITS)', 'Delhi Technological University (DTU)',
    'Vellore Institute of Technology (VIT)', 'Manipal Institute of Technology',
    'Pune Institute of Computer Technology', 'Anna University', 'Jadavpur University'
]

FIRST_NAMES = ['Aarav', 'Ananya', 'Rohan', 'Priya', 'Aditya', 'Sneha', 'Vikram', 'Neha', 'Arjun', 'Pooja', 'Rahul', 'Divya', 'Siddharth', 'Kavya', 'Karan', 'Meera', 'Varun', 'Shreya', 'Amit', 'Rhea']
LAST_NAMES = ['Sharma', 'Verma', 'Patel', 'Rao', 'Nair', 'Iyer', 'Gupta', 'Singh', 'Deshmukh', 'Chopra', 'Reddy', 'Mehta', 'Kulkarni', 'Joshi', 'Bose', 'Menon', 'Bhat', 'Das', 'Chatterjee', 'Raikar']

def generate_resume_text(candidate_name, category, exp_years, degree, university, candidate_skills):
    email = candidate_name.lower().replace(' ', '.') + '@email.com'
    phone = '+91 ' + str(random.randint(70000, 99999)) + ' ' + str(random.randint(10000, 99999))
    linkedin = 'linkedin.com/in/' + candidate_name.lower().replace(' ', '-')
    github = 'github.com/' + candidate_name.lower().replace(' ', '')
    company_1 = random.choice(COMPANIES)
    company_2 = random.choice([c for c in COMPANIES if c != company_1])
    
    selected_skills = random.sample(candidate_skills, min(len(candidate_skills), random.randint(7, 12)))
    general_skills = ['Git', 'Agile / Scrum', 'Problem Solving', 'Team Collaboration', 'Communication Skills', 'Linux']
    selected_skills.extend(random.sample(general_skills, 3))
    skills_str = ', '.join(selected_skills)
    
    summary = f'Results-driven {category} professional with {exp_years} years of proven industry experience in developing scalable solutions, optimizing engineering workflows, and driving technical excellence. Adept at full lifecycle execution, continuous improvement, and cross-functional leadership.'
    
    lines = [
        'CURRICULUM VITAE',
        candidate_name.upper(),
        f'Email: {email} | Phone: {phone} | LinkedIn: {linkedin} | GitHub: {github}',
        '',
        'PROFESSIONAL SUMMARY',
        summary,
        '',
        'CORE COMPETENCIES & TECHNICAL SKILLS',
        f'Technical Skills: {skills_str}',
        'Methodologies: Agile, Scrum, Kanban, Sprint Planning, Code Reviews, Continuous Integration',
        '',
        'PROFESSIONAL EXPERIENCE',
        f'Senior {category} Specialist | {company_1} | 2021 - Present',
        '- Spearheaded core architectural implementations and deployed production services utilized by 100K+ active users.',
        '- Collaborated across multi-functional squads to drive technical roadmaps, achieving 35% reduction in latency and 25% throughput gains.',
        '- Mentored junior engineers on best coding standards, clean architecture, and automated test-driven development.',
        f'- Leveraged {selected_skills[0]}, {selected_skills[1]}, and {selected_skills[2]} to streamline enterprise delivery pipelines.',
        '',
        f'{category} Associate | {company_2} | 2018 - 2021',
        '- Designed and maintained high-availability systems supporting critical business workflows.',
        '- Implemented robust monitoring, automated alerting, and reduced mean time to resolution (MTTR) by 40%.',
        '- Documented technical specifications, API contracts, and user manuals for cross-departmental alignment.',
        f'- Utilized {selected_skills[3]} and {selected_skills[4]} to automate legacy data processing modules.',
        '',
        'EDUCATION & CREDENTIALS',
        f'{degree} | {university} | Graduated with Distinction',
        'Relevant Coursework: Data Structures, Algorithms, Software Engineering, Database Systems, Computer Networks.',
        '',
        'KEY PROJECTS & HIGHLIGHTS',
        f'- Automated Enterprise Pipeline: Built an end-to-end framework leveraging {selected_skills[0]} and {selected_skills[1]} yielding 50% operational cost efficiency.',
        '- High-Performance Microservice: Engineered resilient APIs with sub-50ms response times under peak production loads.',
        '',
        'CERTIFICATIONS',
        f'- Certified Professional in {category}',
        '- Advanced Agile Practitioner Certification'
    ]
    text = '\n'.join(lines)
    return text.strip(), email, phone, skills_str

def build_dataset(num_samples_per_category=80):
    random.seed(42)
    data = []
    sample_id = 1001
    
    for category in CATEGORIES:
        cat_skills = SKILL_ONTOLOGY[category]
        for i in range(num_samples_per_category):
            name = f'{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}'
            exp = round(random.uniform(1.0, 15.0), 1)
            degree = random.choice(DEGREES)
            univ = random.choice(UNIVERSITIES)
            
            resume_text, email, phone, skills_str = generate_resume_text(
                name, category, exp, degree, univ, cat_skills
            )
            
            data.append({
                'Candidate_ID': f'CAND-{sample_id}',
                'Full_Name': name,
                'Category': category,
                'Experience_Years': exp,
                'Degree': degree,
                'University': univ,
                'Skills': skills_str,
                'Email': email,
                'Phone': phone,
                'Resume_Text': resume_text
            })
            sample_id += 1
            
    df = pd.DataFrame(data)
    csv_path = os.path.join(RAW_DATA_DIR, 'resumes_dataset.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'[SUCCESS] Generated {len(df)} resumes across {len(CATEGORIES)} categories saved to {csv_path}')
    return df

def build_job_descriptions():
    jds = {
        'Data Science': {
            'role_title': 'Senior Data Scientist',
            'department': 'Artificial Intelligence & Analytics',
            'min_experience': 4.0,
            'required_education': "Master's or Bachelor's in Computer Science, Data Science, or related field",
            'required_skills': ['Python', 'SQL', 'Scikit-learn', 'Pandas', 'NumPy', 'Data Visualization', 'Statistical Modeling'],
            'preferred_skills': ['PyTorch', 'TensorFlow', 'Tableau', 'BigQuery', 'MLflow', 'Docker'],
            'jd_text': 'We are seeking an experienced Senior Data Scientist to architect predictive analytics and machine learning solutions. You will extract actionable business insights from vast data repositories, design statistical models, and collaborate with engineering teams to deploy models into production. Required skills include Python, SQL, Scikit-learn, Pandas, statistical modeling, hypothesis testing, and data visualization. Hands-on experience with deep learning, PyTorch, or Docker is a strong plus.'
        },
        'Machine Learning': {
            'role_title': 'Machine Learning Engineer',
            'department': 'Core AI Engineering',
            'min_experience': 3.0,
            'required_education': 'B.Tech/M.Tech/M.S. in Computer Science, AI, or equivalent',
            'required_skills': ['Python', 'PyTorch', 'TensorFlow', 'Deep Learning', 'NLP', 'Docker', 'REST APIs'],
            'preferred_skills': ['Transformers', 'HuggingFace', 'MLflow', 'Kubernetes', 'CUDA', 'Computer Vision'],
            'jd_text': 'Join our AI team as a Machine Learning Engineer to design, train, and optimize state-of-the-art deep learning models for production scale. You will work on natural language processing, computer vision, and transformer architectures. Strong programming in Python, PyTorch or TensorFlow, containerization with Docker, and building resilient model-serving APIs are mandatory.'
        },
        'Web Development': {
            'role_title': 'Lead Full Stack Web Developer',
            'department': 'Web Engineering',
            'min_experience': 5.0,
            'required_education': "Bachelor's in Computer Science or Software Engineering",
            'required_skills': ['JavaScript', 'TypeScript', 'React', 'Node.js', 'Next.js', 'PostgreSQL', 'HTML5', 'CSS3'],
            'preferred_skills': ['GraphQL', 'Tailwind CSS', 'Docker', 'Redis', 'Redux', 'AWS'],
            'jd_text': 'Looking for a passionate Lead Full Stack Web Developer proficient in modern frontend and backend architectures. You will lead the development of enterprise web applications using React, Next.js, Node.js, TypeScript, and relational databases. Must possess deep expertise in responsive web design, RESTful APIs, state management, and high-performance server-side rendering.'
        },
        'DevOps & Cloud': {
            'role_title': 'Senior DevOps & Cloud Architect',
            'department': 'Infrastructure & Reliability',
            'min_experience': 4.0,
            'required_education': "Bachelor's in Computer Science, IT, or equivalent experience",
            'required_skills': ['AWS', 'Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'Jenkins', 'Linux', 'Bash Scripting'],
            'preferred_skills': ['Ansible', 'Prometheus', 'Grafana', 'Helm', 'Azure', 'GCP'],
            'jd_text': 'Seeking a Senior DevOps & Cloud Architect to automate and secure our cloud infrastructure. You will manage container orchestration with Kubernetes, implement Infrastructure as Code via Terraform, maintain zero-downtime CI/CD deployment pipelines, and optimize cloud costs across AWS. Strong Linux systems administration and bash scripting are essential.'
        },
        'Java Development': {
            'role_title': 'Principal Java Microservices Engineer',
            'department': 'Backend Platform Engineering',
            'min_experience': 5.0,
            'required_education': 'B.Tech / M.Tech in Computer Science or related discipline',
            'required_skills': ['Java', 'Spring Boot', 'Microservices', 'Hibernate', 'REST APIs', 'MySQL', 'Kafka'],
            'preferred_skills': ['Docker', 'Kubernetes', 'JUnit', 'Multithreading', 'Design Patterns'],
            'jd_text': 'We are looking for a Principal Java Developer to design scalable distributed microservices handling millions of daily transactions. Must possess deep mastery over Java, Spring Boot, Hibernate, Kafka event streaming, and relational database systems. Experience with multi-threading, concurrency, and cloud-native deployments is strongly required.'
        },
        'Cybersecurity': {
            'role_title': 'Information Security & SOC Analyst',
            'department': 'Information Security & Governance',
            'min_experience': 3.0,
            'required_education': 'B.Tech in CS/Cybersecurity or recognized industry certifications (CISSP, CEH)',
            'required_skills': ['Network Security', 'Penetration Testing', 'SIEM', 'Wireshark', 'Vulnerability Assessment', 'Firewalls'],
            'preferred_skills': ['Metasploit', 'Incident Response', 'OWASP Top 10', 'SOC Analysis', 'Cryptography'],
            'jd_text': 'Seeking an Information Security Analyst to safeguard enterprise networks and digital assets. Responsible for conducting penetration tests, managing SIEM alerts, vulnerability assessments, and threat intelligence analysis. Strong knowledge of firewalls, Wireshark, OWASP security principles, and incident response frameworks required.'
        }
    }
    
    jd_path = os.path.join(RAW_DATA_DIR, 'job_descriptions.json')
    with open(jd_path, 'w', encoding='utf-8') as f:
        json.dump(jds, f, indent=4)
    print(f'[SUCCESS] Standard Job Descriptions saved to {jd_path}')
    return jds

def create_sample_pdf_resume(filepath, name, role, exp, email, phone, skills, degree):
    doc = SimpleDocTemplate(filepath, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('NameTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#1a365d'), spaceAfter=2)
    contact_style = ParagraphStyle('ContactInfo', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#4a5568'), spaceAfter=10)
    section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#0f766e'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, textColor=colors.HexColor('#2d3748'))
    
    story = []
    story.append(Paragraph(name, title_style))
    story.append(Paragraph(f"{email} | {phone} | linkedin.com/in/{name.lower().replace(' ', '-')} | Portfolio: {name.lower().replace(' ', '')}.dev", contact_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0f766e'), spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("PROFESSIONAL PROFILE", section_style))
    story.append(Paragraph(f"Accomplished {role} with {exp} years of technical expertise driving product engineering, scalable software design, and high-impact machine intelligence systems. Proven track record of cross-functional delivery in fast-paced enterprise environments.", body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("CORE TECHNICAL COMPETENCIES", section_style))
    story.append(Paragraph(f"<b>Skills & Technologies:</b> {', '.join(skills)}", body_style))
    story.append(Paragraph("<b>Development Methodologies:</b> Agile Scrum, CI/CD Pipelines, Test-Driven Development, System Design", body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
    story.append(Paragraph(f"<b>Senior {role}</b> — TechSolutions Global (2020 - Present)", body_style))
    story.append(Paragraph("• Led architectural modernization initiatives, accelerating query throughput by 42%.", body_style))
    story.append(Paragraph(f"• Deployed high-throughput services with {skills[0]} and {skills[1]}, maintaining 99.98% service uptime.", body_style))
    story.append(Paragraph("• Supervised a team of 6 engineers, standardizing continuous integration and static code analysis.", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>{role} Engineer</b> — Innovatech Corp (2017 - 2020)", body_style))
    story.append(Paragraph(f"• Engineered modular APIs and backend microservices using {skills[2]} and SQL.", body_style))
    story.append(Paragraph("• Enhanced deployment speed by 35% through containerized Docker delivery.", body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("EDUCATION", section_style))
    story.append(Paragraph(f"<b>{degree}</b> — National Institute of Technology (NIT) | First Class with Distinction", body_style))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("HONORS & CERTIFICATIONS", section_style))
    story.append(Paragraph(f"• Certified Professional Practitioner in {role} Systems", body_style))
    story.append(Paragraph("• Outstanding Innovation Award — Annual Corporate Hackathon Winner", body_style))
    
    doc.build(story)

def create_sample_docx_resume(filepath, name, role, exp, email, phone, skills, degree):
    doc = docx.Document()
    doc.add_heading(name, level=0)
    doc.add_paragraph(f"{email} | {phone} | LinkedIn: linkedin.com/in/{name.lower().replace(' ', '-')}")
    
    doc.add_heading('Executive Summary', level=1)
    doc.add_paragraph(f"Seasoned {role} with {exp} years of deep technical hands-on experience designing robust architectures, leading engineering workflows, and solving mission-critical challenges.")
    
    doc.add_heading('Technical Skills', level=1)
    doc.add_paragraph(', '.join(skills))
    
    doc.add_heading('Experience', level=1)
    doc.add_paragraph(f"Senior {role} | Apex Solutions (2021 - Present)", style='List Bullet')
    doc.add_paragraph(f"Developed scalable microservices and data pipelines utilizing {skills[0]}, {skills[1]}, and {skills[2]}.", style='List Bullet')
    doc.add_paragraph("Automated testing and continuous delivery workflows, improving team velocity by 30%.", style='List Bullet')
    
    doc.add_heading('Education', level=1)
    doc.add_paragraph(f"{degree} - Delhi Technological University (DTU)")
    
    doc.save(filepath)

def generate_sample_resumes():
    samples = [
        {
            'filename': 'sample_resume_datascience.pdf',
            'type': 'pdf',
            'name': 'Kartik Raikar',
            'role': 'Data Science',
            'exp': 4.5,
            'email': 'kartikraikar2005@gmail.com',
            'phone': '+91 98765 43210',
            'skills': ['Python', 'SQL', 'Scikit-learn', 'Pandas', 'NumPy', 'Data Visualization', 'Statistical Modeling', 'PyTorch', 'Tableau'],
            'degree': 'B.Tech in Computer Science & Engineering'
        },
        {
            'filename': 'sample_resume_machinelearning.pdf',
            'type': 'pdf',
            'name': 'Aarav Sharma',
            'role': 'Machine Learning',
            'exp': 3.5,
            'email': 'aarav.sharma@techmail.com',
            'phone': '+91 98111 22334',
            'skills': ['Python', 'PyTorch', 'TensorFlow', 'Deep Learning', 'NLP', 'Docker', 'Transformers', 'HuggingFace', 'REST APIs'],
            'degree': 'M.Tech in Artificial Intelligence'
        },
        {
            'filename': 'sample_resume_webdev.docx',
            'type': 'docx',
            'name': 'Priya Iyer',
            'role': 'Web Development',
            'exp': 5.0,
            'email': 'priya.iyer@webdevpro.io',
            'phone': '+91 97222 33445',
            'skills': ['JavaScript', 'TypeScript', 'React', 'Next.js', 'Node.js', 'PostgreSQL', 'Tailwind CSS', 'Redux', 'Git'],
            'degree': 'B.E. in Information Technology'
        },
        {
            'filename': 'sample_resume_devops.docx',
            'type': 'docx',
            'name': 'Rohan Deshmukh',
            'role': 'DevOps & Cloud',
            'exp': 6.0,
            'email': 'rohan.deshmukh@cloudcloud.com',
            'phone': '+91 96333 44556',
            'skills': ['AWS', 'Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'Jenkins', 'Linux', 'Bash Scripting', 'Ansible', 'Prometheus'],
            'degree': 'B.Tech in Computer Science'
        },
        {
            'filename': 'sample_resume_cybersecurity.txt',
            'type': 'txt',
            'name': 'Vikram Gupta',
            'role': 'Cybersecurity',
            'exp': 4.0,
            'email': 'vikram.gupta@cybersec.org',
            'phone': '+91 95444 55667',
            'skills': ['Network Security', 'Penetration Testing', 'SIEM', 'Wireshark', 'Vulnerability Assessment', 'Firewalls', 'Ethical Hacking', 'OWASP Top 10'],
            'degree': 'B.Tech in Information Security'
        }
    ]
    
    for s in samples:
        path = os.path.join(SAMPLE_RESUMES_DIR, s['filename'])
        if s['type'] == 'pdf':
            create_sample_pdf_resume(path, s['name'], s['role'], s['exp'], s['email'], s['phone'], s['skills'], s['degree'])
        elif s['type'] == 'docx':
            create_sample_docx_resume(path, s['name'], s['role'], s['exp'], s['email'], s['phone'], s['skills'], s['degree'])
        elif s['type'] == 'txt':
            resume_text, _, _, _ = generate_resume_text(s['name'], s['role'], s['exp'], s['degree'], 'BITS Pilani', s['skills'])
            with open(path, 'w', encoding='utf-8') as f:
                f.write(resume_text)
    print(f'[SUCCESS] Generated {len(samples)} realistic sample resumes in {SAMPLE_RESUMES_DIR}')

if __name__ == '__main__':
    print('[INFO] Initializing dataset generator...')
    build_dataset(num_samples_per_category=80)
    build_job_descriptions()
    generate_sample_resumes()
    print('[INFO] Data preparation complete!')