import streamlit as st
st.set_page_config(page_title="JobFit AI", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import spacy
import openai
import re
from datetime import datetime
import os

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
nlp = spacy.load("en_core_web_sm")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# --- Utility Functions ---
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def get_similarity(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

@st.cache_data
def load_ESCO_skills():
    df = pd.read_csv("skills_en.csv", encoding="utf-8")
    df = df[df['preferredLabel'].notna()]
    return set(df['preferredLabel'].str.lower().str.strip())

@st.cache_data
def load_custom_skills():
    df = pd.read_csv("custom_skills.csv")
    return set(df['skill'].str.lower().str.strip())

KNOWN_SKILLS = load_ESCO_skills().union(load_custom_skills())

def extract_all_skills(text):
    doc = nlp(text.lower())
    keywords = set(
        token.lemma_ for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and token.is_alpha
    )

    verified = {kw for kw in keywords if kw in KNOWN_SKILLS}
    unverified = keywords - verified

    NOISE_WORDS = {"hand", "datum", "language", "team", "skill", "quality", "project", "experience"}
    filtered_unverified = {word for word in unverified if word not in NOISE_WORDS and len(word) > 2}
    return list(verified), list(filtered_unverified)

def semantic_skill_match(source_skills, target_skills, threshold=0.85):
    matched, unmatched = set(), set()
    if not source_skills or not target_skills:
        return matched, source_skills
    target_embeddings = model.encode(list(target_skills), convert_to_tensor=True)
    for skill in source_skills:
        skill_embedding = model.encode(skill, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(skill_embedding, target_embeddings)[0]
        (matched if scores.max().item() >= threshold else unmatched).add(skill)
    return matched, unmatched

def format_tags(skills, color):
    return "None" if not skills else " ".join(
        f'<span style="background-color:{color}; color:white; padding:4px 8px; border-radius:12px; margin:3px; display:inline-block;">{skill}</span>'
        for skill in sorted(skills)
    )

def detect_resume_sections(text):
    sections = {
        "Contact": ["contact", "phone", "email", "linkedin"],
        "Skills": ["skills", "technical skills"],
        "Experience": ["experience", "employment"],
        "Education": ["education", "qualifications"],
        "Projects": ["projects", "portfolio"],
        "References": ["references", "referees"]
    }
    found_sections = {sec: any(word in text.lower() for word in words) for sec, words in sections.items()}
    return found_sections

# 📥 Extract personal info from resume
def extract_personal_info(text):
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text)
    name = text.strip().split('\n')[0] if text.strip() else "Your Name"
    return {
        "name": name.strip(),
        "email": email.group(0) if email else "[Your Email Address]",
        "phone": phone.group(0) if phone else "[Your Phone Number]",
        "address": "[Your Address]",
        "city": "[City, State, Zip Code]",
        "date": datetime.today().strftime('%B %d, %Y')
    }

# ✉️ Generate Cover Letter with Personal Info
def generate_cover_letter(resume, job_desc, info):
    prompt = f"""Generate a professional, personalized cover letter. Use the contact information below at the top:

{info['name']}
{info['address']}
{info['city']}
{info['email']}
{info['phone']}
{info['date']}

Job Description:
{job_desc}

Resume:
{resume}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes professional cover letters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return ""

# --- UI ---
st.title("🔍 JobFit AI – Resume & Job Description Matcher")
st.subheader("📄 Upload Resume (PDF)")
resume_file = st.file_uploader("Choose a PDF file", type=["pdf"])

st.subheader("🧾 Paste Job Description")
job_desc = st.text_area("Paste the job description here", height=250)

resume_text = ""
cover_letter = ""

if st.button("🚀 Match Now") and resume_file and job_desc.strip():
    resume_text = extract_text_from_pdf(resume_file)
    if not resume_text.strip():
        st.error("❌ Could not extract any text from the PDF.")
        st.stop()

    with st.spinner("Analyzing..."):
        sections = detect_resume_sections(resume_text)
        score = get_similarity(resume_text, job_desc)
        jd_verified, jd_unverified = extract_all_skills(job_desc)
        res_verified, res_unverified = extract_all_skills(resume_text)
        matched_verified, missing_verified = semantic_skill_match(jd_verified, res_verified)
        matched_unverified = set(jd_unverified) & set(res_unverified)

    st.subheader("📋 Resume Section Completeness")
    for sec, present in sections.items():
        st.markdown(f"{'✅' if present else '❌'} **{sec}**")

    st.success(f"🎯 Match Score: **{score*100:.2f}%**")
    if score >= 0.75:
        st.markdown("✅ Great fit!")
    elif score >= 0.5:
        st.markdown("⚠️ Partial match.")
    else:
        st.markdown("❌ Low match.")

    st.subheader("🧠 Skill Match Analysis")
    st.markdown("✅ **Matched Verified Skills:**", unsafe_allow_html=True)
    st.markdown(format_tags(matched_verified, '#28a745'), unsafe_allow_html=True)
    st.markdown("❌ **Missing Verified Skills:**", unsafe_allow_html=True)
    st.markdown(format_tags(missing_verified, '#dc3545'), unsafe_allow_html=True)

    if matched_unverified:
        st.markdown("💡 **Additional Skills Detected:**", unsafe_allow_html=True)
        st.markdown(format_tags(matched_unverified, '#007bff'), unsafe_allow_html=True)

    if missing_verified:
        st.subheader("📌 Suggested Resume Improvements")
        st.markdown("Consider adding these skills:")
        st.markdown(format_tags(missing_verified, '#ffc107'), unsafe_allow_html=True)

# ✉️ Generate and download cover letter
if st.button("📝 Write My Cover Letter") and resume_file and job_desc.strip():
    resume_text = resume_text or extract_text_from_pdf(resume_file)
    personal_info = extract_personal_info(resume_text)

    with st.spinner("Generating your cover letter..."):
        cover_letter = generate_cover_letter(resume_text, job_desc, personal_info)

    if cover_letter:
        st.success("✅ Cover letter generated!")
        st.text_area("📄 Your AI-Generated Cover Letter", cover_letter, height=300)

        # Prepare for download
        filename = f"CoverLetter_{personal_info['name'].replace(' ', '_')}.txt"
        st.download_button(
            label="📥 Download Cover Letter (.txt)",
            data=cover_letter,
            file_name=filename,
            mime="text/plain"
        )

else:
    st.info("Upload a resume and paste a job description to begin.")
