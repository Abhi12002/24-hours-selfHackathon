import streamlit as st
st.set_page_config(page_title="JobFit AI", layout="centered")

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import spacy
import openai

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
nlp = spacy.load("en_core_web_sm")

# Set API key from secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OpenAI API key missing in secrets.toml!")
else:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    try:
        models = openai.Model.list()
        st.success("✅ OpenAI connection successful.")
    except Exception as e:
        st.error(f"❌ OpenAI API connection failed: {e}")

st.title("🔍 JobFit AI – Resume & Job Description Matcher")

# --- Resume Upload ---
st.subheader("📄 Upload Resume (PDF)")
resume_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# --- Job Description Input ---
st.subheader("🧾 Paste Job Description")
job_desc = st.text_area("Paste the job description here", height=250)

# --- Load Skill Sets ---
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

# --- Utility Functions ---
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def get_similarity(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def extract_all_skills(text):
    doc = nlp(text.lower())
    keywords = set(
        token.lemma_ for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and token.is_alpha
    )

    verified = {kw for kw in keywords if kw in KNOWN_SKILLS}
    unverified = keywords - verified

    NOISE_WORDS = {
        "hand", "datum", "language", "component", "foundation", "team", "role", "project",
        "solution", "review", "problem", "app", "user", "support", "skill", "quality", "year",
        "bachelor", "learning", "industry", "experience", "understanding", "education"
    }

    filtered_unverified = {
        word for word in unverified
        if word not in NOISE_WORDS and len(word) > 2
    }

    return list(verified), list(filtered_unverified)

def semantic_skill_match(source_skills, target_skills, threshold=0.85):
    matched = set()
    unmatched = set()

    if not source_skills or not target_skills:
        return matched, source_skills

    target_embeddings = model.encode(list(target_skills), convert_to_tensor=True)

    for skill in source_skills:
        skill_embedding = model.encode(skill, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(skill_embedding, target_embeddings)[0]

        if scores.max().item() >= threshold:
            matched.add(skill)
        else:
            unmatched.add(skill)

    return matched, unmatched

def format_tags(skills, color):
    if not skills:
        return "None"
    return " ".join([
        f'<span style="background-color:{color}; color:white; padding:4px 8px; border-radius:12px; margin:3px; display:inline-block;">{skill}</span>'
        for skill in sorted(skills)
    ])

def detect_resume_sections(text):
    sections = {
        "Contact": ["contact", "phone", "email", "linkedin"],
        "Skills": ["skills", "technical skills", "technologies"],
        "Experience": ["experience", "work history", "employment"],
        "Education": ["education", "qualifications", "academic"],
        "Projects": ["projects", "portfolio", "personal projects"],
        "References": ["references", "referees", "recommendations"]
    }

    found_sections = {}
    lower_text = text.lower()

    for key, keywords in sections.items():
        found_sections[key] = any(kw in lower_text for kw in keywords)

    return found_sections

def generate_cover_letter(resume, job_desc):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes personalized, professional cover letters."},
            {"role": "user", "content": f"Write a concise, tailored cover letter based on the resume below for this job:\n\nJob Description:\n{job_desc}\n\nResume:\n{resume}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",  # uses GPT-4.1 mini
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"❌ Failed to generate cover letter: {e}")
        return ""



# --- Main Logic ---
resume_text = ""
if st.button("🚀 Match Now") and resume_file and job_desc.strip():
    resume_text = extract_text_from_pdf(resume_file)

    if not resume_text.strip():
        st.error("❌ Could not extract any text from the PDF. Is it a scanned document?")
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
        emoji = "✅" if present else "❌"
        st.markdown(f"{emoji} **{sec}**")

    st.success(f"🎯 Match Score: **{score*100:.2f}%**")
    if score >= 0.75:
        st.markdown("✅ Great fit! You’re well aligned with the job.")
    elif score >= 0.5:
        st.markdown("⚠️ Partial match. Consider updating your resume with more relevant skills.")
    else:
        st.markdown("❌ Low match. Resume and JD might not be aligned.")

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
        st.markdown("You may consider adding the following skills to improve your match:")
        st.markdown(format_tags(missing_verified, '#ffc107'), unsafe_allow_html=True)

# --- Cover Letter (now outside Match Now block) ---
if st.button("📝 Write My Cover Letter"):
    st.write("✅ Cover letter button clicked!")
    try:
        if not resume_file or not job_desc:
            st.error("❌ Please upload resume and paste job description first.")
        else:
            if not resume_text:
                resume_text = extract_text_from_pdf(resume_file)
            with st.spinner("Generating your cover letter..."):
                cover_letter = generate_cover_letter(resume_text, job_desc)
                st.text_area("📄 Your AI-Generated Cover Letter", cover_letter, height=300)
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

if st.button("🔍 Test OpenAI API"):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Write a short thank-you message.",
            temperature=0.5,
            max_tokens=50
        )
        st.success("✅ API call successful!")
        st.write(response.choices[0].text.strip())
    except Exception as e:
        st.error(f"❌ API test failed: {e}")

else:
    st.info("Upload a resume and paste a job description to begin.")
