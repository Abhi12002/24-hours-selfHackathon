import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()
    
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and token.is_alpha:
            keywords.add(token.lemma_)
    
    return list(keywords)

def compare_keywords(jd_keywords, resume_keywords):
    jd_set = set(jd_keywords)
    resume_set = set(resume_keywords)
    
    missing = jd_set - resume_set
    matched = jd_set & resume_set
    
    return list(matched), list(missing)
