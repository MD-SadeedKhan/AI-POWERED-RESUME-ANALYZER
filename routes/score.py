# routes/score.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import re

from database import get_db  # <- from your database.py (provides SessionLocal via Depends)
from models.resume import Resume
from vectorstore.faiss_store import FaissStore

router = APIRouter()

# --- Models / Schemas ---
class ScoreRequest(BaseModel):
    resume_id: int
    job_description: str

# --- Embeddings / Vectorstore ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_store = FaissStore(dim=384)  # safe even if empty

# --- Skills dictionary (your expanded set) ---
SKILL_KEYWORDS = [
    # Programming Languages
    "python","java","c","c++","c#","go","rust","scala","r","matlab",
    "typescript","javascript","php","ruby","swift","kotlin","dart",
    "perl","objective-c","bash","shell scripting",
    # Web
    "html","css","react","next.js","angular","vue.js","svelte",
    "node.js","express.js","django","flask","fastapi","spring boot",
    "asp.net","graphql","rest api","redux","webpack","babel",
    "tailwind css","bootstrap","sass","less",
    # Databases
    "sql","mysql","postgresql","postgres","mongodb","redis","cassandra","oracle",
    "firebase","dynamodb","elasticsearch","snowflake","bigquery",
    "redshift","neo4j","cosmos db","couchdb",
    # Cloud
    "aws","azure","gcp","digitalocean","heroku","netlify","vercel",
    "cloud functions","lambda","terraform","ansible","pulumi","openstack","cloudformation",
    # DevOps / CI-CD / Containers
    "docker","kubernetes","jenkins","gitlab ci","github actions",
    "circleci","argo cd","helm","prometheus","grafana","sonarqube",
    "travis ci","splunk","nagios","elk stack",
    # Data / AI / ML
    "machine learning","deep learning","nlp","computer vision",
    "data analysis","data visualization","predictive modeling",
    "reinforcement learning","time series","recommendation systems",
    "pytorch","tensorflow","keras","scikit-learn","xgboost","lightgbm",
    "huggingface","spacy","nltk","openai","llm","langchain",
    "matplotlib","seaborn","plotly","power bi","tableau","excel",
    "pandas","numpy","scipy","statsmodels",
    # Big Data / ETL
    "hadoop","spark","hive","pig","flink","kafka","airflow","beam",
    "databricks","etl","data pipelines","azure data factory","aws glue","snowpipe",
    # Security
    "network security","penetration testing","ethical hacking","cryptography",
    "firewalls","siem","wireshark","owasp","ids/ips","burp suite",
    "nmap","metasploit","security auditing","zero trust","vulnerability scanning",
    # Mobile
    "android","ios","react native","flutter",
    # Testing / QA
    "unit testing","integration testing","selenium","cypress","pytest",
    "junit","mocha","chai","postman","soapui","robot framework","testng","playwright",
    # Tools / VC
    "git","github","gitlab","bitbucket","svn","jira","confluence","slack",
    "trello","notion","figma","miro","azure devops",
    # Soft skills
    "problem solving","communication","teamwork","leadership","critical thinking",
    "adaptability","creativity","time management","collaboration","analytical thinking",
    "decision making","conflict resolution","presentation skills","public speaking"
]

# simple aliases (normalize common variants to dictionary keys)
ALIASES = {
    "postgres": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "ci/cd": "ci cd",
}

def _normalize(text: str) -> str:
    t = text.lower()
    # normalize a few aliases
    for k, v in ALIASES.items():
        t = re.sub(rf"\b{k}\b", v, t)
    # unify punctuation spacing
    t = t.replace("/", " ")
    return t

def extract_skills(text: str) -> list[str]:
    txt = _normalize(text)
    found = []
    for skill in SKILL_KEYWORDS:
        # word-boundary match; escape dots like "node.js"
        pattern = rf"\b{re.escape(skill)}\b"
        if re.search(pattern, txt):
            found.append(skill)
    # de-dup while preserving order
    seen = set()
    ordered = []
    for s in found:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered

def _f1(matched: set, job_skills: set, resume_skills: set) -> float:
    if not job_skills:
        return 0.0
    precision = len(matched) / max(len(resume_skills), 1)
    recall = len(matched) / len(job_skills)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def calculate_ats_score(resume_text: str, job_description: str) -> dict:
    """Pure function: returns ats_score, similarity, matched/missing skills."""
    # embeddings
    resume_emb = embedding_model.encode(resume_text, convert_to_numpy=True)
    job_emb = embedding_model.encode(job_description, convert_to_numpy=True)
    sim = float(cosine_similarity([resume_emb], [job_emb])[0][0])  # [-1,1]
    sim01 = (sim + 1.0) / 2.0                                     # [0,1]

    # skills
    resume_sk = set(extract_skills(resume_text))
    job_sk = set(extract_skills(job_description))
    matched = resume_sk & job_sk
    missing = job_sk - resume_sk

    f1 = _f1(matched, job_sk, resume_sk)

    # final ATS score (0..100): 60% skills, 40% semantic similarity
    ats = 100.0 * (0.6 * f1 + 0.4 * sim01)
    return {
        "ats_score": round(ats, 1),
        "similarity_score": round(sim, 4),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "resume_embedding": resume_emb,   # returned for optional FAISS add by caller
    }

@router.post("/resume", summary="Score a resume against a job description")
async def score_resume(request: ScoreRequest, db: Session = Depends(get_db)):
    try:
        # 1) fetch resume
        resume = db.query(Resume).filter(Resume.id == request.resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")

        # 2) compute ATS
        result = calculate_ats_score(resume.extracted_text or "", request.job_description)

        # 3) (optional) add/update FAISS with the latest embedding for quick future search
        try:
            faiss_store.add(result["resume_embedding"], resume.id)
        except Exception as e:
            logger.warning(f"FAISS add skipped: {e}")

        # 4) build response (don’t return raw embedding)
        payload = {
            "resume_id": request.resume_id,
            "ats_score": result["ats_score"],               # 0..100 number
            "similarity_score": result["similarity_score"], # cosine in [-1,1]
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"],
            "message": "Resume scored successfully",
            # SHAP note: omitted for rule-based; you already get transparent features below
            "explanation": {
                "how_scored": "60% skill-match F1 + 40% semantic similarity",
                "top_positive_signals": result["matched_skills"][:5],
                "top_gaps": result["missing_skills"][:5],
            }
        }
        logger.info(f"Resume {request.resume_id} scored. ATS={payload['ats_score']}, sim={payload['similarity_score']}")
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scoring resume: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error scoring resume")
