from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from google.api_core.exceptions import GoogleAPIError
from sqlalchemy.orm import Session
import google.generativeai as genai
from loguru import logger
from configs.config import GEMINI_API_KEY, DB_URL
from models.resume import Resume
from vectorstore.faiss_store import FaissStore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import spacy
import shap
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Dict, List
import re
import asyncio
import uuid
import time
import os
from dotenv import load_dotenv
import os
load_dotenv()
# Router setup
router = APIRouter(tags=["chat"])

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Models and FAISS setup
model = SentenceTransformer("all-MiniLM-L6-v2")
EXPECTED_DIM = 384
try:
    faiss_store = FaissStore(dim=EXPECTED_DIM)  # No mmap for Windows compatibility
    if faiss_store.dim != EXPECTED_DIM:
        raise ValueError(
            f"FAISS index dimension {faiss_store.dim} does not match expected {EXPECTED_DIM}"
        )
except Exception as e:
    logger.error(f"Failed to initialize FAISS index: {str(e)}")
    raise

nlp = spacy.load("en_core_web_sm")

# Database setup
if not DB_URL:
    raise ValueError("DB_URL is not set")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Custom exceptions
class ResumeNotFoundError(Exception):
    pass


class EmbeddingError(Exception):
    pass


class AIServiceError(Exception):
    pass


# Utility functions
def sanitize_input(text: str) -> str:
    """Remove malicious characters and normalize input."""
    if not text or text.isspace():
        raise ValueError("Input cannot be empty or whitespace-only")
    text = re.sub(r"<[^>]+>|{+}|\"+|;+|`+", "", text)
    text = (
        re.sub(r"\s+", " ", text.strip())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return text[:1000]


def chunk_resume_text(text: str, max_tokens: int = 500) -> str:
    """Chunk resume text by sentences, removing irrelevant sections."""
    irrelevant_patterns = [
        r"contact info.*?\n",
        r"phone:.*?\|",
        r"email:.*?\|",
        r"linkedin:.*?\|",
        r"github:.*?\|",
    ]
    for pattern in irrelevant_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    sentences = text.split(". ")
    result, token_count = [], 0
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if token_count + sentence_tokens <= max_tokens:
            result.append(sentence)
            token_count += sentence_tokens
        else:
            break
    return ". ".join(result).strip()


# Pydantic request model
class ChatRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=1000, pattern=r"^[\w\s,.?!-]+$"
    )
    job_description: Optional[str] = Field(
        None, max_length=5000, pattern=r"^[\w\s,.?!-]+$"
    )


# Skill synonym mapping
SKILL_SYNONYMS = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "python": "Python",
    "sql": "SQL",
    "fastapi": "FastAPI",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "docker": "Docker",
    "javascript": "JavaScript",
    "react": "React",
    "tensorflow": "TensorFlow",
}


# Main endpoint
@router.post(
    "/", summary="Chat with resume data using RAG and Gemini", response_model=dict
)
async def chat_with_resume(request: ChatRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    logger.info(
        {
            "request_id": request_id,
            "query": request.query,
            "job_description": request.job_description,
            "event": "request_start",
        }
    )

    try:
        # Sanitize inputs
        query = sanitize_input(request.query)
        job_description = (
            sanitize_input(request.job_description)
            if request.job_description
            else None
        )

        # Generate query embedding
        loop = asyncio.get_event_loop()
        try:
            query_embedding = await loop.run_in_executor(
                None,
                lambda: np.array(
                    model.encode([query], normalize_embeddings=True)
                ),
            )
        except Exception as e:
            logger.error(f"Request ID: {request_id} | Embedding error: {str(e)}")
            raise EmbeddingError("Failed to generate query embedding")

        embedding_time = time.time() - start_time
        logger.info(
            {
                "request_id": request_id,
                "event": "embedding_completed",
                "latency": embedding_time,
            }
        )

        # FAISS search
        k = min(5, faiss_store.index.ntotal)
        resume_ids = faiss_store.search(query_embedding[0], k=k)
        logger.info(
            f"Request ID: {request_id} | FAISS search returned IDs: {resume_ids}"
        )

        # Fetch resumes from DB
        resumes = db.query(Resume).filter(Resume.id.in_(resume_ids)).all()
        if not resumes:
            logger.warning(
                f"Request ID: {request_id} | No resumes found for IDs: {resume_ids}"
            )
            raise ResumeNotFoundError("No relevant resumes found")

        resume_map = {r.id: r for r in resumes if r}
        relevant_texts, resume_entities = [], []
        for rid in resume_ids:
            resume = resume_map.get(rid)
            if resume:
                relevant_texts.append(chunk_resume_text(resume.extracted_text))
                resume_entities.append(resume.entities)
            else:
                logger.warning(
                    f"Request ID: {request_id} | No resume found for ID {rid}"
                )

        rag_context = relevant_texts[:3]
        if not rag_context:
            logger.warning(f"Request ID: {request_id} | No valid resume texts found")
            raise ResumeNotFoundError("No relevant resumes found")

        # Skill matching
        matched_skills, missing_skills, shap_explanation = [], [], {}
        skill_weights = {}
        if job_description:
            job_doc = nlp(job_description.lower())
            skill_list = list(SKILL_SYNONYMS.keys())
            job_skills = list(
                set(
                    [ent.text for ent in job_doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
                    + [
                        token.text
                        for token in job_doc
                        if token.text.lower() in skill_list
                    ]
                )
            )
            job_skills = [
                SKILL_SYNONYMS.get(skill.lower(), skill) for skill in job_skills
            ]
            skill_weights = {
                skill: job_description.lower().count(skill.lower())
                for skill in job_skills
            }

            all_resume_skills_lower = [
                s.lower() for entities in resume_entities for s in entities.get("skills", [])
            ]
            all_resume_skills = [
                SKILL_SYNONYMS.get(skill, skill) for skill in all_resume_skills_lower
            ]
            matched_skills = [
                skill
                for skill in job_skills
                if skill.lower() in [s.lower() for s in all_resume_skills]
            ]
            missing_skills = [
                skill
                for skill in job_skills
                if skill.lower() not in [s.lower() for s in all_resume_skills]
            ]

            # SHAP computation (optional, optimized)
            if matched_skills or missing_skills:
                start_shap = time.time()
                try:
                    job_embedding = await loop.run_in_executor(
                        None,
                        lambda: np.array(
                            model.encode([job_description], normalize_embeddings=True)
                        ),
                    )
                    resume_embedding = await loop.run_in_executor(
                        None,
                        lambda: np.array(
                            model.encode(
                                [" ".join(relevant_texts[:1])],
                                normalize_embeddings=True,
                            )
                        ),
                    )
                    explainer = shap.KernelExplainer(
                        lambda x: cosine_similarity(x, job_embedding).flatten(),
                        resume_embedding,
                        nsamples=10,  # Reduced for speed
                    )
                    shap_values = explainer.shap_values(resume_embedding)
                    if np.any(np.isnan(shap_values)):
                        logger.warning(
                            f"Request ID: {request_id} | SHAP returned NaN values"
                        )
                        shap_values = [[0.0]]  # Fallback
                    shap_explanation = {
                        "positive_contributors": {
                            skill: float(shap_values[0][0] * skill_weights.get(skill, 1))
                            for skill in matched_skills[:3]
                        },
                        "negative_contributors": {
                            skill: float(-shap_values[0][0] * skill_weights.get(skill, 1))
                            for skill in missing_skills[:3]
                        },
                    }
                except Exception as e:
                    logger.error(
                        f"Request ID: {request_id} | SHAP computation error: {str(e)}"
                    )
                    shap_explanation = {}  # Fallback
                shap_time = time.time() - start_shap
                logger.info(
                    {
                        "request_id": request_id,
                        "event": "shap_completed",
                        "latency": shap_time,
                    }
                )

        # Similarity score calculation
        if relevant_texts:
            resume_embedding = np.array(
                model.encode([" ".join(relevant_texts)], normalize_embeddings=True)
            )
            similarity_score = float(
                cosine_similarity(
                    query_embedding.reshape(1, -1),
                    resume_embedding.reshape(1, -1),
                )[0][0]
            )
        else:
            similarity_score = 0.0
        similarity_score = min(
            similarity_score / (1 + len(" ".join(relevant_texts)) / 1000), 1.0
        )

        # Prepare system prompt
        system_prompt = """
You are an expert AI career coach and recruiter. 
Respond naturally, like a human having a friendly conversation with the user.
Your goal is to provide clear, actionable insights about resumes, skills, education, experience, and job fit.

Always follow these guidelines:
1. Use a warm, conversational tone.
2. Summarize key strengths and matched skills in bullet points (max 5 skills).
3. Highlight gaps or missing skills politely (max 5 skills).
4. Reference relevant sections from the candidate’s resume.
5. Explain ATS similarity if job description is provided.
6. Provide constructive recommendations.

Structured data:
- Matched skills: {matched_skills}
- Missing skills: {missing_skills}
- Similarity score: {similarity_score:.2f}
- Resume excerpts: {rag_context_used}

User's question: {user_query}
"""
        query_escaped = query.replace("{", "{{").replace("}", "}}")
        prompt = system_prompt.format(
            matched_skills=matched_skills[:5],
            missing_skills=missing_skills[:5],
            similarity_score=similarity_score,
            rag_context_used=[ctx[:200] for ctx in rag_context],
            user_query=query_escaped,
        )[:1500]

        # Gemini API call with retry
        gemini_start = time.time()
        for attempt in range(3):
            try:
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                response = await loop.run_in_executor(
                    None, lambda: gemini_model.generate_content(prompt)
                )
                gemini_feedback = getattr(
                    response, "text", "AI service returned no response"
                )[:1000]
                if not gemini_feedback.strip():
                    logger.warning(
                        f"Request ID: {request_id} | Gemini returned empty response"
                    )
                    gemini_feedback = "AI service returned no response"
                break
            except GoogleAPIError as ae:
                if attempt == 2:
                    logger.error(
                        f"Request ID: {request_id} | Gemini API error after retries: {str(ae)}"
                    )
                    raise AIServiceError("Failed to connect to AI service")
                await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                if attempt == 2:
                    logger.error(
                        f"Request ID: {request_id} | Unexpected Gemini error: {str(e)}"
                    )
                    raise AIServiceError("Unexpected error from AI service")
                await asyncio.sleep(2**attempt)

        gemini_time = time.time() - gemini_start
        logger.info(
            {
                "request_id": request_id,
                "event": "gemini_completed",
                "latency": gemini_time,
            }
        )

        latency = time.time() - start_time
        logger.info(
            {
                "request_id": request_id,
                "event": "request_completed",
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "latency": latency,
            }
        )

        return {
            "resume_ids": resume_ids,
            "similarity_score": similarity_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "shap_explanation": shap_explanation,
            "rag_context_used": rag_context,
            "gemini_feedback": gemini_feedback,
            "message": "Chat response generated successfully",
        }

    except ResumeNotFoundError:
        logger.warning(f"Request ID: {request_id} | No relevant resumes found")
        raise HTTPException(status_code=404, detail="No relevant resumes found")
    except ValueError as ve:
        logger.error(f"Request ID: {request_id} | Invalid input: {str(ve)}")
        raise HTTPException(status_code=400, detail="Invalid query or job description")
    except EmbeddingError as ee:
        logger.error(f"Request ID: {request_id} | Embedding error: {str(ee)}")
        raise HTTPException(status_code=500, detail="Failed to process embeddings")
    except AIServiceError as ae:
        logger.error(f"Request ID: {request_id} | AI service error: {str(ae)}")
        raise HTTPException(status_code=503, detail="Failed to connect to AI service")
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} | Unexpected error: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")
