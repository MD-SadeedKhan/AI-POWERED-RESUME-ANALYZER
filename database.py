# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# PostgreSQL URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5432/resume_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- FastAPI dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Extra Helper Functions ---------------- #

# Fetch raw resume text by ID
def get_resume_text(db: Session, resume_id: int):
    from models.resume import Resume  # lazy import to avoid circular import
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    return resume.extracted_text if resume else None

# Save embeddings to DB
def save_resume_embedding(db: Session, resume_id: int, embedding: list):
    from models.resume import Resume  # lazy import
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if resume:
        resume.embedding = embedding
        db.commit()
        db.refresh(resume)
    return resume

# Retrieve embeddings from DB
def get_resume_embedding(db: Session, resume_id: int):
    from models.resume import Resume  # lazy import
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    return resume.embedding if resume else None
