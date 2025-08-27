from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from database import Base   # ✅ use shared Base from database.py

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    extracted_text = Column(String, nullable=True)  # resume text
    entities = Column(JSON, nullable=True)
    embedding = Column(JSON, nullable=True)         # ✅ new column for vector storage
    uploaded_at = Column(DateTime, default=datetime.utcnow)
