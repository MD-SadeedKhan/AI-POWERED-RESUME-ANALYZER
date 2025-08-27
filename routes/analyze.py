from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import os
import PyPDF2
from docx import Document
import spacy
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from loguru import logger
from configs.config import UPLOAD_FOLDER, DB_URL
from models.resume import Resume
from vectorstore.faiss_store import FaissStore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

router = APIRouter()

# Database setup
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Embedding model and FAISS store
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_store = FaissStore(dim=384)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def process_resume_file(file: UploadFile, db: Session = Depends(get_db)) -> dict:
    try:
        if not file.filename:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in [".pdf", ".docx"]:
            logger.error(f"Invalid file extension: {file_extension}")
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
        
        file_size = file.size
        if file_size == 0:
            logger.error("Empty file uploaded")
            raise HTTPException(status_code=400, detail="File is empty")
        if file_size > 10 * 1024 * 1024:
            logger.error(f"File too large: {file_size} bytes")
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

        # Create uploads directory
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Generate unique temporary file name
        temp_file_name = f"{Path(file.filename).stem}_{os.urandom(8).hex()}{file_extension}"
        temp_file_path = os.path.join(UPLOAD_FOLDER, temp_file_name)
        
        # Save file temporarily
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text
        text = ""
        try:
            if file_extension == ".pdf":
                with open(temp_file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        text += extracted if extracted else ""
            elif file_extension == ".docx":
                doc = Document(temp_file_path)
                text = "".join(paragraph.text + "\n" for paragraph in doc.paragraphs if paragraph.text)
        finally:
            # Delete temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info(f"Temporary file deleted: {temp_file_path}")
        
        # Perform NER with spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        # Extract entities
        skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG", "PERSON"]]
        education = []
        experience = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in ["bachelor", "master", "phd", "degree", "university", "college"]):
                education.append(sent.text.strip())
            if any(keyword in sent_text for keyword in ["worked", "employed", "experience", "job", "role", "intern"]):
                experience.append(sent.text.strip())
        
        # Generate embedding
        embedding = model.encode([text])[0]
        
        # Save metadata to PostgreSQL
        resume = Resume(
            filename=file.filename,
            extracted_text=text,
            entities={"skills": skills[:10], "education": education[:5], "experience": experience[:5]}
        )
        db.add(resume)
        db.commit()
        db.refresh(resume)
        
        # Store in FAISS
        faiss_store.add(embedding, resume.id)
        
        response = {
            "filename": file.filename,
            "extracted_text": text[:500],
            "text_length": len(text),
            "entities": {
                "skills": skills[:10],
                "education": education[:5],
                "experience": experience[:5]
            },
            "message": "Resume analyzed and stored successfully"
        }
        logger.info(f"Resume processed: {file.filename}, text length: {len(text)}, resume_id: {resume.id}")
        return response

    except Exception as e:
        # Ensure temporary file is deleted on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Temporary file deleted on error: {temp_file_path}")
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.post("/resume", summary="Analyze a resume file (PDF or DOCX)", response_model=dict)
async def analyze_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        response = await process_resume_file(file, db)
        logger.info(f"Analysis successful: {response}")
        return response
    except HTTPException as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")