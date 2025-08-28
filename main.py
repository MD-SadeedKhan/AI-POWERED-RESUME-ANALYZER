import os
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sentence_transformers import SentenceTransformer
import spacy
import uvicorn

from routes.analyze import router as analyze_router
from routes.chat import router as chat_router
from routes.score import router as score_router
from database import engine
from models.resume import Base
from dotenv import load_dotenv

load_dotenv()

# Initialize database
logger.info("Initializing database...")
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Powered Resume Analyzer",
    description="Advanced resume analysis with ATS scoring, optimization, and chat insights",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins (e.g., ["https://your-frontend.com"]) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload models on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Loading Spacy model...")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading en_core_web_sm model...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
    logger.info("Loading SentenceTransformer model...")
    try:
        SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model, adjust if needed
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        raise

# Routers
app.include_router(analyze_router, prefix="/api/v1/analyze", tags=["Analyze"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(score_router, prefix="/api/v1/score", tags=["Score"])

@app.get("/")
async def root():
    return {"message": "Welcome to AI-Powered Resume Analyzer! Use /docs for API details."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or default to 10000
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="error")