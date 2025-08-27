from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from routes.analyze import router as analyze_router
from routes.chat import router as chat_router
from routes.score import router as score_router

from database import engine
from models.resume import Base
from dotenv import load_dotenv
import os
load_dotenv()
# Create DB tables
logger.info("Initializing database...")
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Powered Resume Analyzer",
    description="Advanced resume analysis with ATS scoring, optimization, and chat insights",
    version="1.0.0"
)

# CORS setup (frontend support)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change ["*"] → ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(analyze_router, prefix="/api/v1/analyze", tags=["Analyze"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(score_router, prefix="/api/v1/score", tags=["Score"])

@app.get("/")
async def root():
    return {"message": "Welcome to AI-Powered Resume Analyzer! Use /docs for API details."}
