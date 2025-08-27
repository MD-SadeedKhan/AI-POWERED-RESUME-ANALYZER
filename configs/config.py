import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = "redis://localhost:6379/0"
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")