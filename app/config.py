import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL")

# FIX: LangChain's ChatGoogleGenerativeAI requires GOOGLE_API_KEY, not GEMINI_API_KEY
# This ensures the env var is always set correctly regardless of what's in .env
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _gemini_key:
    os.environ["GOOGLE_API_KEY"] = _gemini_key
