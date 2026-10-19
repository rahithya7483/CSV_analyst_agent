"""Central configuration, read from environment variables (.env supported)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

GROQ_API_KEY = os.getenv("GROQ_API", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b")
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"

MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))        # model calls per question
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))    # retries per model call
MAX_ROWS_RETURNED = 20                              # rows a tool may send back to the model
MAX_TOOL_CHARS = 6000                               # cap on a tool result sent to the model
