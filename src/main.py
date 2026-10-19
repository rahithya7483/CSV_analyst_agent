"""FastAPI backend: dataset upload / listing and chat.

Run from the project root:  uvicorn src.main:app --reload
"""
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.agent import CsvAnalystAgent
from src.config import DATA_DIR, MOCK_LLM
from src.data_loader import DatasetStore, load_csv_bytes, load_samples
from src.llm_client import get_client

MAX_CACHED_AGENTS = 200

store = DatasetStore()
uploaded_names = set()
agents: Dict[tuple, CsvAnalystAgent] = {}
_client = None


def llm():
    global _client
    if _client is None:
        _client = get_client()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_samples(store, DATA_DIR)
    yield


app = FastAPI(title="CSV Analyst Agent", version="2.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    dataset: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[Dict[str, Any]]
    steps: int
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": MOCK_LLM, "datasets": len(store.names())}


@app.get("/datasets")
def list_datasets():
    return [store.summary(name) for name in store.names()]


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        df = load_csv_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(file.filename).stem)[:40] or "dataset"
    if name in store.names() and name not in uploaded_names:
        name += "_upload"  # never overwrite a built-in sample dataset
    store.add(name, df)
    uploaded_names.add(name)
    for key in [k for k in agents if k[1] == name]:  # drop stale agents bound to the old data
        del agents[key]
    return store.summary(name)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):  # sync on purpose: FastAPI runs it in a worker thread
    try:
        df = store.get(req.dataset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{req.dataset}'.")

    key = (req.session_id, req.dataset)
    agent = agents.get(key)
    if agent is None:
        try:
            agent = CsvAnalystAgent(df, llm())
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        if len(agents) >= MAX_CACHED_AGENTS:
            agents.pop(next(iter(agents)))
        agents[key] = agent
    return ChatResponse(**agent.ask(req.question).to_dict())
