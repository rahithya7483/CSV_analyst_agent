"""Streamlit Community Cloud version of the CSV Analyst Agent.

Streamlit Community Cloud runs ONE Streamlit script (no Docker, no second server),
so this file calls CsvAnalystAgent directly instead of going through the FastAPI API.
The Docker version (streamlit_app.py + src/main.py) is unchanged.

Local test:   streamlit run app_cloud.py
Cloud:        set the app's main file to app_cloud.py and add your key under Secrets.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Analyst Agent", page_icon="📊", layout="wide")

# Streamlit Cloud keeps secrets in st.secrets, but the project reads environment variables.
# This must run before src.config is imported.
try:
    for _key in ("GROQ_API", "MODEL", "MOCK_LLM"):
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:
    pass  # no secrets file (local run): .env is used instead

from src.agent import CsvAnalystAgent  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.data_loader import DatasetStore, load_csv_bytes, load_samples  # noqa: E402
from src.llm_client import get_client  # noqa: E402

MAX_QUESTIONS_PER_SESSION = 25  # protects the shared API key from a single visitor

st.title("📊 CSV Analyst Agent")
st.caption("Ask questions about a CSV in plain English. Every number in an answer comes from a pandas tool call.")
st.info("Demo only: do not upload sensitive data. Your file and questions are sent to the Groq API, "
        "and the file is kept in memory for your session only.")


@st.cache_resource
def sample_store() -> DatasetStore:
    store = DatasetStore()
    load_samples(store, DATA_DIR)
    return store


@st.cache_resource
def llm():
    return get_client()


ss = st.session_state
ss.setdefault("uploads", {})   # uploaded datasets: private to this visitor's session
ss.setdefault("agents", {})    # dataset name -> agent (holds follow-up memory)
ss.setdefault("chats", {})     # dataset name -> list of messages
ss.setdefault("asked", 0)
ss.setdefault("last_upload", None)

samples = sample_store()

# ---- sidebar: upload and dataset picker ---------------------------------------
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV (max 5 MB)", type="csv")
    if uploaded is not None and ss.last_upload != (uploaded.name, uploaded.size):
        try:
            new_df = load_csv_bytes(uploaded.getvalue())
        except ValueError as exc:
            st.error(f"Upload failed: {exc}")
        else:
            name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(uploaded.name).stem)[:40] or "dataset"
            if name in samples.names():
                name += "_upload"  # never shadow a built-in sample
            ss.uploads[name] = new_df
            ss.agents.pop(name, None)
            ss.chats.pop(name, None)
            ss.last_upload = (uploaded.name, uploaded.size)
            ss.dataset = name

    names = samples.names() + [n for n in ss.uploads if n not in samples.names()]
    selected = st.selectbox("Choose a dataset", names, key="dataset")
    df = ss.uploads[selected] if selected in ss.uploads else samples.get(selected)
    st.caption(f"{len(df):,} rows")
    with st.expander("Columns"):
        for column in df.columns:
            kind = "numeric" if pd.api.types.is_numeric_dtype(df[column]) else "text"
            st.write(f"`{column}` ({kind})")
    if st.button("Clear conversation"):
        ss.chats.pop(selected, None)
        ss.agents.pop(selected, None)
        st.rerun()

# ---- chat ---------------------------------------------------------------------
messages = ss.chats.setdefault(selected, [])


def show_tool_calls(calls):
    if calls:
        with st.expander(f"Tool calls ({len(calls)})"):
            for call in calls:
                st.markdown(f"**{call['name']}**")
                st.code(json.dumps(call["arguments"], indent=2), language="json")
                st.code(json.dumps(call["result"], indent=2)[:1500], language="json")


for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        show_tool_calls(message.get("tool_calls"))

if question := st.chat_input(f"Ask something about {selected}..."):
    if ss.asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning(f"Demo limit reached ({MAX_QUESTIONS_PER_SESSION} questions per session). "
                   "Refresh the page to start again.")
    else:
        ss.asked += 1
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                agent = ss.agents.get(selected)
                if agent is None:
                    try:
                        agent = CsvAnalystAgent(df, llm())
                    except RuntimeError as exc:  # e.g. GROQ_API missing
                        st.error(str(exc))
                        st.stop()
                    ss.agents[selected] = agent
                result = agent.ask(question).to_dict()
            if result.get("error"):
                st.warning(f"Agent note: {result['error']}")
            st.write(result["answer"])
            show_tool_calls(result["tool_calls"])
        messages.append({"role": "assistant", "content": result["answer"],
                         "tool_calls": result["tool_calls"]})