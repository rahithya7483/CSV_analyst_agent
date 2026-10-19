"""Streamlit chat UI for the CSV Analyst Agent (talks to the FastAPI backend)."""
import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CSV Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 CSV Analyst Agent")
st.caption("Ask questions about a CSV in plain English. Every number in an answer comes from a pandas tool call.")

st.session_state.setdefault("session_id", uuid.uuid4().hex)
st.session_state.setdefault("chats", {})       # dataset name -> list of messages
st.session_state.setdefault("last_upload", None)


def api_get(path):
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def error_text(exc):
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            return exc.response.json().get("detail", str(exc))
        except ValueError:
            pass
    return str(exc)


# ---- sidebar: dataset picker / upload -------------------------------------
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV (max 5 MB)", type="csv")
    if uploaded is not None and st.session_state.last_upload != (uploaded.name, uploaded.size):
        try:
            response = requests.post(f"{API_URL}/datasets/upload",
                                     files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}, timeout=60)
            response.raise_for_status()
            st.session_state.last_upload = (uploaded.name, uploaded.size)
            st.session_state.dataset = response.json()["name"]
            st.session_state.chats.pop(st.session_state.dataset, None)
        except requests.RequestException as exc:
            st.error(f"Upload failed: {error_text(exc)}")

    try:
        datasets = api_get("/datasets")
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API_URL}: {exc}")
        st.stop()

    names = [d["name"] for d in datasets]
    selected = st.selectbox("Choose a dataset", names, key="dataset")
    info = next(d for d in datasets if d["name"] == selected)
    st.caption(f"{info['rows']:,} rows")
    with st.expander("Columns"):
        for column in info["columns"]:
            st.write(f"`{column['name']}` ({column['dtype']})")
    if st.button("Clear conversation"):
        st.session_state.chats.pop(selected, None)
        st.rerun()

# ---- chat -------------------------------------------------------------------
messages = st.session_state.chats.setdefault(selected, [])


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
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            try:
                response = requests.post(f"{API_URL}/chat", timeout=120, json={
                    "session_id": st.session_state.session_id, "dataset": selected, "question": question})
                response.raise_for_status()
                data = response.json()
                answer, calls = data["answer"], data["tool_calls"]
                if data.get("error"):
                    st.warning(f"Agent note: {data['error']}")
            except requests.RequestException as exc:
                answer, calls = f"Request failed: {error_text(exc)}", []
        st.write(answer)
        show_tool_calls(calls)
    messages.append({"role": "assistant", "content": answer, "tool_calls": calls})
