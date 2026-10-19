#!/bin/sh
# Start the API in the background, wait until it is healthy, then run the UI in the foreground.
uvicorn src.main:app --host 0.0.0.0 --port 8000 &
python - <<'PY'
import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:8000/health")
        break
    except Exception:
        time.sleep(1)
PY
exec streamlit run streamlit_app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0
