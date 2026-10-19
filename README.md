# CSV Analyst Agent

A tool-calling LLM agent that answers plain-English questions about any CSV file.
The model never does the maths itself: every number in an answer comes from a
deterministic pandas tool, and the project ships with an evaluation harness that
measures accuracy against pandas ground truth.

Built with GPT-OSS-20B (Groq API), FastAPI, Streamlit and Docker.

**Live demo:** `https://csvanalystagent.streamlit.app/`

## Why this exists

The first version was a notebook with three tools and a hardcoded 15-row table.
Asked *"Which department has higher salaries, Engineering or Marketing?"* it answered
with an Engineering average of 78,500 and a Marketing average of 55,000. The real
averages are 81,667 and 60,000. The agent had no tool that could group by department,
so the model averaged the few rows it had seen and presented the result confidently.

This version fixes the cause (a `group_stats` tool and filters on every aggregate),
and adds what was missing to trust it: input validation, retries, a step limit, real
CSV upload, and a measured accuracy score.

## Features

- **Six pandas tools** with JSON schemas: dataset description, column statistics,
  filtering, group-by aggregation, top-N and value counts. Filters combine with AND.
- **Self-correcting loop**: tool errors (unknown column, text column used as numeric,
  bad operator) are returned to the model with the list of valid options so it can retry.
- **Reliability**: retry with exponential backoff on rate limits, server errors and
  malformed tool calls; a bounded step limit; every failure ends as a clean error
  result rather than a crash.
- **Any CSV**: upload through the UI or API (5 MB / 200,000 row limit, validated).
- **Follow-up questions**: per-session conversation memory.
- **Transparent**: the UI shows every tool call, its arguments and its raw result.
- **Evaluation harness**: 27 questions with ground truth computed by pandas, plus a
  no-tools baseline to show what the tools add.
- **Offline mode**: `MOCK_LLM=true` runs the whole stack without an API key.

## Architecture

```
Streamlit UI  --HTTP-->  FastAPI  -->  CsvAnalystAgent  -->  Groq (GPT-OSS-20B)
 (chat, upload,          /chat          loop, max 6 steps        |
  tool-call view)        /datasets      retries, history         | tool calls
                         /upload             |                   v
                                             +----------->  CsvTools (pandas)
                                                            validate -> compute -> JSON
```

1. The system prompt tells the model the dataset's size and columns, and to take every
   number from a tool.
2. The model picks a tool and arguments. The tool validates them, runs pandas and
   returns JSON (or an error message).
3. The result goes back to the model, which calls more tools or writes the final answer.
4. If the loop reaches the step limit, the agent stops and says so.

## Tools

| Tool | What it does |
|---|---|
| `describe_dataset` | Row count, column types, missing values, example values |
| `get_stats` | Count, mean, median, min, max, sum, std of a numeric column (optional filters) |
| `filter_rows` | Count of matching rows plus a preview |
| `group_stats` | mean / sum / min / max / count / median per group (optional filters) |
| `top_n` | Highest or lowest rows by a numeric column |
| `value_counts` | Frequency of each value in a column |

Filter operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`. Text matching ignores
case, and column names are matched case-insensitively.

## Evaluation

`evaluation/questions.py` holds 27 questions: 15 on a small employee table and 12 on a
240-row synthetic sales table. Each question's correct answer is computed with pandas
at run time, and the harness refuses to run a question whose answer is a tie.

```bash
python -m evaluation.run_eval --baseline          # agent vs. no-tools baseline
python -m evaluation.run_eval --show-answers      # print every answer for review
```

The baseline gives the model only the first 20 rows as text and no tools, which is
what the first version effectively did.

**Results** (GPT-OSS-20B via Groq, temperature 0):

| | Agent with tools | Baseline (no tools) |
|---|---|---|
| Overall (27 questions) | `26/27 (96.3%)` | `16/27 (59.3%)` |
| employees (15 rows) | `15/15` | `15/15` |
| sales (240 rows) | `11/12 (91.7%)` | `1/12(8.3%)` |
| Avg tool calls per question | `1.15` | 0 |
| Avg time per question |	`2.4 s` |	`1.0 s` |

How answers are scored: a number counts as correct if the answer contains it within
0.5%; a name counts if it appears in the answer text; list questions need every item.
This is a containment check, so it can rarely pass or fail an answer a human would
judge differently. Use `--show-answers` to review borderline cases.

The sales data is synthetic (`scripts/generate_sample_sales.py`, seed 42), so results
describe behaviour on clean tabular data, not on messy real-world files.

## Project structure

```
csv-analyst-agent/
├── src/
│   ├── config.py          # environment-driven settings
│   ├── data_loader.py     # CSV loading, validation, dataset registry
│   ├── tools.py           # six pandas tools, argument validation, JSON schemas
│   ├── llm_client.py      # Groq client, retry with backoff, offline mock
│   ├── agent.py           # bounded tool-calling loop with conversation history
│   └── main.py            # FastAPI: /health /datasets /datasets/upload /chat
├── streamlit_app.py       # chat UI with tool-call inspector
├── evaluation/            # questions, answer checks, runner, baseline, CLI
├── tests/                 # 30 unit tests (no API key needed)
├── data/                  # employees.csv, sales.csv (synthetic)
├── scripts/generate_sample_sales.py
├── Dockerfile, start.sh, requirements.txt, .env.example
```

## Setup

```bash
git clone <repo-url> && cd csv-analyst-agent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the API and UI in two terminals:

```bash
uvicorn src.main:app --reload          # http://localhost:8000/docs
streamlit run streamlit_app.py         # http://localhost:8501
```

No key yet? Set `MOCK_LLM=true` in `.env` to try the interface with a canned reply.

### Docker

```bash
docker build -t csv-analyst-agent .
docker run -p 7860:7860 --env-file .env csv-analyst-agent
```

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status check |
| `GET /datasets` | Loaded datasets and their columns |
| `POST /datasets/upload` | Upload a CSV (multipart `file`) |
| `POST /chat` | `{session_id, dataset, question}` returns the answer, tool calls, steps |

### Tests

```bash
python -m unittest discover -s tests -t . -v
```

The tests cover every tool against pandas, error handling, the agent loop (with a
scripted fake LLM: recovery from tool errors, step limit, retry and backoff, fast
failure on non-retryable errors, follow-up memory) and the evaluation scoring.

## Limitations

- **One table at a time.** No joins across files, and no derived columns or date
  arithmetic (the sales data includes a `month` column for this reason).
- **Data leaves your machine.** The schema, example values and tool outputs (including
  previews of up to 20 rows) are sent to the Groq API. Don't upload sensitive data.
- **Free-tier limits.** Groq rate limits slow the evaluation run; the harness waits
  between questions and retries automatically.
- **State is in memory.** Uploaded datasets and conversations are lost on restart.
- **Small evaluation set.** 27 questions is enough to compare agent and baseline, not
  to claim general accuracy. Model outputs are not fully deterministic even at
  temperature 0.

## Future work

- Derived columns and date handling (month, year) as a tool
- A chart tool that returns a matplotlib figure for the UI
- Persistent sessions and datasets (SQLite)
- A larger, messier evaluation set, including questions that should be refused
- Compare other models and a self-check step that re-verifies numbers before answering
