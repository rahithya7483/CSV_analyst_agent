"""No-tools baseline: the model sees only the first rows of the CSV as text and must answer from them."""
from src.config import MODEL
from src.llm_client import chat_with_retry


def baseline_answer(client, df, question: str, model: str = MODEL, sample_rows: int = 20) -> str:
    prompt = (f"Here are the first {min(sample_rows, len(df))} of {len(df)} rows of a dataset (CSV):\n"
              f"{df.head(sample_rows).to_csv(index=False)}\n"
              f"Answer the question briefly using only this data.\nQuestion: {question}")
    response = chat_with_retry(client, model=model, temperature=0,
                               messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content or ""
