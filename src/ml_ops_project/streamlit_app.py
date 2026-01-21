import json
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    data: dict | None
    error: str | None
    status: int | None


def _request_json(method: str, url: str, payload: dict | None = None, timeout: int = 10) -> ApiResult:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return ApiResult(ok=True, data=parsed, error=None, status=resp.status)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        if detail:
            with suppress(json.JSONDecodeError):
                detail = json.loads(detail)
        return ApiResult(ok=False, data=None, error=f"HTTP {exc.code}: {detail}", status=exc.code)
    except Exception as exc:
        return ApiResult(ok=False, data=None, error=str(exc), status=None)


def _clean_lines(raw: str) -> list[str]:
    lines = [line.strip() for line in raw.splitlines()]
    return [line for line in lines if line]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
        :root {
            --ink: #0b1d1c;
            --muted: #4e5d5b;
            --accent: #ff6f3c;
            --accent-2: #0ea5a4;
            --card: #ffffff;
            --stroke: #e6dfd1;
            --bg-1: #f6f2ea;
            --bg-2: #e7efe9;
        }
        html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
        }
        .stApp {
            background: radial-gradient(1200px 600px at 10% -10%, #fff7ea 0%, transparent 60%),
                        radial-gradient(1000px 500px at 100% 0%, #e0f3f2 0%, transparent 55%),
                        linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--ink);
        }
        .hero {
            background: var(--card);
            border: 1px solid var(--stroke);
            border-radius: 18px;
            padding: 28px 32px;
            box-shadow: 0 12px 24px rgba(11, 29, 28, 0.08);
            animation: fadeUp 0.7s ease-out;
        }
        .hero-title {
            font-size: 38px;
            font-weight: 700;
            margin: 0 0 6px 0;
        }
        .hero-sub {
            color: var(--muted);
            font-size: 16px;
            margin: 0;
        }
        .badge {
            font-family: "IBM Plex Mono", monospace;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent-2);
            margin-bottom: 8px;
            display: inline-block;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--stroke);
            border-radius: 16px;
            padding: 20px 24px 10px 24px;
            box-shadow: 0 8px 16px rgba(11, 29, 28, 0.08);
            animation: fadeUp 0.8s ease-out;
        }
        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin: 0 0 6px 0;
        }
        .hint {
            color: var(--muted);
            font-size: 14px;
            margin: 0 0 12px 0;
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_predictions(predictions: list[dict]) -> None:
    if not predictions:
        st.info("No predictions returned.")
        return
    for pred in predictions:
        label_id = pred.get("label_id")
        label = pred.get("label") or (f"Label {label_id}" if label_id is not None else "Unknown")
        confidence = float(pred.get("confidence", 0.0))
        cols = st.columns([2, 1, 1])
        cols[0].metric("Label", label)
        cols[1].metric("Label ID", str(label_id))
        cols[2].metric("Confidence", f"{confidence:.3f}")
        st.progress(max(0.0, min(confidence, 1.0)))


def main() -> None:
    st.set_page_config(page_title="Transaction Classifier", page_icon=":credit_card:", layout="wide")
    _inject_styles()

    st.sidebar.markdown("### API control")
    api_url = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
    api_url = api_url.rstrip("/")

    if st.sidebar.button("Check health"):
        health = _request_json("GET", f"{api_url}/health")
        if health.ok:
            st.sidebar.success(f"API healthy ({health.data.get('status', 'ok')})")
        else:
            st.sidebar.error(health.error or "Health check failed.")

    st.markdown(
        """
        <div class="hero">
            <div class="badge">Transaction Classifier</div>
            <div class="hero-title">Receipt line-item predictions</div>
            <p class="hero-sub">Connect to the FastAPI endpoint and label transaction text in real time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Single prediction</div>
                <p class="hint">Try short merchant descriptions like "STARBUCKS" or "UBER".</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        single_text = st.text_input("Transaction text", value="", placeholder="e.g. STARBUCKS #1247 SEATTLE")
        if st.button("Predict single"):
            if not single_text.strip():
                st.warning("Enter a transaction description first.")
            else:
                result = _request_json("POST", f"{api_url}/predict", {"text": single_text})
                if result.ok and result.data:
                    st.success(f"Model: {result.data.get('model', 'unknown')}")
                    _render_predictions(result.data.get("predictions", []))
                else:
                    st.error(result.error or "Prediction failed.")

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Batch prediction</div>
                <p class="hint">Paste one transaction per line to score a batch.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        batch_text = st.text_area(
            "Transactions (one per line)",
            height=180,
            placeholder="STARBUCKS\nAMAZON PRIME\nUBER TRIP",
        )
        if st.button("Predict batch"):
            texts = _clean_lines(batch_text)
            if not texts:
                st.warning("Add at least one transaction line.")
            else:
                result = _request_json("POST", f"{api_url}/predict", {"texts": texts})
                if result.ok and result.data:
                    st.success(f"Model: {result.data.get('model', 'unknown')}")
                    predictions = result.data.get("predictions", [])
                    rows = []
                    for text, pred in zip(texts, predictions, strict=False):
                        label_id = pred.get("label_id")
                        rows.append(
                            {
                                "text": text,
                                "label": pred.get("label")
                                or (f"Label {label_id}" if label_id is not None else "Unknown"),
                                "label_id": label_id,
                                "confidence": round(float(pred.get("confidence", 0.0)), 3),
                            }
                        )
                    st.dataframe(rows, width="stretch")
                else:
                    st.error(result.error or "Batch prediction failed.")

    st.write("")
    st.info("Tip: Set `USE_DUMMY_PREDICTOR=1` when running the API for fast local demos without loading a model.")


if __name__ == "__main__":
    main()
