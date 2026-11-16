"""LexiAudit AI - Contract Risk Audit API (Atyq Capital & AI Labs)."""
import hmac
import json
import os
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

VERSION = "1.0.0"
MIN_CHARS = 50
MAX_CHARS = 60000
OPENAI_TIMEOUT = 50.0  # seconds, stays under the 60s function limit
STARTED = time.time()

SYSTEM_PROMPT = """You are LexiAudit, an elite contract risk analyst for business and legal professionals. Audit the contract text provided and identify liability traps, uncapped or one-sided liability, indemnity risks, non-compete and non-solicit overreach, IP assignment overreach, confidentiality definition and duration problems, unilateral termination, auto-renewal, payment/penalty terms, unfavorable governing law, venue and arbitration terms, and missing protections.
Respond with ONLY a valid JSON object, no markdown, in exactly this schema:
{"risk_score": <integer 0-100, higher = riskier>,
"risk_level": "High Risk" | "Moderate Risk" | "Low Risk",
"summary": "<3-5 sentence executive summary>",
"clauses": [{"title": "<short clause title>", "risk": "High" | "Medium" | "Low", "issue": "<why this clause is risky, referencing its actual wording>", "redline": "<exact replacement or edited clause language that can be pasted into the contract>"}]}
Rules: risk_level must be High Risk for scores 67-100, Moderate Risk for 34-66, Low Risk for 0-33. Order clauses from highest to lowest risk. Redlines must be concrete, balanced contract language, not advice. If the text is not a contract, return risk_score 0, risk_level "Low Risk", a summary explaining that, and an empty clauses array."""

app = FastAPI(title="LexiAudit AI API", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None)


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ---------- Gateway secret check (RapidAPI / API.market proxy) ----------
@app.middleware("http")
async def gateway_check(request: Request, call_next):
    secret = os.environ.get("PROXY_SECRET", "")
    if secret and request.method != "OPTIONS" and request.url.path.startswith("/v1/"):
        provided = request.headers.get("x-rapidapi-proxy-secret") or request.headers.get("x-proxy-secret") or ""
        if not provided or not hmac.compare_digest(provided.encode(), secret.encode()):
            return error(401, "unauthorized", "Requests must come through the API gateway.")
    return await call_next(request)


# CORS is added last so it is the outermost layer and answers preflight requests first.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-RapidAPI-Proxy-Secret", "X-Proxy-Secret"],
)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return error(404, "not_found", "Route not found.")
    if exc.status_code == 405:
        return error(405, "method_not_allowed", "Method not allowed for this route.")
    return error(exc.status_code, "http_error", str(exc.detail))


# ---------- Helpers ----------
def normalize(d: dict) -> dict:
    try:
        score = round(float(d.get("risk_score")))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))
    level = "High Risk" if score >= 67 else "Moderate Risk" if score >= 34 else "Low Risk"

    def norm_risk(v) -> str:
        v = str(v or "").lower()
        return "High" if v.startswith("h") else "Medium" if v.startswith("m") else "Low"

    rank = {"High": 0, "Medium": 1, "Low": 2}
    raw = d.get("clauses")
    clauses = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        clauses.append({
            "title": str(c.get("title") or "Untitled clause"),
            "risk": norm_risk(c.get("risk") or c.get("risk_level")),
            "issue": str(c.get("issue") or ""),
            "redline": str(c.get("redline") or ""),
        })
    clauses.sort(key=lambda c: rank[c["risk"]])
    return {"risk_score": score, "risk_level": level, "summary": str(d.get("summary") or ""), "clauses": clauses}


async def call_openai(text: str, jurisdiction: str) -> dict:
    user_msg = (f"Jurisdiction context: {jurisdiction}\n\n" if jurisdiction else "") + f'CONTRACT TEXT:\n"""\n{text}\n"""'
    payload = {
        "model": model_name(),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    if r.status_code != 200:
        print("OpenAI error", r.status_code, r.text[:300])
        raise UpstreamError()
    content = r.json()["choices"][0]["message"]["content"] or ""
    content = re.sub(r"^```json|^```|```$", "", content.strip(), flags=re.M).strip()
    return json.loads(content)


class UpstreamError(Exception):
    pass


# ---------- Endpoints ----------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "LexiAudit AI API", "version": VERSION,
            "uptime_seconds": round(time.time() - STARTED)}


@app.post("/v1/audit")
async def audit(request: Request):
    if not os.environ.get("OPENAI_API_KEY"):
        return error(500, "server_misconfigured", "Server is not configured correctly.")
    try:
        body = await request.json()
    except Exception:
        return error(400, "invalid_json", "Request body is not valid JSON.")
    if not isinstance(body, dict):
        return error(400, "invalid_request", "Request body must be a JSON object.")

    text = body.get("text")
    text = text.strip() if isinstance(text, str) else ""
    jurisdiction = body.get("jurisdiction")
    jurisdiction = jurisdiction.strip()[:100] if isinstance(jurisdiction, str) else ""

    if not text:
        return error(400, "invalid_request", 'The "text" field is required and must be a string.')
    if len(text) < MIN_CHARS:
        return error(400, "text_too_short", f'The "text" field must contain at least {MIN_CHARS} characters.')
    if len(text) > MAX_CHARS:
        return error(413, "text_too_long", f'The "text" field must not exceed {MAX_CHARS} characters.')

    started = time.time()
    try:
        result = normalize(await call_openai(text, jurisdiction))
    except httpx.TimeoutException:
        return error(504, "timeout", "The analysis timed out. Please try again.")
    except json.JSONDecodeError:
        return error(502, "bad_model_output", "The analysis engine returned an invalid response. Please try again.")
    except (UpstreamError, httpx.HTTPError):
        return error(502, "upstream_error", "The analysis engine is temporarily unavailable. Please try again.")
    except Exception as e:  # noqa: BLE001
        print("Unexpected error:", repr(e))
        return error(500, "internal_error", "Unexpected server error.")

    result["meta"] = {"model": model_name(), "characters_analyzed": len(text),
                      "processing_ms": round((time.time() - started) * 1000)}
    return result
