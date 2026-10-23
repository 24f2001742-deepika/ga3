import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, Boolean,
    select, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker

APP_NAME = "A2A Invoice Agent"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./invoice_agent.db")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")

# Render/Postgres URLs sometimes use postgres://; SQLAlchemy expects postgresql://.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class TaskRecord(Base):
    __tablename__ = "invoice_tasks"
    task_id = Column(String(128), primary_key=True)
    package_hash = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="received")
    result_json = Column(Text, nullable=False, default="{}")
    action_key = Column(String(256), nullable=True, unique=True)
    action_executed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)

app = FastAPI(title=APP_NAME, version="1.0.0")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def public_url(path: str) -> str:
    return f"{PUBLIC_BASE_URL}{path}" if PUBLIC_BASE_URL else path


def agent_card() -> dict:
    return {
        "name": APP_NAME,
        "description": "Reviews invoice packages, explains evidence-based decisions, and prevents repeated actions.",
        "url": public_url("/a2a"),
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [{
            "id": "review_invoice",
            "name": "Review invoice package",
            "description": "Evaluate an invoice package and return a proposed business action with evidence.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-payment"],
            "examples": ["Review this invoice package and recommend the safe next action."]
        }]
    }


@app.get("/")
def root():
    return {
        "service": APP_NAME,
        "status": "ok",
        "agent_card": public_url("/.well-known/agent-card.json"),
        "a2a_endpoint": public_url("/a2a"),
        "health": public_url("/health"),
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": APP_NAME}


@app.get("/.well-known/agent-card.json")
@app.get("/.well-known/agent.json")
def get_agent_card():
    return agent_card()


async def understand_package(package: Any) -> dict:
    """
    Ask AI Pipe to interpret the provided invoice package when configured.
    The fallback is intentionally conservative: it never authorizes payment.
    """
    if not AIPIPE_TOKEN:
        return {
            "proposed_action": "request_review",
            "confidence": 0.0,
            "evidence": ["AIPIPE_TOKEN is not configured; automated document interpretation was not run."],
            "reason": "Manual review is required because invoice evidence could not be interpreted."
        }

    prompt = """
You are a cautious accounts-payable invoice reviewer. Review the supplied invoice package.
Return ONLY a JSON object with these keys:
proposed_action (one of: "pay", "request_approval", "request_information", "reject", "hold", "request_review"),
confidence (number 0 to 1),
invoice_id (string or null),
vendor (string or null),
amount (number or null),
currency (string or null),
evidence (array of short strings quoting or pointing to facts in the supplied package),
reason (short string),
approval_required (boolean),
duplicate_of (string or null).
Rules:
- Use only evidence in the package; do not invent facts.
- A payment for a different invoice is not a duplicate.
- If invoice identity, vendor, amount, currency, receipt, or approval evidence is missing/unclear, do not recommend pay.
- If package indicates a duplicate invoice/payment, recommend hold or reject and cite the matching invoice/payment evidence.
- Never claim a payment has actually been sent.
- Be conservative when policies or approval limits are not provided.
"""
    # AI Pipe's OpenAI-compatible endpoint. If this endpoint is not enabled for
    # the user's account, the exception is handled below and the agent fails safe.
    url = os.getenv("AIPIPE_CHAT_COMPLETIONS_URL", "https://aipipe.org/openrouter/v1/chat/completions")
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"},
                json={
                    "model": os.getenv("AIPIPE_MODEL", "openai/gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(package, ensure_ascii=False, default=str)}
                    ],
                    "temperature": 0
                }
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            # Support responses wrapped in Markdown fences.
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            if result.get("proposed_action") not in {
                "pay", "request_approval", "request_information", "reject", "hold", "request_review"
            }:
                raise ValueError("AI returned an unsupported proposed_action")
            result.setdefault("evidence", [])
            result.setdefault("reason", "No reason supplied.")
            result.setdefault("approval_required", True)
            result["action_executed"] = False
            return result
    except Exception as exc:
        return {
            "proposed_action": "request_review",
            "confidence": 0.0,
            "evidence": ["AI interpretation failed; no action was executed."],
            "reason": f"Could not safely interpret package ({type(exc).__name__}). Manual review required.",
            "approval_required": True,
            "action_executed": False
        }


def extract_package(params: Any) -> Any:
    if isinstance(params, dict):
        # Accept common A2A and REST wrapper shapes.
        for key in ("invoice_package", "package", "invoice", "input", "data", "message"):
            if key in params:
                value = params[key]
                if isinstance(value, dict) and "parts" in value:
                    return value
                if key == "message" and isinstance(value, dict):
                    return value
                return value
    return params


async def review_and_store(package: Any, task_id: Optional[str] = None) -> dict:
    package_hash = stable_hash(package)
    task_id = task_id or f"task-{package_hash[:24]}"

    with SessionLocal() as db:
        existing = db.get(TaskRecord, task_id)
        if existing:
            return json.loads(existing.result_json)

        # If the same exact package is retried under another task ID, return the
        # saved decision instead of running a second action.
        prior = db.scalar(
            select(TaskRecord).where(TaskRecord.package_hash == package_hash).order_by(TaskRecord.created_at.desc())
        )
        if prior:
            saved = json.loads(prior.result_json)
            saved["duplicate_request"] = True
            saved["original_task_id"] = prior.task_id
            return saved

    decision = await understand_package(package)
    decision.update({
        "task_id": task_id,
        "package_hash": package_hash,
        "status": "completed",
        "action_executed": False,
        "created_at": utcnow().isoformat(),
    })

    # Do not perform real-world payment automatically. A proposed "pay" is
    # merely a recommendation; an explicit, separately approved action is needed.
    if decision.get("proposed_action") == "pay":
        decision["status"] = "awaiting_approval"
        decision["reason"] = (
            str(decision.get("reason", "")) +
            " Payment has NOT been sent. Explicit approval and a payment integration are required."
        ).strip()

    with SessionLocal() as db:
        # Protect against a simultaneous retry inserting the same task.
        existing = db.get(TaskRecord, task_id)
        if existing:
            return json.loads(existing.result_json)
        record = TaskRecord(
            task_id=task_id,
            package_hash=package_hash,
            status=decision["status"],
            result_json=json.dumps(decision, ensure_ascii=False, default=str),
            action_key=None,
            action_executed=False,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        db.add(record)
        try:
            db.commit()
        except Exception:
            db.rollback()
            existing = db.get(TaskRecord, task_id)
            if existing:
                return json.loads(existing.result_json)
            raise
    return decision


def jsonrpc_result(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def jsonrpc_error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


@app.post("/review")
async def review_rest(payload: dict):
    package = extract_package(payload)
    task_id = payload.get("task_id") if isinstance(payload, dict) else None
    return await review_and_store(package, task_id)




@app.post("/message:send")
async def message_send(request: Request):
    """Compatibility endpoint expected by the assignment's A2A checker."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    params = body.get("params") if isinstance(body.get("params"), dict) else body
    package = extract_package(params)

    task_id = (
        params.get("taskId")
        or params.get("task_id")
        or body.get("taskId")
        or body.get("task_id")
    )
    context_id = params.get("contextId") or params.get("context_id")

    # Accept the common A2A message shape: {"message": {"parts": [...]}}.
    message_obj = params.get("message")
    if isinstance(message_obj, dict) and "parts" in message_obj:
        package = message_obj

    decision = await review_and_store(package, task_id)
    task_state = "input-required" if decision.get("status") == "awaiting_approval" else "completed"

    return {
        "id": decision["task_id"],
        "taskId": decision["task_id"],
        "contextId": context_id,
        "status": {"state": task_state},
        "artifacts": [{
            "name": "invoice-decision",
            "parts": [{"kind": "data", "data": decision}]
        }],
        "result": decision
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    with SessionLocal() as db:
        record = db.get(TaskRecord, task_id)
        if not record:
            raise HTTPException(status_code=404, detail="Task not found")
        return json.loads(record.result_json)


@app.post("/a2a")
async def a2a(request: Request):
    try:
        body = await request.json()
    except Exception:
        return jsonrpc_error(None, -32700, "Parse error")

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or "method" not in body:
        return jsonrpc_error(body.get("id") if isinstance(body, dict) else None, -32600, "Invalid JSON-RPC request")

    request_id = body.get("id")
    method = body["method"]
    params = body.get("params") or {}

    if method in ("message/send", "tasks/send", "invoice.review", "review_invoice"):
        package = extract_package(params)
        task_id = None
        if isinstance(params, dict):
            task_id = params.get("taskId") or params.get("task_id")
        decision = await review_and_store(package, task_id)

        # A2A-style task response; include a convenient structured result for clients.
        task_state = "input-required" if decision.get("status") == "awaiting_approval" else "completed"
        task = {
            "id": decision["task_id"],
            "contextId": params.get("contextId") if isinstance(params, dict) else None,
            "status": {"state": task_state},
            "artifacts": [{
                "name": "invoice-decision",
                "parts": [{"kind": "data", "data": decision}]
            }]
        }
        return jsonrpc_result(request_id, task)

    if method in ("tasks/get", "task.get"):
        task_id = params.get("id") or params.get("taskId") or params.get("task_id")
        if not task_id:
            return jsonrpc_error(request_id, -32602, "Missing task id")
        with SessionLocal() as db:
            record = db.get(TaskRecord, str(task_id))
            if not record:
                return jsonrpc_error(request_id, -32004, "Task not found")
            decision = json.loads(record.result_json)
        return jsonrpc_result(request_id, {
            "id": decision["task_id"],
            "status": {"state": "input-required" if decision.get("status") == "awaiting_approval" else "completed"},
            "artifacts": [{"name": "invoice-decision", "parts": [{"kind": "data", "data": decision}]}]
        })

    if method in ("tasks/cancel", "task.cancel"):
        # No external action is executed, so cancellation cannot reverse a payment.
        task_id = params.get("id") or params.get("taskId") or params.get("task_id")
        if not task_id:
            return jsonrpc_error(request_id, -32602, "Missing task id")
        with SessionLocal() as db:
            record = db.get(TaskRecord, str(task_id))
            if not record:
                return jsonrpc_error(request_id, -32004, "Task not found")
            record.status = "cancelled"
            record.updated_at = utcnow()
            db.commit()
        return jsonrpc_result(request_id, {"id": str(task_id), "status": {"state": "canceled"}})

    return jsonrpc_error(request_id, -32601, f"Method not found: {method}")
