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
