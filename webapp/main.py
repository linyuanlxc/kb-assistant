"""FastAPI application entrypoint for LabKB."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.evaluation.recorder import EvaluationRecorder
from core.types import RetrieverRequest, RetrieverResult, SearchMode
from webapp.bootstrap import get_logger, get_pipeline, get_settings

ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT_DIR / "webapp" / "templates"
STATIC_DIR = ROOT_DIR / "webapp" / "static"
UPLOAD_DIR = ROOT_DIR / "runtime" / "uploads"

app = FastAPI(title="LabKB", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 初始化评估记录仪
recorder = EvaluationRecorder(log_dir=ROOT_DIR / "runtime" / "eval")

MODE_OPTIONS = [
    {
        "label": "Hybrid",
        "value": SearchMode.HYBRID.value,
        "description": "Dense + BM25 + Graph + Image",
    },
    {
        "label": "Text Only",
        "value": SearchMode.TEXT_ONLY.value,
        "description": "Vector and keyword retrieval only",
    },
    {
        "label": "Multimodal",
        "value": SearchMode.MULTIMODAL.value,
        "description": "Image-guided cross-modal retrieval",
    },
    {
        "label": "Graph First",
        "value": SearchMode.GRAPH_FIRST.value,
        "description": "Prioritize graph evidence first",
    },
]


def _ensure_workspace_file(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    root = ROOT_DIR.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="File path is outside the workspace.") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return path


def _save_upload(uploaded: UploadFile) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original_name = uploaded.filename or "image.png"
    suffix = Path(original_name).suffix.lower() or ".png"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in Path(original_name).stem)
    safe_stem = safe_stem.strip("-") or "image"
    target = UPLOAD_DIR / f"{safe_stem}-{uuid4().hex[:8]}{suffix}"
    target.write_bytes(uploaded.file.read())
    return target


def _parse_history(raw_history: str | None) -> list[tuple[str, str]]:
    if not raw_history:
        return []

    try:
        parsed = json.loads(raw_history)
    except json.JSONDecodeError:
        return []

    history: list[tuple[str, str]] = []
    if not isinstance(parsed, list):
        return history

    for item in parsed:
        if isinstance(item, dict):
            role = str(item.get("role", "human"))
            content = str(item.get("content", ""))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role = str(item[0])
            content = str(item[1])
        else:
            continue

        if content.strip():
            history.append((role, content))
    return history


def _serialize_item(item: Any) -> dict[str, Any]:
    payload = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
    source = str(payload.get("source", "") or "")
    payload["is_image"] = source.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    payload["source_name"] = Path(source).name if source else "Unknown"
    payload["source_url"] = f"/files?path={quote(source, safe='')}" if source else ""
    return payload


def _serialize_result(result: RetrieverResult) -> dict[str, Any]:
    return {
        "items": [_serialize_item(item) for item in result.items],
        "scores": result.scores,
        "sources": [
            {
                "path": source,
                "name": Path(source).name if source else "Unknown",
                "url": f"/files?path={quote(source, safe='')}" if source else "",
                "is_image": source.lower().endswith((".jpg", ".jpeg", ".png", ".webp")),
            }
            for source in result.sources
        ],
        "graph_evidence": result.graph_evidence,
        "latency_ms": result.latency_ms,
        "debug_info": result.debug_info,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "mode_options": MODE_OPTIONS,
            "default_mode": SearchMode.HYBRID.value,
        },
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    qdrant_ok = False
    neo4j_ok = False
    errors: dict[str, str] = {}

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        client.get_collections()
        qdrant_ok = True
    except Exception as exc:
        errors["qdrant"] = str(exc)

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1").single()
        driver.close()
        neo4j_ok = True
    except Exception as exc:
        errors["neo4j"] = str(exc)

    return {
        "ready": qdrant_ok and neo4j_ok,
        "qdrant": qdrant_ok,
        "neo4j": neo4j_ok,
        "errors": errors,
        "modes": MODE_OPTIONS,
    }


@app.get("/files")
def read_workspace_file(path: str = Query(...)) -> FileResponse:
    file_path = _ensure_workspace_file(path)
    return FileResponse(file_path)


@app.post("/api/chat/stream")
async def chat_stream(
    query: str = Form(...),
    mode: str = Form(SearchMode.HYBRID.value),
    top_k: int = Form(10),
    debug: bool = Form(False),
    chat_history: str = Form("[]"),
    image: UploadFile | None = File(default=None),
) -> StreamingResponse:
    logger = get_logger()
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline bootstrap failed: {exc}") from exc

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")

    history = _parse_history(chat_history)
    try:
        search_mode = SearchMode(mode)
    except ValueError:
        search_mode = SearchMode.HYBRID

    image_inputs: list[str] = []
    if image and image.filename:
        saved_path = _save_upload(image)
        image_inputs.append(str(saved_path))

    req = RetrieverRequest(
        query=query,
        chat_history=history,
        modality=search_mode,
        image_inputs=image_inputs,
        top_k=top_k,
        debug=debug,
    )

    try:
        stream, rewritten_query, retrieval_result = pipeline.answer_stream(req)
    except Exception as exc:
        logger.exception("chat stream bootstrap failed")
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc

    async def event_stream():
        answer_parts: list[str] = []
        yield _sse(
            "meta",
            {
                "rewritten_query": rewritten_query,
                "uploaded_images": [
                    {
                        "path": path,
                        "name": Path(path).name,
                        "url": f"/files?path={quote(path, safe='')}",
                    }
                    for path in image_inputs
                ],
            },
        )

        try:
            for delta in stream:
                answer_parts.append(delta)
                yield _sse("token", {"delta": delta})
        except Exception as exc:
            logger.exception("chat stream failed")
            yield _sse("error", {"message": str(exc)})
            return

        answer = "".join(answer_parts)

        # 记录评估数据（异步，不影响响应）
        try:
            recorder.record(
                query=query,
                rewritten_query=rewritten_query,
                retrieval_result=retrieval_result,
                answer=answer,
                search_mode=search_mode.value,
                image_inputs=image_inputs,
            )
        except Exception as e:
            # 评估记录失败不影响主流程
            logger.debug(f"Evaluation recording failed: {e}")

        yield _sse(
            "done",
            {
                "answer": answer,
                "retrieval": _serialize_result(retrieval_result),
                "rewritten_query": rewritten_query,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

