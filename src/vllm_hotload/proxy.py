from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class ProxyReplica:
    node: str
    base_url: str
    v1_url: str
    managed_url: str


def _coerce_replica(replica: ProxyReplica | dict[str, str]) -> ProxyReplica:
    if isinstance(replica, ProxyReplica):
        return replica
    return ProxyReplica(
        node=replica["node"],
        base_url=replica["base_url"],
        v1_url=replica["v1_url"],
        managed_url=replica["managed_url"],
    )


def _load_replicas_from_state_file(state_file: Path) -> list[ProxyReplica]:
    data = json.loads(state_file.read_text())
    return [_coerce_replica(replica) for replica in data["replicas"]]


class ReplicaPool:
    def __init__(
        self,
        replica_loader: Callable[[], list[ProxyReplica]],
        health_timeout: float = 2.0,
    ) -> None:
        self._replica_loader = replica_loader
        self._health_timeout = health_timeout
        self._cursor = 0

    def _load_replicas(self) -> list[ProxyReplica]:
        return self._replica_loader()

    async def _is_healthy(
        self, client: httpx.AsyncClient, replica: ProxyReplica
    ) -> bool:
        try:
            managed = await client.get(
                f"{replica.managed_url}/status", timeout=self._health_timeout
            )
            if managed.status_code >= 400:
                return False

            models = await client.get(
                f"{replica.v1_url}/models", timeout=self._health_timeout
            )
            return models.status_code < 400
        except httpx.HTTPError:
            return False

    async def healthy_replicas(self, client: httpx.AsyncClient) -> list[ProxyReplica]:
        replicas = self._load_replicas()
        healthy: list[ProxyReplica] = []
        for replica in replicas:
            if await self._is_healthy(client, replica):
                healthy.append(replica)

        if not healthy:
            return []

        start = self._cursor % len(healthy)
        ordered = healthy[start:] + healthy[:start]
        self._cursor = (self._cursor + 1) % len(healthy)
        return ordered


def _filter_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _create_app(pool: ReplicaPool) -> FastAPI:
    app = FastAPI(title="vLLM hotload proxy")

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.client = httpx.AsyncClient(timeout=None, follow_redirects=False)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.client.aclose()

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        replicas = await pool.healthy_replicas(app.state.client)
        return JSONResponse(
            {
                "healthy_replicas": [replica.node for replica in replicas],
                "replica_count": len(replicas),
            }
        )

    @app.api_route(
        "/managed/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
    )
    async def reject_managed(path: str) -> JSONResponse:
        del path
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    @app.api_route("/v1", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_v1(request: Request, path: str = "") -> StreamingResponse:
        candidates = await pool.healthy_replicas(app.state.client)
        if not candidates:
            raise HTTPException(status_code=503, detail="No healthy replicas available")

        query = request.url.query
        request_headers = _filter_headers(dict(request.headers.items()))
        body = await request.body()
        target_path = "/v1" if not path else f"/v1/{path}"

        last_error: str | None = None
        for replica in candidates:
            target_url = f"{replica.base_url}{target_path}"
            if query:
                target_url = f"{target_url}?{query}"
            upstream_request = app.state.client.build_request(
                request.method,
                target_url,
                content=body,
                headers=request_headers,
            )
            try:
                upstream = await app.state.client.send(upstream_request, stream=True)
                response_headers = _filter_headers(upstream.headers)
                return StreamingResponse(
                    upstream.aiter_raw(),
                    status_code=upstream.status_code,
                    headers=response_headers,
                    background=BackgroundTask(upstream.aclose),
                )
            except httpx.HTTPError as exc:
                last_error = str(exc)

        raise HTTPException(
            status_code=503,
            detail=last_error or "Healthy replicas became unavailable during proxying",
        )

    return app


def create_app(state_file: str | Path) -> FastAPI:
    return _create_app(
        ReplicaPool(lambda: _load_replicas_from_state_file(Path(state_file)))
    )


def create_app_from_replicas(
    replicas: Iterable[ProxyReplica | dict[str, str]],
) -> FastAPI:
    replica_list = [_coerce_replica(replica) for replica in replicas]
    return _create_app(ReplicaPool(lambda: list(replica_list)))


def run_proxy(state_file: str | Path, host: str, port: int) -> None:
    uvicorn.run(create_app(state_file), host=host, port=port, log_level="info")
