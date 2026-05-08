# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from enum import Enum
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)

_SLEEP_MODE_HINT = "start server with --enable-sleep-mode"
_WEIGHT_TRANSFER_HINT = (
    'start server with --weight-transfer-config \'{"backend":"ipc"}\' '
    "or another supported backend"
)
_WAKE_TAGS = {"weights", "kv_cache"}


class ManagedWeightSyncError(Exception):
    def __init__(
        self,
        status_code: HTTPStatus,
        error: str,
        hint: str,
    ) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.hint = hint


def _engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _get_args(request: Request) -> Namespace | None:
    return getattr(request.app.state, "args", None)


def _ok_response(content: dict[str, Any] | None = None) -> JSONResponse:
    body: dict[str, Any] = {"ok": True}
    if content:
        body.update(content)
    return JSONResponse(content=body, status_code=HTTPStatus.OK.value)


def _error_response(
    status_code: HTTPStatus,
    error: str,
    hint: str,
    content: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"ok": False, "error": error, "hint": hint}
    if content:
        body.update(content)
    return JSONResponse(content=body, status_code=status_code.value)


def _expected_error_response(error: ManagedWeightSyncError) -> JSONResponse:
    return _error_response(error.status_code, error.error, error.hint)


def _unexpected_error_response(operation: str, error: Exception) -> JSONResponse:
    logger.exception("Managed weight-sync %s failed", operation)
    return _error_response(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        f"Unexpected internal failure while trying to {operation}.",
        "See server logs for details.",
    )


async def _read_json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as err:
        raise ManagedWeightSyncError(
            HTTPStatus.BAD_REQUEST,
            f"Invalid JSON request body: {err}",
            "Send a JSON object request body.",
        ) from err

    if body is None:
        return {}
    if not isinstance(body, dict):
        raise ManagedWeightSyncError(
            HTTPStatus.BAD_REQUEST,
            "Invalid JSON request body: expected an object.",
            "Send a JSON object request body.",
        )
    return body


def _validate_sleep_level(level: Any) -> int:
    if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2):
        raise ManagedWeightSyncError(
            HTTPStatus.BAD_REQUEST,
            "Invalid sleep level: expected 1 or 2.",
            "Send {'level': 1} or {'level': 2}.",
        )
    return level


def _validate_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ManagedWeightSyncError(
            HTTPStatus.BAD_REQUEST,
            f"Invalid {field_name}: expected a boolean.",
            f"Send '{field_name}' as true or false.",
        )
    return value


def _validate_wake_tags(tags: Any) -> list[str] | None:
    if tags is None:
        return None
    if not isinstance(tags, list):
        raise ManagedWeightSyncError(
            HTTPStatus.BAD_REQUEST,
            "Invalid wake tags: expected a list or null.",
            "Supported tags are 'weights' and 'kv_cache'.",
        )

    invalid_tags = [
        tag for tag in tags if not isinstance(tag, str) or tag not in _WAKE_TAGS
    ]
    if invalid_tags:
        raise ManagedWeightSyncError(
            HTTPStatus.BAD_REQUEST,
            f"Invalid wake tags: {invalid_tags}.",
            "Supported tags are 'weights' and 'kv_cache'.",
        )
    return tags


def _get_vllm_config(request: Request) -> Any | None:
    if vllm_config := getattr(request.app.state, "vllm_config", None):
        return vllm_config
    engine = getattr(request.app.state, "engine_client", None)
    return getattr(engine, "vllm_config", None)


def _get_parallel_config(request: Request) -> Any | None:
    vllm_config = _get_vllm_config(request)
    return getattr(vllm_config, "parallel_config", None)


def _get_world_size(request: Request) -> int | None:
    parallel_config = _get_parallel_config(request)
    if parallel_config is None:
        return None
    return getattr(
        parallel_config,
        "world_size_across_dp",
        getattr(parallel_config, "world_size", None),
    )


def _get_weight_transfer_config(request: Request) -> Any | None:
    args = _get_args(request)
    if args is not None and hasattr(args, "weight_transfer_config"):
        return args.weight_transfer_config
    vllm_config = _get_vllm_config(request)
    return getattr(vllm_config, "weight_transfer_config", None)


def _get_sleep_mode_enabled(request: Request) -> bool | None:
    args = _get_args(request)
    if args is not None and hasattr(args, "enable_sleep_mode"):
        return bool(args.enable_sleep_mode)

    vllm_config = _get_vllm_config(request)
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is not None and hasattr(model_config, "enable_sleep_mode"):
        return bool(model_config.enable_sleep_mode)
    return None


def _ensure_weight_transfer_configured(request: Request) -> None:
    if _get_weight_transfer_config(request) is None:
        raise ManagedWeightSyncError(
            HTTPStatus.CONFLICT,
            "Weight transfer is not configured.",
            _WEIGHT_TRANSFER_HINT,
        )


def _ensure_sleep_mode_enabled(request: Request) -> None:
    if _get_sleep_mode_enabled(request) is not True:
        raise ManagedWeightSyncError(
            HTTPStatus.CONFLICT,
            "Sleep mode is not enabled.",
            _SLEEP_MODE_HINT,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(val) for key, val in asdict(value).items()}  # pyright: ignore[reportArgumentType]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "backend"):
        return {"backend": _jsonable(getattr(value, "backend"))}
    return str(value)


def _served_model_names(args: Namespace | None) -> list[str] | None:
    if args is None:
        return None

    names = getattr(args, "served_model_name", None)
    if names is None:
        model = getattr(args, "model", None) or getattr(args, "model_tag", None)
        return [model] if model is not None else None
    if isinstance(names, str):
        return [names]
    return list(names)


async def _safe_is_sleeping(request: Request) -> bool | None:
    try:
        return await _engine_client(request).is_sleeping()
    except Exception:
        logger.debug("Unable to fetch managed weight-sync sleep state", exc_info=True)
        return None


async def _pause(request: Request) -> None:
    await _engine_client(request).pause_generation(mode="abort", clear_cache=True)


async def _resume(request: Request) -> None:
    await _engine_client(request).resume_generation()


async def _sleep(request: Request, level: int) -> None:
    _ensure_sleep_mode_enabled(request)
    await _engine_client(request).sleep(level=level, mode="abort")


async def _wake(request: Request, tags: list[str] | None) -> None:
    _ensure_sleep_mode_enabled(request)
    await _engine_client(request).wake_up(tags)


def _step_error_response(
    status_code: HTTPStatus,
    failed_step: str,
    steps: list[dict[str, Any]],
    error: str,
    hint: str,
) -> JSONResponse:
    return JSONResponse(
        content={
            "ok": False,
            "failed_step": failed_step,
            "steps": steps,
            "error": error,
            "hint": hint,
        },
        status_code=status_code.value,
    )


router = APIRouter()


@router.get("/status")
async def status(raw_request: Request) -> JSONResponse:
    args = _get_args(raw_request)
    prefix = getattr(raw_request.app.state, "managed_weight_sync_prefix", "/managed")
    return _ok_response(
        {
            "managed_weight_sync": True,
            "prefix": prefix,
            "dev_mode": envs.VLLM_SERVER_DEV_MODE,
            "served_model_names": _served_model_names(args),
            "load_format": _jsonable(getattr(args, "load_format", None)),
            "weight_transfer_config": _jsonable(
                _get_weight_transfer_config(raw_request)
            ),
            "sleep_mode_enabled": _get_sleep_mode_enabled(raw_request),
            "is_sleeping": await _safe_is_sleeping(raw_request),
            "world_size": _get_world_size(raw_request),
        }
    )


@router.get("/world_size")
async def world_size(raw_request: Request) -> JSONResponse:
    return _ok_response({"world_size": _get_world_size(raw_request)})


@router.post("/pause")
async def pause(raw_request: Request) -> JSONResponse:
    try:
        await _pause(raw_request)
        return _ok_response({"paused": True})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception as err:
        return _unexpected_error_response("pause generation", err)


@router.post("/resume")
async def resume(raw_request: Request) -> JSONResponse:
    try:
        await _resume(raw_request)
        return _ok_response({"resumed": True})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception as err:
        return _unexpected_error_response("resume generation", err)


@router.post("/sleep")
async def sleep(raw_request: Request) -> JSONResponse:
    try:
        body = await _read_json_body(raw_request)
        if "level" not in body:
            raise ManagedWeightSyncError(
                HTTPStatus.BAD_REQUEST,
                "Missing 'level' in request body.",
                "Send {'level': 1} or {'level': 2}.",
            )
        level = _validate_sleep_level(body["level"])
        await _sleep(raw_request, level)
        sleeping = await _safe_is_sleeping(raw_request)
        return _ok_response(
            {"level": level, "sleeping": sleeping if sleeping is not None else True}
        )
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception as err:
        return _unexpected_error_response("sleep the engine", err)


@router.post("/wake")
async def wake(raw_request: Request) -> JSONResponse:
    try:
        body = await _read_json_body(raw_request)
        tags = _validate_wake_tags(body.get("tags"))
        await _wake(raw_request, tags)
        return _ok_response({"woke": True, "tags": tags})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception as err:
        return _unexpected_error_response("wake the engine", err)


@router.post("/init_weight_transfer")
async def init_weight_transfer(raw_request: Request) -> JSONResponse:
    try:
        _ensure_weight_transfer_configured(raw_request)
        body = await _read_json_body(raw_request)
        init_info = body.get("init_info")
        if init_info is None:
            raise ManagedWeightSyncError(
                HTTPStatus.BAD_REQUEST,
                "Missing 'init_info' in request body.",
                "Send {'init_info': {}} for IPC or backend-specific init info.",
            )
        if not isinstance(init_info, dict):
            raise ManagedWeightSyncError(
                HTTPStatus.BAD_REQUEST,
                "Invalid 'init_info': expected an object.",
                "Send {'init_info': {}} for IPC or backend-specific init info.",
            )
        from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

        await _engine_client(raw_request).init_weight_transfer_engine(
            WeightTransferInitRequest(init_info=init_info)
        )
        return _ok_response({"initialized": True})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception as err:
        return _unexpected_error_response("initialize weight transfer", err)


@router.post("/prepare_weight_update")
async def prepare_weight_update(raw_request: Request) -> JSONResponse:
    steps: list[dict[str, Any]] = []
    try:
        body = await _read_json_body(raw_request)
        sleep_level = body.get("sleep_level")
        if sleep_level is not None:
            sleep_level = _validate_sleep_level(sleep_level)
        wake_weights = _validate_bool(body.get("wake_weights"), "wake_weights", False)

        await _pause(raw_request)
        steps.append({"step": "pause", "ok": True})

        if sleep_level is not None:
            try:
                await _sleep(raw_request, sleep_level)
                steps.append({"step": "sleep", "level": sleep_level, "ok": True})
            except ManagedWeightSyncError as err:
                steps.append({"step": "sleep", "ok": False, "error": err.error})
                return _step_error_response(
                    err.status_code, "sleep", steps, err.error, err.hint
                )
            except Exception:
                logger.exception("Managed weight-sync sleep step failed")
                steps.append(
                    {
                        "step": "sleep",
                        "ok": False,
                        "error": "Unexpected internal failure.",
                    }
                )
                return _step_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "sleep",
                    steps,
                    "Unexpected internal failure while sleeping the engine.",
                    "See server logs for details.",
                )

        if wake_weights:
            try:
                await _wake(raw_request, ["weights"])
                steps.append({"step": "wake_weights", "ok": True})
            except ManagedWeightSyncError as err:
                steps.append({"step": "wake_weights", "ok": False, "error": err.error})
                return _step_error_response(
                    err.status_code, "wake_weights", steps, err.error, err.hint
                )
            except Exception:
                logger.exception("Managed weight-sync wake_weights step failed")
                steps.append(
                    {
                        "step": "wake_weights",
                        "ok": False,
                        "error": "Unexpected internal failure.",
                    }
                )
                return _step_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "wake_weights",
                    steps,
                    "Unexpected internal failure while waking model weights.",
                    "See server logs for details.",
                )

        return _ok_response({"steps": steps})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception:
        failed_step = "pause" if not steps else steps[-1]["step"]
        logger.exception("Managed weight-sync prepare step failed")
        steps.append(
            {
                "step": failed_step,
                "ok": False,
                "error": "Unexpected internal failure.",
            }
        )
        return _step_error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            failed_step,
            steps,
            "Unexpected internal failure while preparing a weight update.",
            "See server logs for details.",
        )


@router.post("/load_weights")
async def load_weights(raw_request: Request) -> JSONResponse:
    try:
        _ensure_weight_transfer_configured(raw_request)
        body = await _read_json_body(raw_request)

        model_path = body.get("model_path")
        if not model_path or not isinstance(model_path, str):
            raise ManagedWeightSyncError(
                HTTPStatus.BAD_REQUEST,
                "Missing or invalid 'model_path'.",
                "Send {'model_path': '/path/to/checkpoint'}.",
            )

        kwargs = {
            "model_path": model_path,
            "load_format": body.get("load_format", "safetensors"),
            "safetensors_load_strategy": body.get(
                "safetensors_load_strategy"
            ),
            "model_loader_extra_config": body.get("model_loader_extra_config"),
        }

        await _engine_client(raw_request).load_weights_from_path(kwargs)
        return _ok_response({"model_path": model_path})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception as err:
        return _unexpected_error_response("load_weights", err)


@router.post("/finish_weight_update")
async def finish_weight_update(raw_request: Request) -> JSONResponse:
    steps: list[dict[str, Any]] = []
    try:
        body = await _read_json_body(raw_request)
        wake_kv_cache = _validate_bool(
            body.get("wake_kv_cache"), "wake_kv_cache", False
        )
        resume_generation = _validate_bool(body.get("resume"), "resume", False)

        if wake_kv_cache:
            try:
                await _wake(raw_request, ["kv_cache"])
                steps.append({"step": "wake_kv_cache", "ok": True})
            except ManagedWeightSyncError as err:
                steps.append({"step": "wake_kv_cache", "ok": False, "error": err.error})
                return _step_error_response(
                    err.status_code, "wake_kv_cache", steps, err.error, err.hint
                )

        if resume_generation:
            try:
                await _resume(raw_request)
                steps.append({"step": "resume", "ok": True})
            except Exception:
                logger.exception("Managed weight-sync resume step failed")
                steps.append(
                    {
                        "step": "resume",
                        "ok": False,
                        "error": "Unexpected internal failure.",
                    }
                )
                return _step_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "resume",
                    steps,
                    "Unexpected internal failure while resuming generation.",
                    "See server logs for details.",
                )

        return _ok_response({"steps": steps})
    except ManagedWeightSyncError as err:
        return _expected_error_response(err)
    except Exception:
        failed_step = "resume" if steps else "wake_kv_cache"
        logger.exception("Managed weight-sync finish step failed")
        steps.append(
            {
                "step": failed_step,
                "ok": False,
                "error": "Unexpected internal failure.",
            }
        )
        return _step_error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            failed_step,
            steps,
            "Unexpected internal failure while finishing a weight update.",
            "See server logs for details.",
        )


def _normalize_prefix(prefix: str) -> str:
    if not prefix.startswith("/"):
        raise ValueError("managed weight-sync prefix must start with '/'")
    if prefix != "/":
        prefix = prefix.rstrip("/")
    return prefix


def register_managed_weight_sync_routes(
    app: FastAPI,
    engine_client: EngineClient | None = None,
    args: Namespace | None = None,
    prefix: str = "/managed",
) -> None:
    prefix = _normalize_prefix(prefix)
    if engine_client is not None:
        app.state.engine_client = engine_client
    if args is not None:
        app.state.args = args
    app.state.managed_weight_sync_prefix = prefix
    app.include_router(router, prefix=prefix)


def attach_router(app: FastAPI) -> None:
    args = getattr(app.state, "args", None)
    if args is None or not getattr(args, "managed_weight_sync", False):
        return
    if (
        getattr(args, "managed_weight_sync_require_dev_mode", True)
        and not envs.VLLM_SERVER_DEV_MODE
    ):
        raise RuntimeError(
            "--managed-weight-sync requires VLLM_SERVER_DEV_MODE=1. "
            "Managed weight-sync endpoints are unsafe and must not be exposed "
            "to untrusted networks."
        )

    logger.warning(
        "WARNING: managed weight-sync endpoints are enabled. "
        "Do not expose this server to an untrusted network."
    )
    register_managed_weight_sync_routes(
        app,
        args=args,
        prefix=getattr(args, "managed_weight_sync_prefix", "/managed"),
    )
