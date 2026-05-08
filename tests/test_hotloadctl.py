from __future__ import annotations

# ruff: noqa: E402
# pyright: reportMissingImports=false

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import requests
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_hotload.hotloadctl import (
    ClusterState,
    ReplicaState,
    collect_cluster_status,
    main,
)
from vllm_hotload.proxy import create_app


class HotloadCtlTests(unittest.TestCase):
    def test_start_dry_run_generates_replica_and_proxy_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.json"
            args = [
                "start",
                "--nodes",
                "node0,node1,node2",
                "--gpus-per-replica",
                "8",
                "--model-path",
                "/models/dummy",
                "--served-model-name",
                "qwen3-1.7b",
                "--workspace-dir",
                tmpdir,
                "--state-file",
                str(Path(tmpdir) / "state.json"),
                "--public-host",
                "head-node",
                "--dry-run",
            ]
            with patch("sys.stdout.write") as write_mock:
                main(args)
            output = "".join(call.args[0] for call in write_mock.call_args_list)
            output_path.write_text(output)
            payload = json.loads(output_path.read_text())

        self.assertEqual(payload["public_base_url"], "http://head-node:8000/v1")
        expected_ray_bootstrap = (
            f"env RAY_BIN={Path(tmpdir) / '.venv' / 'bin' / 'ray'} "
            f"{Path('~/images/06-start-ray.sh').expanduser()} -n 3"
        )
        self.assertEqual(payload["ray_bootstrap_command"], expected_ray_bootstrap)
        self.assertIn(
            "tmux new-session -d -s vllm-hotload-node0-serve",
            payload["commands"]["node0"],
        )
        self.assertIn("--load-format dummy", payload["commands"]["node0"])
        self.assertIn("--managed-weight-sync", payload["commands"]["node0"])
        self.assertIn("--enable-sleep-mode", payload["commands"]["node0"])
        self.assertIn("--weight-transfer-config", payload["commands"]["node0"])
        self.assertIn('{"backend":"ipc"}', payload["commands"]["node0"])
        self.assertIn(
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", payload["commands"]["node0"]
        )
        self.assertIn(
            "tmux new-session -d -s vllm-hotload-proxy", payload["proxy_command"]
        )
        self.assertIn("hotloadctl _serve-proxy", payload["proxy_command"])

    def test_start_dry_run_skips_ray_bootstrap_for_single_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.stdout.write") as write_mock:
                main(
                    [
                        "start",
                        "--nodes",
                        "node0",
                        "--gpus-per-replica",
                        "8",
                        "--model-path",
                        "/models/dummy",
                        "--served-model-name",
                        "qwen3-1.7b",
                        "--workspace-dir",
                        tmpdir,
                        "--state-file",
                        str(Path(tmpdir) / "state.json"),
                        "--dry-run",
                    ]
                )
            payload = json.loads(
                "".join(call.args[0] for call in write_mock.call_args_list)
            )

        self.assertIsNone(payload["ray_bootstrap_command"])

    def test_collect_cluster_status_aggregates_replica_health(self) -> None:
        state = ClusterState(
            workspace_dir="/workspace",
            public_base_url="http://head-node:8000/v1",
            public_host="head-node",
            public_port=8000,
            proxy_host="0.0.0.0",
            proxy_session_name="vllm-hotload-proxy",
            base_port=8100,
            served_model_name="qwen3-1.7b",
            start_model_path="/models/dummy",
            gpus_per_replica=2,
            ray_address="auto",
            nodes=["node0", "node1"],
            replicas=[
                ReplicaState(
                    node="node0",
                    base_url="http://node0:8100",
                    v1_url="http://node0:8100/v1",
                    managed_url="http://node0:8100/managed",
                    session_name="s0",
                    gpus_per_replica=2,
                    gpu_devices=[0, 1],
                ),
                ReplicaState(
                    node="node1",
                    base_url="http://node1:8100",
                    v1_url="http://node1:8100/v1",
                    managed_url="http://node1:8100/managed",
                    session_name="s1",
                    gpus_per_replica=2,
                    gpu_devices=[0, 1],
                ),
            ],
        )

        class FakeResponse:
            def __init__(self, payload: dict[str, object], status_code: int = 200):
                self._payload = payload
                self.status_code = status_code
                self.content = b"x"

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError("boom")

            def json(self) -> dict[str, object]:
                return self._payload

        def fake_request(method: str, url: str, json: object, timeout: float):
            del method, json, timeout
            if url == "http://node0:8100/managed/status":
                return FakeResponse({"sleeping": False, "model": "qwen3-1.7b"})
            if url == "http://node0:8100/v1/models":
                return FakeResponse({"data": [{"id": "qwen3-1.7b"}]})
            raise requests.RequestException("node1 down")

        with patch(
            "vllm_hotload.hotloadctl.requests.request", side_effect=fake_request
        ):
            status = collect_cluster_status(state, 5.0)

        self.assertEqual(status["public_base_url"], "http://head-node:8000/v1")
        self.assertEqual(status["replicas"][0]["health"], "healthy")
        self.assertEqual(status["replicas"][0]["managed_status"]["model"], "qwen3-1.7b")
        self.assertEqual(status["replicas"][1]["health"], "unhealthy")
        self.assertIsNone(status["replicas"][1]["managed_status"])

    def test_push_dry_run_uses_saved_ray_state_and_public_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "workspace_dir": "/workspace",
                        "ray_address": "auto",
                        "public_base_url": "http://head-node:8000/v1",
                        "public_host": "head-node",
                        "public_port": 8000,
                        "proxy_host": "0.0.0.0",
                        "proxy_session_name": "vllm-hotload-proxy",
                        "base_port": 8100,
                        "served_model_name": "qwen3-1.7b",
                        "start_model_path": "/models/dummy",
                        "gpus_per_replica": 2,
                        "nodes": ["node0"],
                        "replicas": [
                            {
                                "node": "node0",
                                "base_url": "http://node0:8100",
                                "v1_url": "http://node0:8100/v1",
                                "managed_url": "http://node0:8100/managed",
                                "session_name": "vllm-hotload-node0-serve",
                                "gpus_per_replica": 2,
                                "gpu_devices": [0, 1],
                            }
                        ],
                    }
                )
            )

            with patch("sys.stdout.write") as write_mock:
                main(
                    [
                        "push",
                        "/checkpoints/step-100",
                        "--state-file",
                        str(state_file),
                        "--dry-run",
                    ]
                )

            payload = json.loads(
                "".join(call.args[0] for call in write_mock.call_args_list)
            )

        self.assertEqual(
            payload["public_verify_models"], "http://head-node:8000/v1/models"
        )
        self.assertTrue(
            payload["operations"][0]["push_command"].startswith("cd /workspace &&")
        )
        self.assertIn(
            "--base-url http://127.0.0.1:8100",
            payload["operations"][0]["push_command"],
        )

    def test_stop_dry_run_uses_tmux_commands_without_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "workspace_dir": "/workspace",
                        "ray_address": "auto",
                        "public_base_url": "http://head-node:8000/v1",
                        "public_host": "head-node",
                        "public_port": 8000,
                        "proxy_host": "0.0.0.0",
                        "proxy_session_name": "vllm-hotload-proxy",
                        "base_port": 8100,
                        "served_model_name": "qwen3-1.7b",
                        "start_model_path": "/models/dummy",
                        "gpus_per_replica": 2,
                        "nodes": ["node0"],
                        "replicas": [
                            {
                                "node": "node0",
                                "base_url": "http://node0:8100",
                                "v1_url": "http://node0:8100/v1",
                                "managed_url": "http://node0:8100/managed",
                                "session_name": "vllm-hotload-node0-serve",
                                "gpus_per_replica": 2,
                                "gpu_devices": [0, 1],
                            }
                        ],
                    }
                )
            )

            with patch("sys.stdout.write") as write_mock:
                main(["stop", "--state-file", str(state_file), "--dry-run"])

            payload = json.loads(
                "".join(call.args[0] for call in write_mock.call_args_list)
            )

        self.assertEqual(
            payload["commands"]["node0"],
            "tmux kill-session -t vllm-hotload-node0-serve",
        )
        self.assertEqual(
            payload["proxy_command"],
            "tmux kill-session -t vllm-hotload-proxy",
        )

    def test_proxy_rejects_managed_and_round_robins_healthy_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "replicas": [
                            {
                                "node": "node0",
                                "base_url": "http://node0:8100",
                                "v1_url": "http://node0:8100/v1",
                                "managed_url": "http://node0:8100/managed",
                            },
                            {
                                "node": "node1",
                                "base_url": "http://node1:8100",
                                "v1_url": "http://node1:8100/v1",
                                "managed_url": "http://node1:8100/managed",
                            },
                        ]
                    }
                )
            )
            app = create_app(state_file)
            upstream_urls: list[str] = []

            async def fake_is_healthy(self, client, replica):  # type: ignore[no-untyped-def]
                del self, client, replica
                return True

            async def fake_send(request, stream=False):  # type: ignore[no-untyped-def]
                del stream
                upstream_urls.append(str(request.url))
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    stream=httpx.ByteStream(b'{"data":[{"id":"qwen3-1.7b"}]}'),
                    request=request,
                )

            with patch(
                "vllm_hotload.proxy.ReplicaPool._is_healthy", new=fake_is_healthy
            ):
                with TestClient(app) as client:
                    cast(Any, client.app).state.client.send = AsyncMock(
                        side_effect=fake_send
                    )
                    managed_response = client.get("/managed/status")
                    first_response = client.get("/v1/models")
                    second_response = client.get("/v1/models")

        self.assertEqual(managed_response.status_code, 404)
        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(
            upstream_urls,
            [
                "http://node0:8100/v1/models",
                "http://node1:8100/v1/models",
            ],
        )


if __name__ == "__main__":
    unittest.main()
