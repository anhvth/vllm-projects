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
from vllm_hotload.ray_vllm import main as ray_vllm_main
from vllm_hotload.proxy import create_app


def capture_main_output(args: list[str]) -> dict[str, Any]:
    with patch("sys.stdout.write") as write_mock:
        main(args)
    return json.loads("".join(call.args[0] for call in write_mock.call_args_list))


class HotloadCtlTests(unittest.TestCase):
    def test_start_dry_run_emits_multi_app_serve_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = capture_main_output(
                [
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
            )

        self.assertEqual(payload["public_base_url"], "http://head-node:8000/v1")
        self.assertEqual(
            payload["serve_config_path"],
            str(Path(tmpdir) / ".hotloadctl" / "serve_config.yaml"),
        )
        self.assertIn("serve run", payload["serve_run_command"])
        self.assertNotIn("tmux", json.dumps(payload))

        applications = payload["serve_config"]["applications"]
        public_apps = [
            app
            for app in applications
            if app["import_path"] == "vllm_hotload.ray_serve_app:build_public_proxy_app"
        ]
        replica_apps = [
            app
            for app in applications
            if app["import_path"] == "vllm_hotload.ray_serve_app:build_vllm_replica_app"
        ]

        self.assertEqual(len(public_apps), 1)
        self.assertEqual(public_apps[0]["route_prefix"], "/")
        self.assertEqual(len(replica_apps), 3)
        self.assertTrue(
            all(
                app["route_prefix"].startswith("/_hotloadctl/replicas/")
                for app in replica_apps
            )
        )
        self.assertTrue(
            all(
                app["deployments"][0]["name"] == "ManagedVLLMReplica"
                for app in replica_apps
            )
        )
        self.assertTrue(
            all(
                app["deployments"][0]["ray_actor_options"]["num_gpus"] == 8
                for app in replica_apps
            )
        )
        self.assertFalse(replica_apps[0]["args"]["fast_loading_ram"])

    def test_ray_vllm_serve_dry_run_maps_vllm_like_args_and_ram_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.stdout.write") as write_mock:
                ray_vllm_main(
                    [
                        "serve",
                        "/models/qwen3",
                        "--nodes",
                        "node0,node1",
                        "--tensor-parallel-size",
                        "4",
                        "--served-model-name",
                        "qwen3-1.7b",
                        "--port",
                        "9000",
                        "--public-host",
                        "head-node",
                        "--workspace-dir",
                        tmpdir,
                        "--state-file",
                        str(Path(tmpdir) / "state.json"),
                        "--trust-remote-code",
                        "--fast-loading",
                        "ram",
                        "--ram-stage-num-workers",
                        "3",
                        "--dry-run",
                    ]
                )
            payload = json.loads(
                "".join(call.args[0] for call in write_mock.call_args_list)
            )

        replica_apps = [
            app
            for app in payload["serve_config"]["applications"]
            if app["import_path"] == "vllm_hotload.ray_serve_app:build_vllm_replica_app"
        ]

        self.assertEqual(payload["public_base_url"], "http://head-node:9000/v1")
        self.assertEqual(len(replica_apps), 2)
        self.assertEqual(
            replica_apps[0]["deployments"][0]["ray_actor_options"]["num_gpus"],
            4,
        )
        self.assertTrue(replica_apps[0]["args"]["fast_loading_ram"])
        self.assertEqual(replica_apps[0]["args"]["ram_stage_num_workers"], 3)
        self.assertTrue(replica_apps[0]["args"]["trust_remote_code"])

    def test_start_preflight_surfaces_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "vllm_hotload.hotloadctl._run_cli",
                    side_effect=RuntimeError(
                        "ray status failed due to a Ray version mismatch.\ncluster=2.55.1 local=2.54.0"
                    ),
                ),
                patch(
                    "vllm_hotload.hotloadctl.write_serve_config"
                ) as write_config_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    main(
                        [
                            "start",
                            "--nodes",
                            "localhost",
                            "--gpus-per-replica",
                            "1",
                            "--model-path",
                            "/models/dummy",
                            "--served-model-name",
                            "qwen3-1.7b",
                            "--workspace-dir",
                            tmpdir,
                            "--state-file",
                            str(Path(tmpdir) / "state.json"),
                        ]
                    )

        write_config_mock.assert_not_called()

    def test_collect_cluster_status_aggregates_replica_health(self) -> None:
        state = ClusterState(
            workspace_dir="/workspace",
            ray_address="auto",
            dashboard_address="http://localhost:8265",
            public_base_url="http://head-node:8000/v1",
            public_host="head-node",
            public_port=8000,
            proxy_host="0.0.0.0",
            public_app_name="vllm-hotload-public",
            public_route_prefix="/",
            serve_config_path="/workspace/.hotloadctl/serve_config.yaml",
            served_model_name="qwen3-1.7b",
            start_model_path="/models/dummy",
            gpus_per_replica=2,
            fast_loading_ram=False,
            ram_stage_num_workers=8,
            ram_stage_copy_delay=0.0,
            ram_stage_small_file_threshold=10_000_000,
            nodes=["node0", "node1"],
            replicas=[
                ReplicaState(
                    node="node0",
                    route_prefix="/_hotloadctl/replicas/1-node0",
                    base_url="http://head-node:8000/_hotloadctl/replicas/1-node0",
                    v1_url="http://head-node:8000/_hotloadctl/replicas/1-node0/v1",
                    managed_url="http://head-node:8000/_hotloadctl/replicas/1-node0/managed",
                    app_name="vllm-hotload-replica-1-node0",
                    gpus_per_replica=2,
                ),
                ReplicaState(
                    node="node1",
                    route_prefix="/_hotloadctl/replicas/2-node1",
                    base_url="http://head-node:8000/_hotloadctl/replicas/2-node1",
                    v1_url="http://head-node:8000/_hotloadctl/replicas/2-node1/v1",
                    managed_url="http://head-node:8000/_hotloadctl/replicas/2-node1/managed",
                    app_name="vllm-hotload-replica-2-node1",
                    gpus_per_replica=2,
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
            if (
                url
                == "http://head-node:8000/_hotloadctl/replicas/1-node0/managed/status"
            ):
                return FakeResponse({"sleeping": False, "model": "qwen3-1.7b"})
            if url == "http://head-node:8000/_hotloadctl/replicas/1-node0/v1/models":
                return FakeResponse({"data": [{"id": "qwen3-1.7b"}]})
            raise requests.RequestException("node1 down")

        with (
            patch("requests.request", side_effect=fake_request),
            patch(
                "vllm_hotload.hotloadctl._serve_status",
                return_value={
                    "applications": {
                        "vllm-hotload-public": {"status": "RUNNING", "message": ""}
                    }
                },
            ),
        ):
            status = collect_cluster_status(state, 5.0)

        self.assertEqual(status["public_base_url"], "http://head-node:8000/v1")
        self.assertEqual(status["replicas"][0]["health"], "healthy")
        self.assertEqual(status["replicas"][0]["managed_status"]["model"], "qwen3-1.7b")
        self.assertEqual(status["replicas"][1]["health"], "unhealthy")
        self.assertIsNone(status["replicas"][1]["managed_status"])
        self.assertEqual(
            status["serve_applications"]["vllm-hotload-public"]["status"], "RUNNING"
        )

    def test_push_dry_run_uses_replica_route_prefix_managed_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "workspace_dir": "/workspace",
                        "ray_address": "auto",
                        "dashboard_address": "http://localhost:8265",
                        "public_base_url": "http://head-node:8000/v1",
                        "public_host": "head-node",
                        "public_port": 8000,
                        "proxy_host": "0.0.0.0",
                        "public_app_name": "vllm-hotload-public",
                        "public_route_prefix": "/",
                        "serve_config_path": "/workspace/.hotloadctl/serve_config.yaml",
                        "served_model_name": "qwen3-1.7b",
                        "start_model_path": "/models/dummy",
                        "gpus_per_replica": 2,
                        "fast_loading_ram": True,
                        "ram_stage_num_workers": 5,
                        "ram_stage_copy_delay": 0.25,
                        "ram_stage_small_file_threshold": 1024,
                        "nodes": ["node0"],
                        "replicas": [
                            {
                                "node": "node0",
                                "route_prefix": "/_hotloadctl/replicas/1-node0",
                                "base_url": "http://head-node:8000/_hotloadctl/replicas/1-node0",
                                "v1_url": "http://head-node:8000/_hotloadctl/replicas/1-node0/v1",
                                "managed_url": "http://head-node:8000/_hotloadctl/replicas/1-node0/managed",
                                "app_name": "vllm-hotload-replica-1-node0",
                                "gpus_per_replica": 2,
                            }
                        ],
                    }
                )
            )

            payload = capture_main_output(
                [
                    "push",
                    "/checkpoints/step-100",
                    "--state-file",
                    str(state_file),
                    "--dry-run",
                ]
            )

        self.assertEqual(
            payload["public_verify_models"], "http://head-node:8000/v1/models"
        )
        self.assertEqual(
            payload["operations"][0]["prepare_weight_update"]["url"],
            "http://head-node:8000/_hotloadctl/replicas/1-node0/managed/prepare_weight_update",
        )
        self.assertEqual(
            payload["operations"][0]["load_weights"]["url"],
            "http://head-node:8000/_hotloadctl/replicas/1-node0/managed/load_weights",
        )
        self.assertEqual(
            payload["operations"][0]["load_weights"]["payload"],
            {
                "model_path": "/checkpoints/step-100",
                "load_format": "safetensors",
                "safetensors_load_strategy": "ram_stage",
                "model_loader_extra_config": {
                    "ram_stage_num_workers": 5,
                    "ram_stage_copy_delay": 0.25,
                    "ram_stage_small_file_threshold": 1024,
                },
            },
        )
        self.assertNotIn("push_command", payload["operations"][0])

    def test_stop_refuses_shutdown_when_foreign_apps_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "workspace_dir": "/workspace",
                        "ray_address": "auto",
                        "dashboard_address": "http://localhost:8265",
                        "public_base_url": "http://head-node:8000/v1",
                        "public_host": "head-node",
                        "public_port": 8000,
                        "proxy_host": "0.0.0.0",
                        "public_app_name": "vllm-hotload-public",
                        "public_route_prefix": "/",
                        "serve_config_path": "/workspace/.hotloadctl/serve_config.yaml",
                        "served_model_name": "qwen3-1.7b",
                        "start_model_path": "/models/dummy",
                        "gpus_per_replica": 2,
                        "nodes": ["node0"],
                        "replicas": [
                            {
                                "node": "node0",
                                "route_prefix": "/_hotloadctl/replicas/1-node0",
                                "base_url": "http://head-node:8000/_hotloadctl/replicas/1-node0",
                                "v1_url": "http://head-node:8000/_hotloadctl/replicas/1-node0/v1",
                                "managed_url": "http://head-node:8000/_hotloadctl/replicas/1-node0/managed",
                                "app_name": "vllm-hotload-replica-1-node0",
                                "gpus_per_replica": 2,
                            }
                        ],
                    }
                )
            )

            with patch(
                "vllm_hotload.hotloadctl._serve_status",
                return_value={
                    "applications": {
                        "vllm-hotload-public": {"status": "RUNNING"},
                        "vllm-hotload-replica-1-node0": {"status": "RUNNING"},
                        "other-app": {"status": "RUNNING"},
                    }
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "non-hotload applications"):
                    main(["stop", "--state-file", str(state_file)])

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
