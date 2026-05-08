from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
BUILD_SCRIPT = ROOT / "build.sh"
E2E_SCRIPT = ROOT / "PR" / "run_hotload_vllm_e2e.sh"
SHIM_SCRIPT = (
    ROOT / "vllm_patch" / "examples" / "managed_weight_sync" / "hf_push_ipc.py"
)


class PackagingRegressionTests(unittest.TestCase):
    def test_pyproject_registers_console_script(self) -> None:
        with PYPROJECT.open("rb") as handle:
            data = tomllib.load(handle)

        self.assertEqual(data["tool"]["uv"]["package"], True)
        self.assertEqual(
            data["project"]["scripts"]["vllm-hotload-hf-push-ipc"],
            "vllm_hotload.hf_push_ipc:main",
        )
        self.assertIn("src", data["tool"]["setuptools"]["packages"]["find"]["where"])
        self.assertIn(
            "vllm_patch",
            data["tool"]["setuptools"]["packages"]["find"]["where"],
        )

    def test_build_script_no_longer_writes_pth_overlay(self) -> None:
        content = BUILD_SCRIPT.read_text()

        self.assertIn("uv sync --locked --active", content)
        self.assertNotIn(".pth", content)
        self.assertNotIn("sys.path.insert", content)
        self.assertNotIn("vllm_hotpatch", content)

    def test_e2e_script_uses_uv_run_and_no_pythonpath(self) -> None:
        content = E2E_SCRIPT.read_text()

        self.assertIn('uv run --directory "$BASE_DIR"', content)
        self.assertIn("exec uv_run vllm serve", content)
        self.assertIn("uv_run vllm-hotload-hf-push-ipc", content)
        self.assertNotIn("PYTHONPATH=", content)
        self.assertNotIn("VLLM_PATCH_DIR=", content)

    def test_example_script_remains_thin_import_shim(self) -> None:
        content = SHIM_SCRIPT.read_text()

        self.assertIn("from vllm_hotload.hf_push_ipc import main", content)
        self.assertNotIn("import argparse", content)
        self.assertNotIn("import requests", content)


if __name__ == "__main__":
    unittest.main()