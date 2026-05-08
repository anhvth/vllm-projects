from __future__ import annotations

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINT_PATH = ROOT / "tools" / "lint.py"
SPEC = importlib.util.spec_from_file_location("workspace_lint", LINT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
workspace_lint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workspace_lint)


class LintCacheTests(unittest.TestCase):
    def test_cached_report_is_reused_when_inputs_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vllm_patch").mkdir()
            (root / "pyproject.toml").write_text("[project]\nname='tmp'\n")
            (root / "pyrightconfig.json").write_text('{"include": ["src"]}\n')
            (root / "src").mkdir()
            target = root / "src" / "demo.py"
            target.write_text("x = 1\n")

            targets = [str(root / "src")]
            report = {"summary": {"filesAnalyzed": 1, "timeInSec": 0.01}}

            workspace_lint._store_cached_report(str(root), targets, report)

            self.assertEqual(workspace_lint._load_cached_report(str(root), targets), report)

    def test_cached_report_is_invalidated_when_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vllm_patch").mkdir()
            (root / "pyproject.toml").write_text("[project]\nname='tmp'\n")
            (root / "pyrightconfig.json").write_text('{"include": ["src"]}\n')
            (root / "src").mkdir()
            target = root / "src" / "demo.py"
            target.write_text("x = 1\n")

            targets = [str(root / "src")]
            report = {"summary": {"filesAnalyzed": 1, "timeInSec": 0.01}}

            workspace_lint._store_cached_report(str(root), targets, report)
            time.sleep(0.01)
            target.write_text("x = 2\n")

            self.assertIsNone(workspace_lint._load_cached_report(str(root), targets))


if __name__ == "__main__":
    unittest.main()