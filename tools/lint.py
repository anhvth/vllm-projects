#!/usr/bin/env python3
"""Run pyright on the same analysis targets VS Code/Pylance uses.

Output format matches VS Code's Problems view:
  {file}:{line}:{col} - {severity}: {message} [{rule}]

Also validates config files for the same issues Pylance reports.

Notebook diagnostics are still editor-driven. Use VS Code/Pylance diagnostics as
the source of truth for `.ipynb` warnings because CLI pyright does not mirror
cell-scoped notebook analysis.

Supports:
  --file path   : lint a single file
  --json        : machine-readable JSON output (filtered)
    --verbose     : include the resolved analysis targets / config diagnostics in output
  --help        : this message

Exit code:
    0  all good
  1  pyright or invocation error
    2  diagnostics found
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


_SEVERITY_MAP: dict[str, str] = {
    "error": "error",
    "warning": "warning",
    "information": "info",
}

_KNOWN_NOISE: list[tuple[str, str, str]] = [
    (
        "vllm/__init__.py",
        "reportMissingImports",
        '.version" could not be resolved',
    ),
    (
        "entrypoints/cli/main.py",
        "reportMissingImports",
        'vllm_omni.entrypoints.cli.main" could not be resolved',
    ),
    (
        "entrypoints/cli/main.py",
        "reportOptionalMemberAccess",
        '"subparser" is not a known attribute of "None"',
    ),
    (
        "entrypoints/cli/main.py",
        "reportOptionalMemberAccess",
        '"dispatch_function" is not a known attribute of "None"',
    ),
    (
        "entrypoints/serve/__init__.py",
        "reportMissingImports",
        '.instrumentator" could not be resolved',
    ),
]

# Keys that Pylance flags as settingsNotOverridable when pyrightconfig.json exists
_SETTINGS_NOT_OVERRIDABLE = frozenset({
    "python.analysis.extraPaths",
    "python.analysis.exclude",
    "python.analysis.typeCheckingMode",
    "python.analysis.diagnosticSeverityOverrides",
    "python.analysis.include",
})

# Default excludes that Pylance expects in exclude lists
_VSCODE_DEFAULT_EXCLUDES = frozenset({"**/.*"})
_PYRIGHT_DEFAULT_EXCLUDES = frozenset({"**/node_modules", "**/__pycache__", "**/.*"})


def _is_known_noise(diag: dict) -> bool:
    file_path = diag.get("file", "")
    rule = diag.get("rule", "")
    message = diag.get("message", "")

    for suffix, expected_rule, expected_message in _KNOWN_NOISE:
        if (
            file_path.endswith(suffix)
            and rule == expected_rule
            and expected_message in message
        ):
            return True
    return False


def _resolve_workspace_root() -> str:
    """Walk up from CWD or script dir to find repo root."""
    candidates = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for start in candidates:
        d = start
        while True:
            if os.path.isdir(os.path.join(d, "vllm_patch")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return os.getcwd()


def _resolve_analysis_targets(root: str) -> list[str]:
    """Mirror the workspace analysis scope from VS Code or pyrightconfig."""
    config_candidates = [
        (os.path.join(root, ".vscode", "settings.json"), "python.analysis.include"),
        (os.path.join(root, "pyrightconfig.json"), "include"),
    ]

    targets: list[str] = []

    for config_path, include_key in config_candidates:
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        includes = config.get(include_key, [])
        if not isinstance(includes, list):
            continue

        for include in includes:
            if not isinstance(include, str):
                continue
            target = include.replace("${workspaceFolder}", root)
            target = os.path.normpath(target)
            if os.path.exists(target):
                targets.append(target)

        if targets:
            break

    if not targets:
        return [root]

    return list(dict.fromkeys(targets))


def _check_vscode_settings(root: str) -> list[dict]:
    """Check .vscode/settings.json for the same issues Pylance reports.

    Returns diagnostics in the same dict format as pyright's generalDiagnostics
    (file, range, severity, rule, message).
    """
    settings_path = os.path.join(root, ".vscode", "settings.json")
    diags: list[dict] = []
    if not os.path.isfile(settings_path):
        return diags
    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
    except (OSError, json.JSONDecodeError):
        return diags

    rel_path = os.path.relpath(settings_path, root)
    has_pyrightconfig = os.path.isfile(os.path.join(root, "pyrightconfig.json"))

    # Check for settingsNotOverridable
    if has_pyrightconfig:
        for key in _SETTINGS_NOT_OVERRIDABLE:
            if key in settings:
                short_key = key.split(".", 1)[1]  # "analysis.extraPaths" etc.
                diags.append({
                    "file": rel_path,
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "severity": "warning",
                    "rule": "settingsNotOverridable",
                    "message": f"'{key}' cannot be set when a pyrightconfig.json or pyproject.toml is being used.",
                })

    # Check for missingDefaultExcludes in the "exclude" key
    # NOTE: VS Code reports at the exclude list's position; we approximate at line 0
    exclude_list = settings.get("exclude")
    if isinstance(exclude_list, list):
        existing = frozenset(exclude_list)
        missing = _VSCODE_DEFAULT_EXCLUDES - existing
        if missing:
            missing_str = ", ".join(sorted(missing))
            diags.append({
                "file": rel_path,
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 0},
                },
                "severity": "information",
                "rule": "missingDefaultExcludes",
                "message": f"The exclude list is missing default excludes such as '{missing_str}'. Consider adding them to avoid analyzing unnecessary files.",
            })

    return diags


def _check_pyrightconfig(root: str) -> list[dict]:
    """Check pyrightconfig.json for missingDefaultExcludes."""
    config_path = os.path.join(root, "pyrightconfig.json")
    diags: list[dict] = []
    if not os.path.isfile(config_path):
        return diags
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return diags

    rel_path = os.path.relpath(config_path, root)

    exclude_list = config.get("exclude")
    if isinstance(exclude_list, list):
        existing = frozenset(exclude_list)
        missing = _PYRIGHT_DEFAULT_EXCLUDES - existing
        if missing:
            missing_str = ", ".join(sorted(missing))
            diags.append({
                "file": rel_path,
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 0},
                },
                "severity": "information",
                "rule": "missingDefaultExcludes",
                "message": f"The exclude list is missing default excludes such as '{missing_str}'. Consider adding them to avoid analyzing unnecessary files.",
            })

    return diags


def _print_diag(diag: dict, root: str) -> None:
    """Print a single diagnostic in VS Code Problems format."""
    file_path = diag.get("file", "")
    line = diag.get("range", {}).get("start", {}).get("line", 0) + 1
    col = diag.get("range", {}).get("start", {}).get("character", 0) + 1
    sev = _SEVERITY_MAP.get(diag.get("severity", "information"), "info")
    rule = diag.get("rule", "?")
    message = diag.get("message", "")
    print(f"{file_path}:{line}:{col} - {sev}: {message} [{rule}]")


def _config_diag_to_json(diag: dict) -> dict:
    """Convert a config diagnostic dict to the JSON output format."""
    return {
        "file": diag["file"],
        "severity": diag["severity"],
        "rule": diag["rule"],
        "message": diag["message"],
        "range": diag["range"],
    }


def main() -> int:
    root = _resolve_workspace_root()

    parser = argparse.ArgumentParser(
        description="Lint the workspace analysis targets used by VS Code/Pylance.",
        add_help=False,
    )
    parser.add_argument("--file", type=str, default=None, help="Single file to lint")
    parser.add_argument("--json", action="store_true", help="Emit filtered JSON output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include the resolved analysis targets in output",
    )
    parser.add_argument("--help", action="store_true", help="Show this help")
    args, _ = parser.parse_known_args()

    if args.help:
        parser.print_help()
        return 0

    # --- Config validation (same checks Pylance does in VS Code) ---
    config_diags: list[dict] = []
    config_diags.extend(_check_vscode_settings(root))
    config_diags.extend(_check_pyrightconfig(root))

    # --- Pyright analysis ---
    # Resolve target scope.
    if args.file:
        target = os.path.abspath(args.file)
        if not os.path.exists(target):
            print(f"error: file not found: {target}", file=sys.stderr)
            return 1
        targets = [target]
    else:
        targets = _resolve_analysis_targets(root)

    project = os.path.join(root, "pyrightconfig.json")

    # Run pyright.
    cmd = [
        sys.executable,
        "-m",
        "pyright",
        "--outputjson",
        "--project",
        project,
        *targets,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)

    # Parse output
    try:
        report = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        print(result.stderr or result.stdout, file=sys.stderr)
        return 1

    diagnostics: list[dict] = report.get("generalDiagnostics", [])
    real_diagnostics = [diag for diag in diagnostics if not _is_known_noise(diag)]
    filtered_count = len(diagnostics) - len(real_diagnostics)

    all_diagnostics = real_diagnostics + config_diags

    rc = 0
    if all_diagnostics:
        rc = 2

    if args.json:
        output = {
            "version": report.get("version"),
            "time": report.get("time"),
            "generalDiagnostics": [_config_diag_to_json(d) if "rule" in d and d.get("rule") in ("settingsNotOverridable", "missingDefaultExcludes") else d for d in real_diagnostics],
            "configDiagnostics": config_diags,
            "summary": {
                "filesAnalyzed": report.get("summary", {}).get("filesAnalyzed", 0),
                "diagnosticCount": len(real_diagnostics),
                "configDiagnosticCount": len(config_diags),
                "filteredCount": filtered_count,
                "analysisTargets": targets,
                "timeInSec": report.get("summary", {}).get("timeInSec", 0),
            },
        }
        # Merge config diags into generalDiagnostics for total picture
        output["generalDiagnostics"] = real_diagnostics + config_diags
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        if args.verbose:
            printable_targets = [
                os.path.relpath(path, root) if path.startswith(root) else path
                for path in targets
            ]
            print(f"analysis targets: {', '.join(printable_targets)}")

        if all_diagnostics:
            for diag in all_diagnostics:
                _print_diag(diag, root)
        else:
            print("\x1b[32mAll clear.\x1b[0m")

        total = len(all_diagnostics)
        cfgs = len(config_diags)
        print(
            f"\n[summary] targets={len(targets)}, files={report.get('summary', {}).get('filesAnalyzed', 0)}, "
            f"diagnostics={total} (pyright={len(real_diagnostics)}, config={cfgs}), filtered={filtered_count}, "
            f"time={report.get('summary', {}).get('timeInSec', 0)}s"
        )

    return rc


if __name__ == "__main__":
    sys.exit(main())
