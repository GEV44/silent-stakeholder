"""Fail-closed, offline production gate for Firecode.

The gate intentionally captures subprocess output instead of echoing it.  This
keeps test failures from accidentally copying credentials or protected evidence
into CI logs.  A failed check is reported by name and exit code only; reproduce
that command directly in a trusted development shell for detailed diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

Status = Literal["PASS", "BLOCKED", "SKIP"]

PROJECT_CREDENTIAL_ENV_VARS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "GEMINI_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_ACCESS_TOKEN",
        "OPENAI_API_KEY",
        "PGPASSWORD",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_URL",
    }
)


@dataclass(frozen=True)
class CheckResult:
    """One deterministic gate result."""

    name: str
    status: Status
    detail: str


def _safe_environment() -> dict[str, str]:
    """Return a subprocess environment with all supported offline switches set."""

    environment = os.environ.copy()
    credential_names = {name.casefold() for name in PROJECT_CREDENTIAL_ENV_VARS}
    for name in list(environment):
        if name.casefold() in credential_names:
            del environment[name]
    environment.update(
        {
            "ALL_PROXY": "http://127.0.0.1:9",
            "DATASETS_OFFLINE": "1",
            "FIRECODE_OFFLINE": "true",
            "GIT_TERMINAL_PROMPT": "0",
            "HF_HUB_OFFLINE": "1",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "NO_COLOR": "1",
            "NO_PROXY": "",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONUTF8": "1",
            "TERM": "dumb",
            "TRANSFORMERS_OFFLINE": "1",
            "all_proxy": "http://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "no_proxy": "",
        }
    )
    return environment


def _credential_environment_check(environment: dict[str, str]) -> CheckResult:
    credential_names = {name.casefold() for name in PROJECT_CREDENTIAL_ENV_VARS}
    residual = [name for name in environment if name.casefold() in credential_names]
    if residual:
        return CheckResult(
            "credential environment", "BLOCKED", f"credential variables remain={len(residual)}"
        )
    return CheckResult("credential environment", "PASS", "known credentials removed")


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    root: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> CheckResult:
    """Run a check without exposing its stdout or stderr."""

    try:
        completed = subprocess.run(  # noqa: S603 - argv is an internal, shell-free sequence.
            list(command),
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return CheckResult(name, "BLOCKED", "command unavailable")
    except subprocess.TimeoutExpired:
        return CheckResult(name, "BLOCKED", f"timeout after {timeout_seconds}s")
    except OSError:
        return CheckResult(name, "BLOCKED", "command could not start")

    if completed.returncode == 0:
        return CheckResult(name, "PASS", "exit 0")
    return CheckResult(name, "BLOCKED", f"exit {completed.returncode}")


def _git_bytes(root: Path, *arguments: str) -> tuple[int, bytes]:
    """Run a read-only Git query and suppress its diagnostic output."""

    git_executable = shutil.which("git")
    if git_executable is None:
        return 1, b""
    try:
        completed = subprocess.run(  # noqa: S603 - absolute Git path; shell is disabled.
            [git_executable, "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return 1, b""
    return completed.returncode, completed.stdout


def _tracked_files(root: Path) -> tuple[CheckResult, list[str]]:
    return_code, output = _git_bytes(root, "ls-files", "-z")
    if return_code != 0:
        return CheckResult("tracked-file inventory", "BLOCKED", "git query failed"), []

    try:
        paths = sorted(
            item.decode("utf-8", errors="strict").replace("\\", "/")
            for item in output.split(b"\0")
            if item
        )
    except UnicodeDecodeError:
        return CheckResult("tracked-file inventory", "BLOCKED", "non-UTF-8 tracked filename"), []
    return CheckResult("tracked-file inventory", "PASS", f"{len(paths)} files"), paths


def _clean_tree_check(root: Path) -> CheckResult:
    return_code, output = _git_bytes(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    if return_code != 0:
        return CheckResult("candidate working tree", "BLOCKED", "git query failed")
    records = sorted(record for record in output.split(b"\0") if record)
    if not records:
        return CheckResult("candidate working tree", "PASS", "clean")

    fingerprints = sorted(hashlib.sha256(record).hexdigest()[:12] for record in records)
    return CheckResult(
        "candidate working tree",
        "BLOCKED",
        f"dirty records={len(records)}[{','.join(fingerprints)}]",
    )


def _protected_category(path_text: str) -> str | None:
    """Classify a protected tracked filename without opening the file."""

    normalized = path_text.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    lowered = normalized.casefold()
    basename = path.name.casefold()

    permitted_placeholders = {
        "data/llm_cache/.gitkeep",
        "data/processed/.gitkeep",
        "data/raw/.gitkeep",
        "out/.gitkeep",
        "research/quarantine/readme.md",
    }
    if lowered in permitted_placeholders:
        return None

    protected_roots = {
        "data/llm_cache/": "model cache",
        "data/processed/": "processed evidence",
        "data/raw/": "raw evidence",
        "out/": "real-run output",
        "research/quarantine/": "quarantined evidence",
    }
    for prefix, category in protected_roots.items():
        if lowered.startswith(prefix):
            return category

    allowed_environment_templates = {".env.example", ".env.sample", ".env.template"}
    if basename.startswith(".env") and basename not in allowed_environment_templates:
        return "environment secret"

    exact_secret_names = {
        ".envrc",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.toml",
        "service-account.json",
        "service_account.json",
    }
    if basename in exact_secret_names:
        return "credential file"

    if path.suffix.casefold() in {".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"}:
        return "private key or certificate"

    protected_data_areas = {"data", "eval", "out", "review"}
    first_part = path.parts[0].casefold() if path.parts else ""
    answer_key_markers = ("answer-key", "answer_key", "answerkey")
    if first_part in protected_data_areas and any(
        marker in basename for marker in answer_key_markers
    ):
        return "answer key"

    return None


def _protected_file_check(paths: Sequence[str]) -> CheckResult:
    categories: dict[str, list[str]] = {}
    for path in paths:
        category = _protected_category(path)
        if category is not None:
            categories.setdefault(category, []).append(path)

    if not categories:
        return CheckResult("protected tracked files", "PASS", "none")

    # Only category counts and irreversible path fingerprints are printed.  This
    # identifies repeat findings without exposing sensitive filenames or values.
    details: list[str] = []
    for category in sorted(categories):
        fingerprints = sorted(
            hashlib.sha256(path.encode("utf-8")).hexdigest()[:12] for path in categories[category]
        )
        details.append(f"{category}={len(fingerprints)}[{','.join(fingerprints)}]")
    return CheckResult("protected tracked files", "BLOCKED", "; ".join(details))


def _report_export_privacy_check(root: Path, paths: Sequence[str]) -> CheckResult:
    """Reject tracked judge reports that embed redistribution-restricted text.

    The path-only protected-file inventory cannot distinguish a safe public
    report from an internal export.  Inspect only the structured payload and
    report aggregate counts; never print a span or other evidence value.
    """

    export_paths = [path for path in ("report.html", "docs/index.html") if path in paths]
    if not export_paths:
        return CheckResult("tracked report exports", "PASS", "none")

    marker = '<script id="payload" type="application/json">'
    invalid = internal = nonempty_spans = 0
    for relative_path in export_paths:
        candidate = root / relative_path
        try:
            if candidate.is_symlink() or not candidate.is_file():
                invalid += 1
                continue
            if candidate.stat().st_size > 10_000_000:
                invalid += 1
                continue
            document = candidate.read_text(encoding="utf-8", errors="strict")
            start = document.find(marker)
            end = document.find("</script>", start + len(marker))
            if start < 0 or end < 0:
                invalid += 1
                continue
            payload = json.loads(document[start + len(marker) : end])
            if not isinstance(payload, dict):
                invalid += 1
                continue
            findings = payload.get("findings")
            if not isinstance(findings, list):
                invalid += 1
                continue

            proof_count = 0
            export_nonempty = 0
            malformed = False
            for finding in findings:
                if not isinstance(finding, dict) or not isinstance(finding.get("proof"), list):
                    malformed = True
                    break
                for proof in finding["proof"]:
                    if not isinstance(proof, dict):
                        malformed = True
                        break
                    proof_count += 1
                    span = proof.get("span")
                    # Public exports must carry the field as the exact empty
                    # string emitted by report.py.  Missing, null, numeric, or
                    # structured values are schema drift and must fail closed;
                    # accepting them could hide restricted text from this
                    # string-only counter while leaving it in tracked JSON.
                    if not isinstance(span, str):
                        malformed = True
                        break
                    export_nonempty += bool(span)
                if malformed:
                    break
            if malformed:
                invalid += 1
                continue

            profile = payload.get("profile")
            withheld = payload.get("withheldQuotes")
            synthetic_demo = (
                payload.get("demo") is True and payload.get("mode") == "demo_fixture"
            )
            if profile != "public":
                internal += 1
            if export_nonempty and not synthetic_demo:
                nonempty_spans += export_nonempty
            if (
                profile != "public"
                or not isinstance(withheld, int)
                or isinstance(withheld, bool)
                or (
                    synthetic_demo
                    and withheld != 0
                )
                or (
                    not synthetic_demo
                    and (export_nonempty or withheld != proof_count)
                )
            ):
                invalid += 1
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            invalid += 1

    detail = (
        f"exports={len(export_paths)} invalid={invalid} "
        f"internal={internal} nonempty_spans={nonempty_spans}"
    )
    status: Status = "BLOCKED" if invalid else "PASS"
    return CheckResult("tracked report exports", status, detail)


def _pre_commit_hook_check(root: Path) -> CheckResult:
    return_code, output = _git_bytes(root, "ls-files", "--stage", "--", ".githooks/pre-commit")
    if return_code != 0 or not output.strip():
        return CheckResult("pre-commit hook executable", "BLOCKED", "hook is not tracked")

    try:
        mode = output.split(maxsplit=1)[0].decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return CheckResult("pre-commit hook executable", "BLOCKED", "invalid index mode")
    if mode != "100755":
        return CheckResult("pre-commit hook executable", "BLOCKED", f"tracked mode is {mode}")

    hook = root / ".githooks" / "pre-commit"
    if not hook.is_file():
        return CheckResult("pre-commit hook executable", "BLOCKED", "hook is missing")
    if os.name != "nt" and not os.access(hook, os.X_OK):
        return CheckResult(
            "pre-commit hook executable", "BLOCKED", "filesystem mode is not executable"
        )
    return CheckResult("pre-commit hook executable", "PASS", "tracked mode 100755")


def _hooks_path_check(root: Path) -> CheckResult:
    return_code, output = _git_bytes(root, "config", "--get", "core.hooksPath")
    if return_code != 0 or not output.strip():
        return CheckResult("core.hooksPath", "BLOCKED", "not configured")

    try:
        configured = output.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return CheckResult("core.hooksPath", "BLOCKED", "invalid configuration")

    configured_path = Path(configured).expanduser()
    if not configured_path.is_absolute():
        configured_path = root / configured_path
    try:
        matches = configured_path.resolve() == (root / ".githooks").resolve()
    except OSError:
        matches = False
    if not matches:
        return CheckResult("core.hooksPath", "BLOCKED", "does not resolve to .githooks")
    return CheckResult("core.hooksPath", "PASS", "configured")


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _optional_check(
    name: str,
    module_name: str,
    command: Sequence[str],
    *,
    strict: bool,
    root: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> CheckResult:
    if not _module_available(module_name):
        status: Status = "BLOCKED" if strict else "SKIP"
        detail = "required tool missing" if strict else "tool not installed"
        return CheckResult(name, status, detail)
    return _run_command(
        name,
        command,
        root=root,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )


def _dependency_structure_check(
    *,
    strict: bool,
    root: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> CheckResult:
    """Run an installed dependency-structure checker without calling a registry."""

    if _module_available("deptry"):
        command = [sys.executable, "-m", "deptry", "."]
        scanner = "deptry"
    elif _module_available("pipdeptree"):
        command = [sys.executable, "-m", "pipdeptree", "--warn", "fail"]
        scanner = "pipdeptree"
    else:
        status: Status = "BLOCKED" if strict else "SKIP"
        detail = (
            "required structure checker missing" if strict else "structure checker not installed"
        )
        return CheckResult("dependency structure", status, detail)

    result = _run_command(
        "dependency structure",
        command,
        root=root,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    return CheckResult(result.name, result.status, f"{scanner}: {result.detail}")


def _directory_has_files(path: Path) -> bool:
    try:
        return path.is_dir() and any(item.is_file() for item in path.rglob("*"))
    except OSError:
        return False


def _offline_cve_check(
    *,
    strict: bool,
    cache_dir: Path | None,
    root: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> CheckResult:
    """Run pip-audit only with a caller-supplied, populated local HTTP cache.

    Network proxies already point at a closed loopback port.  Therefore an
    incomplete or expired cache makes pip-audit fail closed instead of fetching.
    """

    requested = cache_dir is not None
    if not _module_available("pip_audit"):
        status: Status = "BLOCKED" if strict or requested else "SKIP"
        detail = "pip-audit missing" if status == "BLOCKED" else "pip-audit not installed"
        return CheckResult("offline CVE audit", status, detail)

    if cache_dir is None:
        status = "BLOCKED" if strict else "SKIP"
        detail = "local CVE cache required" if strict else "local CVE cache not supplied"
        return CheckResult("offline CVE audit", status, detail)

    resolved_cache = cache_dir.expanduser()
    if not resolved_cache.is_absolute():
        resolved_cache = root / resolved_cache
    if not _directory_has_files(resolved_cache):
        return CheckResult("offline CVE audit", "BLOCKED", "local CVE cache is missing or empty")

    return _run_command(
        "offline CVE audit",
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--local",
            "--disable-pip",
            "--cache-dir",
            str(resolved_cache),
            "--progress-spinner",
            "off",
        ],
        root=root,
        environment=environment,
        timeout_seconds=timeout_seconds,
    )


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Firecode's fail-closed production checks without network access or "
            "printing subprocess output."
        )
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "block when optional type/lint/dependency tools or a populated offline "
            "CVE cache are unavailable"
        ),
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="run only tracked-file, export-privacy, and hook-mode policy checks",
    )
    parser.add_argument(
        "--cve-cache-dir",
        type=Path,
        metavar="PATH",
        help=(
            "populated local pip-audit HTTP cache; external network remains blocked "
            "and cache misses fail the audit"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        metavar="N",
        help="maximum time for each subprocess check (default: 900)",
    )
    parsed = parser.parse_args(arguments)
    if parsed.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    root = Path(__file__).resolve().parents[1]
    environment = _safe_environment()

    print("Firecode production gate")
    print(f"mode={'strict' if args.strict else 'standard'} offline=true")

    results: list[CheckResult] = [
        CheckResult("offline environment", "PASS", "FIRECODE_OFFLINE=true"),
        _credential_environment_check(environment),
    ]
    if not args.policy_only:
        results.append(_clean_tree_check(root))

    inventory_result, tracked_files = _tracked_files(root)
    results.append(inventory_result)
    if inventory_result.status == "PASS":
        results.append(_protected_file_check(tracked_files))
        results.append(_report_export_privacy_check(root, tracked_files))
    else:
        results.append(CheckResult("protected tracked files", "BLOCKED", "inventory unavailable"))
        results.append(CheckResult("tracked report exports", "BLOCKED", "inventory unavailable"))

    results.append(_pre_commit_hook_check(root))
    if not args.policy_only:
        results.append(_hooks_path_check(root))

    if args.policy_only:
        for result in results:
            print(f"[{result.status}] {result.name}: {result.detail}")
        blocked = sum(result.status == "BLOCKED" for result in results)
        passed = sum(result.status == "PASS" for result in results)
        final_status = "BLOCKED" if blocked else "PASS"
        print(f"SUMMARY {final_status}: pass={passed} blocked={blocked} skip=0")
        return 1 if blocked else 0

    tracked_python = sorted(path for path in tracked_files if path.casefold().endswith(".py"))
    if tracked_python:
        compile_command = [sys.executable, "-m", "compileall", "-q", *tracked_python]
        pylint_command = [sys.executable, "-m", "pylint", *tracked_python]
    else:
        compile_command = [sys.executable, "-m", "compileall", "-q", "src"]
        pylint_command = [sys.executable, "-m", "pylint", "src"]

    mandatory_commands = [
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "--strict-config",
                "--strict-markers",
                "-o",
                "xfail_strict=true",
            ],
        ),
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        (
            "ruff security",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "S",
                "--ignore",
                "S311",
                "src",
                "scripts",
                "report.py",
            ],
        ),
        ("compileall", compile_command),
        ("pip check", [sys.executable, "-m", "pip", "check"]),
    ]
    for name, command in mandatory_commands:
        results.append(
            _run_command(
                name,
                command,
                root=root,
                environment=environment,
                timeout_seconds=args.timeout_seconds,
            )
        )

    results.append(
        _optional_check(
            "mypy",
            "mypy",
            [
                sys.executable,
                "-m",
                "mypy",
                "--no-incremental",
                "src",
                "scripts",
                "report.py",
            ],
            strict=args.strict,
            root=root,
            environment=environment,
            timeout_seconds=args.timeout_seconds,
        )
    )
    results.append(
        _optional_check(
            "pylint",
            "pylint",
            pylint_command,
            strict=args.strict,
            root=root,
            environment=environment,
            timeout_seconds=args.timeout_seconds,
        )
    )
    results.append(
        _dependency_structure_check(
            strict=args.strict,
            root=root,
            environment=environment,
            timeout_seconds=args.timeout_seconds,
        )
    )
    results.append(
        _offline_cve_check(
            strict=args.strict,
            cache_dir=args.cve_cache_dir,
            root=root,
            environment=environment,
            timeout_seconds=args.timeout_seconds,
        )
    )

    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    blocked = sum(result.status == "BLOCKED" for result in results)
    passed = sum(result.status == "PASS" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    final_status = "BLOCKED" if blocked else "PASS"
    print(f"SUMMARY {final_status}: pass={passed} blocked={blocked} skip={skipped}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
