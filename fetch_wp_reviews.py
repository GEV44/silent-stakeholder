"""Fetch the app-review corpus that this repository deliberately does not ship.

`data/raw/` is gitignored because the dataset's redistribution licence is
unresolved, so a fresh clone has no corpus and must re-acquire it from source.
This downloads the published Hugging Face parquet snapshot to the path the
documented ingest recipe expects (README.md), then reports how many rows belong
to the configured app.

    python fetch_wp_reviews.py                 # WordPress (default)
    python fetch_wp_reviews.py --app ppsspp
    python fetch_wp_reviews.py --verify-only   # count rows in an existing file

The download is stdlib-only. Verification needs pandas + pyarrow and is skipped
with a warning if they are missing — the parquet is still usable by src.ingest.

Never commit what this writes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PARQUET_INDEX = "https://huggingface.co/api/datasets/sealuzh/app_reviews/parquet/default/train"
DEFAULT_TARGET = Path("data/raw/app_reviews.parquet")
APPS_CONFIG = Path("config/apps.json")
EXPECTED_TOTAL_ROWS = 288_065  # per README.md; a mismatch means the snapshot moved
USER_AGENT = "silent-stakeholder-ingest/1.0"


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def resolve_package(app: str) -> tuple[str, str]:
    """Read the package name from config/apps.json so this cannot drift from it."""

    try:
        config = json.loads(APPS_CONFIG.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {APPS_CONFIG}: {exc}") from exc
    if app not in config:
        raise SystemExit(f"unknown app {app!r}; choices: {', '.join(sorted(config))}")
    entry = config[app]
    return entry["package_name"], entry.get("display_name", app)


def download(target: Path, *, force: bool) -> Path:
    if target.exists() and not force:
        size = target.stat().st_size
        print(f"already present: {target} ({size / 1_048_576:.1f} MiB) — use --force to refetch")
        return target

    print(f"resolving parquet URL from {PARQUET_INDEX}")
    try:
        urls = json.loads(_get(PARQUET_INDEX))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise SystemExit(f"could not resolve the parquet URL: {exc}") from exc
    if not urls:
        raise SystemExit("Hugging Face returned no parquet shards")
    if len(urls) > 1:
        print(f"note: {len(urls)} shards published; this script expects 1 and will use the first")

    url = urls[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")

    # Write to a temp file in the same directory and move into place, so an
    # interrupted download can never leave a half-written parquet that later
    # looks like a valid corpus.
    tmp = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with (
            urllib.request.urlopen(request, timeout=300) as response,
            tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".partial", delete=False
            ) as handle,
        ):
            tmp = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise SystemExit(f"download failed: {exc}") from exc

    tmp.replace(target)
    print(f"wrote {target} ({target.stat().st_size / 1_048_576:.1f} MiB)")
    return target


def verify(target: Path, package_name: str, display_name: str) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("\npandas/pyarrow not available — skipping verification.")
        print("The parquet is still usable by `python -m src.ingest`.")
        return -1

    print(f"\nreading {target}")
    frame = pd.read_parquet(target)
    total = len(frame)
    print(f"  total rows      : {total:,}")
    if total != EXPECTED_TOTAL_ROWS:
        print(f"  WARNING: expected {EXPECTED_TOTAL_ROWS:,} per README — snapshot may have changed")

    if "package_name" not in frame.columns:
        print(f"  WARNING: no 'package_name' column; found {list(frame.columns)}")
        return -1

    matching = int((frame["package_name"] == package_name).sum())
    print(f"  {display_name} rows : {matching:,}  ({package_name})")
    print(f"  distinct apps   : {frame['package_name'].nunique():,}")
    if matching == 0:
        print("  WARNING: zero rows for this package — check config/apps.json")
    return matching


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app", default="wordpress", help="key in config/apps.json")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--force", action="store_true", help="refetch even if present")
    parser.add_argument("--verify-only", action="store_true", help="do not download")
    args = parser.parse_args(argv)

    package_name, display_name = resolve_package(args.app)

    if args.verify_only:
        if not args.target.exists():
            raise SystemExit(f"{args.target} does not exist; run without --verify-only")
        target = args.target
    else:
        target = download(args.target, force=args.force)

    verify(target, package_name, display_name)

    print("\nNext:")
    print(f"  python -m src.ingest --app {args.app} \\")
    print(f"    --reviews-csv {target} \\")
    print("    --github-state open \\")
    print(f"    --out-dir data/processed/{args.app}-open")
    print("\nDo not commit anything under data/ — the corpus licence is unresolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
