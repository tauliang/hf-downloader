#!/usr/bin/env python3
"""
hf_download.py — Resilient HuggingFace model downloader
Supports: resume, infinite retries, all huggingface-cli download parameters
Usage: python hf_download.py --help
"""

import os
import sys
import time
import signal
import hashlib
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import HfApi, hf_hub_url, constants
from huggingface_hub.utils import build_hf_headers, EntryNotFoundError, RepositoryNotFoundError

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_download")

# ── Signal handling ───────────────────────────────────────────────────────────

_stop_event = threading.Event()

def _handle_signal(signum, frame):
    log.warning("Interrupted — you can resume this download by running the same command.")
    _stop_event.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Helpers ───────────────────────────────────────────────────────────────────

CHUNK = 8 * 1024 * 1024   # 8 MB read-buffer

_ANSI_UP   = "\x1b[1A"
_ANSI_ERASE = "\x1b[2K"

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def _fmt_speed(bps: float) -> str:
    return f"{_fmt_bytes(bps)}/s"

def _sha256_of_file(path: Path, start: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()

# ── Core downloader ───────────────────────────────────────────────────────────

def download_file(
    url: str,
    dest: Path,
    headers: dict,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    max_retries: int = -1,          # -1 = infinite
    retry_delay: float = 5.0,
    retry_backoff: float = 1.5,
    retry_max_delay: float = 300.0,
) -> bool:
    """
    Download *url* into *dest*, resuming where we left off.
    Returns True on success, False if _stop_event is set.
    Raises on unrecoverable errors.
    """
    attempt  = 0
    delay    = retry_delay

    while not _stop_event.is_set():
        attempt += 1
        resume_pos = dest.stat().st_size if dest.exists() else 0

        if expected_size and resume_pos >= expected_size:
            log.info("  ✓ already complete: %s", dest.name)
            return True

        req_headers = dict(headers)
        if resume_pos:
            req_headers["Range"] = f"bytes={resume_pos}-"
            log.info("  ↻ resuming %s at %s (attempt %d)",
                     dest.name, _fmt_bytes(resume_pos), attempt)
        else:
            log.info("  ↓ starting %s (attempt %d)", dest.name, attempt)

        try:
            with requests.get(url, headers=req_headers, stream=True,
                              timeout=(30, 60)) as resp:

                if resp.status_code == 416:          # Range Not Satisfiable
                    # Server says we already have everything
                    log.info("  ✓ server confirmed complete: %s", dest.name)
                    return True

                if resp.status_code not in (200, 206):
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code} for {url}", response=resp
                    )

                mode   = "ab" if resp.status_code == 206 else "wb"
                offset = resume_pos if resp.status_code == 206 else 0

                total_raw = resp.headers.get("Content-Range") or resp.headers.get("Content-Length")
                if resp.status_code == 206 and "Content-Range" in resp.headers:
                    # Content-Range: bytes 12345-/999999
                    total = int(resp.headers["Content-Range"].split("/")[-1])
                else:
                    total = int(resp.headers.get("Content-Length", 0)) + offset or expected_size or 0

                downloaded = offset
                t0 = time.monotonic()
                last_print = t0
                printed_line = False

                dest.parent.mkdir(parents=True, exist_ok=True)

                with open(dest, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if _stop_event.is_set():
                            log.warning("  ✗ download interrupted, progress saved.")
                            return False
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)

                        now = time.monotonic()
                        if now - last_print >= 1.0:
                            elapsed = now - t0
                            speed   = (downloaded - offset) / elapsed if elapsed else 0
                            pct     = f"{100*downloaded/total:.1f}%" if total else "?%"
                            eta_s   = int((total - downloaded) / speed) if (speed and total) else 0
                            eta     = f"{eta_s//3600:02d}:{(eta_s%3600)//60:02d}:{eta_s%60:02d}"
                            bar_w   = 30
                            done_w  = int(bar_w * downloaded / total) if total else 0
                            bar     = "█" * done_w + "░" * (bar_w - done_w)
                            line    = (
                                f"  [{bar}] {pct}  "
                                f"{_fmt_bytes(downloaded)}/{_fmt_bytes(total)}  "
                                f"{_fmt_speed(speed)}  ETA {eta}"
                            )
                            if printed_line:
                                sys.stdout.write(_ANSI_UP + _ANSI_ERASE)
                            sys.stdout.write(line + "\n")
                            sys.stdout.flush()
                            last_print   = now
                            printed_line = True

                # Final newline after the progress bar
                if printed_line:
                    print()

                log.info("  ✓ finished: %s (%s)", dest.name, _fmt_bytes(downloaded))

                # Optional SHA-256 verification
                if expected_sha256:
                    log.info("  … verifying sha256 …")
                    got = _sha256_of_file(dest)
                    if got != expected_sha256:
                        raise ValueError(
                            f"SHA-256 mismatch for {dest.name}:\n"
                            f"  expected {expected_sha256}\n  got      {got}"
                        )
                    log.info("  ✓ sha256 OK")

                delay = retry_delay  # reset back-off on success
                return True

        except (requests.RequestException, OSError, ValueError) as exc:
            log.warning("  ✗ error on attempt %d: %s", attempt, exc)

            if max_retries != -1 and attempt >= max_retries:
                raise RuntimeError(
                    f"Giving up after {attempt} attempts on {dest.name}"
                ) from exc

            if _stop_event.is_set():
                return False

            log.info("  … retrying in %.0f s …", delay)
            # Interruptible sleep
            for _ in range(int(delay * 10)):
                if _stop_event.is_set():
                    return False
                time.sleep(0.1)

            delay = min(delay * retry_backoff, retry_max_delay)

    return False

# ── Repo-level download ───────────────────────────────────────────────────────

def download_repo(args: argparse.Namespace) -> None:
    api = HfApi(
        endpoint=args.endpoint or constants.ENDPOINT,
        token=args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )

    repo_id   = args.repo_id
    repo_type = args.repo_type or "model"
    revision  = args.revision or "main"

    # Resolve local dir
    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser().resolve()
    else:
        cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir \
                    else Path(constants.HF_HUB_CACHE)
        local_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision

    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("Destination: %s", local_dir)

    # List files in repo
    try:
        all_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        )
    except RepositoryNotFoundError:
        log.error("Repository '%s' not found (check spelling / token permissions).", repo_id)
        sys.exit(1)

    # Apply include / exclude filters
    include_patterns = args.include or []
    exclude_patterns = args.exclude or []

    import fnmatch

    def _match(filename: str, patterns: list) -> bool:
        return any(fnmatch.fnmatch(filename, p) for p in patterns)

    files_to_download = []
    for fname in sorted(all_files):
        if include_patterns and not _match(fname, include_patterns):
            continue
        if exclude_patterns and _match(fname, exclude_patterns):
            log.debug("  skip (excluded): %s", fname)
            continue
        files_to_download.append(fname)

    if not files_to_download:
        log.warning("No files matched the given include/exclude filters.")
        return

    log.info("Files to download: %d", len(files_to_download))

    headers = build_hf_headers(
        token=args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        library_name="hf_download",
    )

    ok = 0
    for i, fname in enumerate(files_to_download, 1):
        if _stop_event.is_set():
            break

        dest = local_dir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = hf_hub_url(
            repo_id=repo_id,
            filename=fname,
            repo_type=repo_type,
            revision=revision,
            endpoint=args.endpoint or constants.ENDPOINT,
        )

        log.info("[%d/%d] %s", i, len(files_to_download), fname)

        success = download_file(
            url=url,
            dest=dest,
            headers=headers,
            max_retries=-1 if not args.max_retries else args.max_retries,
            retry_delay=args.retry_delay,
            retry_backoff=args.retry_backoff,
            retry_max_delay=args.retry_max_delay,
        )

        if success:
            ok += 1
        elif _stop_event.is_set():
            log.warning("Download paused. Re-run the same command to resume.")
            break
        else:
            log.error("Failed to download: %s", fname)

    log.info("Done. %d/%d files complete.", ok, len(files_to_download))

# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hf_download",
        description=(
            "Resilient HuggingFace downloader — resumes partial files, "
            "retries forever on flaky connections."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Download a full model (will auto-resume if interrupted):
  python hf_download.py meta-llama/Llama-3-70B-Instruct

  # Only the safetensors shards:
  python hf_download.py meta-llama/Llama-3-70B-Instruct --include "*.safetensors"

  # Dataset:
  python hf_download.py HuggingFaceFW/fineweb --repo-type dataset

  # Custom cache location:
  python hf_download.py bigscience/bloom --cache-dir /mnt/fast-disk/hf-cache

  # Use a token (or set HUGGING_FACE_HUB_TOKEN env var):
  python hf_download.py meta-llama/Meta-Llama-3-8B --token hf_XXXX
""",
    )

    # ── Positional ─────────────────────────────────────────────────────────
    p.add_argument("repo_id", help="Model/dataset repo id, e.g. meta-llama/Llama-3-70B-Instruct")

    # ── huggingface-cli parity ──────────────────────────────────────────────
    p.add_argument("--repo-type",   default="model",
                   choices=["model", "dataset", "space"],
                   help="Type of repo (default: model)")
    p.add_argument("--revision",    default="main",
                   help="Branch, tag, or commit hash (default: main)")
    p.add_argument("--include",     nargs="+", metavar="PATTERN",
                   help="Glob patterns of files to include, e.g. '*.safetensors'")
    p.add_argument("--exclude",     nargs="+", metavar="PATTERN",
                   help="Glob patterns of files to exclude")
    p.add_argument("--token",       default=None,
                   help="HuggingFace token (or set HUGGING_FACE_HUB_TOKEN)")
    p.add_argument("--cache-dir",   dest="cache_dir", default=None,
                   help="Override the default HF cache directory")
    p.add_argument("--local-dir",   dest="local_dir", default=None,
                   help="Download directly into this directory (flat layout)")
    p.add_argument("--endpoint",    default=None,
                   help="HuggingFace Hub endpoint (default: https://huggingface.co)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output")

    # ── Retry / resilience ─────────────────────────────────────────────────
    rg = p.add_argument_group("retry / resilience")
    rg.add_argument("--max-retries",   type=int, default=-1, dest="max_retries",
                    help="Max retry attempts per file (-1 = infinite, default)")
    rg.add_argument("--retry-delay",   type=float, default=5.0, dest="retry_delay",
                    help="Initial delay between retries in seconds (default: 5)")
    rg.add_argument("--retry-backoff", type=float, default=1.5, dest="retry_backoff",
                    help="Exponential back-off multiplier (default: 1.5)")
    rg.add_argument("--retry-max-delay", type=float, default=300.0, dest="retry_max_delay",
                    help="Cap on retry delay in seconds (default: 300)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.quiet:
        log.setLevel(logging.WARNING)

    download_repo(args)


if __name__ == "__main__":
    main()
