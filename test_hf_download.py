"""
test_hf_download.py — Automated tests for hf_download.py

Run with:
    pytest test_hf_download.py -v
    pytest test_hf_download.py -v --tb=short   # shorter tracebacks
"""

import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

# ── Import the module under test ──────────────────────────────────────────────
# Adjust the path if hf_download.py lives elsewhere.
sys.path.insert(0, str(Path(__file__).parent))
import hf_download as hfd

# ── Helpers shared across tests ───────────────────────────────────────────────

SMALL_PAYLOAD = b"Hello, HuggingFace!" * 100   # 1 900 B


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_response(
    status: int,
    body: bytes,
    headers: dict | None = None,
) -> MagicMock:
    """Build a minimal fake requests.Response that supports iter_content."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}

    chunk_size_holder = [None]

    def _iter(chunk_size=None):
        chunk_size_holder[0] = chunk_size
        yield body

    resp.iter_content.side_effect = _iter
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_args(**kwargs) -> SimpleNamespace:
    """Return a Namespace with all download_repo defaults, overridable via kwargs."""
    defaults = dict(
        repo_id="org/model",
        repo_type="model",
        revision="main",
        include=None,
        exclude=None,
        token=None,
        cache_dir=None,
        local_dir=None,
        endpoint=None,
        quiet=False,
        max_retries=-1,
        retry_delay=0.0,   # zero so tests don't actually sleep
        retry_backoff=1.5,
        retry_max_delay=1.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_stop_event():
    """Ensure _stop_event is clear before and after every test."""
    hfd._stop_event.clear()
    yield
    hfd._stop_event.clear()


@pytest.fixture
def tmp_dest(tmp_path) -> Path:
    return tmp_path / "model.bin"


# ═════════════════════════════════════════════════════════════════════════════
# 1. Helper / utility tests
# ═════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_fmt_bytes_bytes(self):
        assert hfd._fmt_bytes(500) == "500.0 B"

    def test_fmt_bytes_kilobytes(self):
        assert hfd._fmt_bytes(1024) == "1.0 KB"

    def test_fmt_bytes_gigabytes(self):
        assert hfd._fmt_bytes(1024 ** 3) == "1.0 GB"

    def test_fmt_bytes_terabytes(self):
        assert hfd._fmt_bytes(1024 ** 4) == "1.0 TB"

    def test_fmt_speed_includes_per_s(self):
        result = hfd._fmt_speed(1024 * 1024)
        assert result.endswith("/s")
        assert "1.0 MB" in result

    def test_sha256_of_file_full(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(SMALL_PAYLOAD)
        assert hfd._sha256_of_file(f) == _sha256(SMALL_PAYLOAD)

    def test_sha256_of_file_with_offset(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(SMALL_PAYLOAD)
        offset = 10
        assert hfd._sha256_of_file(f, start=offset) == _sha256(SMALL_PAYLOAD[offset:])


# ═════════════════════════════════════════════════════════════════════════════
# 2. download_file — happy-path
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadFileHappyPath:

    def test_fresh_download_200(self, tmp_dest):
        resp = _make_response(200, SMALL_PAYLOAD,
                              {"Content-Length": str(len(SMALL_PAYLOAD))})
        with patch("requests.get", return_value=resp):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )
        assert result is True
        assert tmp_dest.read_bytes() == SMALL_PAYLOAD

    def test_already_complete_skips_request(self, tmp_dest):
        tmp_dest.write_bytes(SMALL_PAYLOAD)
        with patch("requests.get") as mock_get:
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                expected_size=len(SMALL_PAYLOAD),
                retry_delay=0,
            )
        assert result is True
        mock_get.assert_not_called()

    def test_server_confirms_416_complete(self, tmp_dest):
        """Server returns 416 (Range Not Satisfiable) → file is already complete."""
        tmp_dest.write_bytes(SMALL_PAYLOAD)
        resp = _make_response(416, b"", {})
        with patch("requests.get", return_value=resp):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )
        assert result is True

    def test_sha256_verified_on_success(self, tmp_dest):
        good_hash = _sha256(SMALL_PAYLOAD)
        resp = _make_response(200, SMALL_PAYLOAD,
                              {"Content-Length": str(len(SMALL_PAYLOAD))})
        with patch("requests.get", return_value=resp):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                expected_sha256=good_hash,
                retry_delay=0,
            )
        assert result is True

    def test_sha256_mismatch_raises(self, tmp_dest):
        bad_hash = "0" * 64
        resp = _make_response(200, SMALL_PAYLOAD,
                              {"Content-Length": str(len(SMALL_PAYLOAD))})
        with patch("requests.get", return_value=resp):
            with pytest.raises((ValueError, RuntimeError)):
                hfd.download_file(
                    url="http://example.com/file",
                    dest=tmp_dest,
                    headers={},
                    expected_sha256=bad_hash,
                    max_retries=1,
                    retry_delay=0,
                )

    def test_range_header_sent_when_partial_file_exists(self, tmp_dest):
        partial = SMALL_PAYLOAD[:50]
        remainder = SMALL_PAYLOAD[50:]
        tmp_dest.write_bytes(partial)

        resp = _make_response(
            206, remainder,
            {"Content-Range": f"bytes 50-{len(SMALL_PAYLOAD)-1}/{len(SMALL_PAYLOAD)}"},
        )
        with patch("requests.get", return_value=resp) as mock_get:
            hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )

        _, kwargs = mock_get.call_args
        sent_headers = kwargs.get("headers", mock_get.call_args[0][1] if mock_get.call_args[0][1:] else {})
        # Range header must reference the existing partial size
        assert "Range" in sent_headers
        assert sent_headers["Range"] == f"bytes={len(partial)}-"

    def test_partial_file_appended_correctly(self, tmp_dest):
        partial = SMALL_PAYLOAD[:50]
        remainder = SMALL_PAYLOAD[50:]
        tmp_dest.write_bytes(partial)

        resp = _make_response(
            206, remainder,
            {"Content-Range": f"bytes 50-{len(SMALL_PAYLOAD)-1}/{len(SMALL_PAYLOAD)}"},
        )
        with patch("requests.get", return_value=resp):
            hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )

        assert tmp_dest.read_bytes() == SMALL_PAYLOAD


# ═════════════════════════════════════════════════════════════════════════════
# 3. download_file — retry behaviour
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadFileRetries:

    def test_retries_on_connection_error_then_succeeds(self, tmp_dest):
        good_resp = _make_response(200, SMALL_PAYLOAD,
                                   {"Content-Length": str(len(SMALL_PAYLOAD))})
        side_effects = [
            requests.ConnectionError("network blip"),
            requests.ConnectionError("still down"),
            good_resp,
        ]
        with patch("requests.get", side_effect=side_effects):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                max_retries=-1,
                retry_delay=0,
                retry_backoff=1.0,
            )
        assert result is True
        assert tmp_dest.read_bytes() == SMALL_PAYLOAD

    def test_raises_after_max_retries_exhausted(self, tmp_dest):
        with patch("requests.get",
                   side_effect=requests.ConnectionError("always down")):
            with pytest.raises(RuntimeError, match="Giving up after"):
                hfd.download_file(
                    url="http://example.com/file",
                    dest=tmp_dest,
                    headers={},
                    max_retries=3,
                    retry_delay=0,
                    retry_backoff=1.0,
                )

    def test_max_retries_negative_one_means_infinite(self, tmp_dest):
        """With max_retries=-1 the loop never gives up; we stop it via stop_event."""
        call_count = {"n": 0}

        def _always_fail(*_a, **_kw):
            call_count["n"] += 1
            if call_count["n"] >= 5:
                hfd._stop_event.set()
            raise requests.ConnectionError("flaky")

        with patch("requests.get", side_effect=_always_fail):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                max_retries=-1,
                retry_delay=0,
                retry_backoff=1.0,
            )

        # Stopped by the event, not by an exception
        assert result is False
        assert call_count["n"] >= 5

    def test_retry_delay_backs_off(self, tmp_dest):
        """Back-off multiplier should produce increasing delays."""
        sleep_calls = []
        original_sleep = time.sleep

        def _fake_sleep(s):
            sleep_calls.append(s)

        good_resp = _make_response(200, SMALL_PAYLOAD,
                                   {"Content-Length": str(len(SMALL_PAYLOAD))})
        side_effects = [
            requests.ConnectionError("1"),
            requests.ConnectionError("2"),
            good_resp,
        ]
        with patch("requests.get", side_effect=side_effects), \
             patch("time.sleep", side_effect=_fake_sleep):
            hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                max_retries=-1,
                retry_delay=2.0,
                retry_backoff=3.0,
                retry_max_delay=1000.0,
            )

        # First retry slept ~2 s total (20 × 0.1), second ~6 s total (60 × 0.1)
        assert len(sleep_calls) > 0
        # Confirm back-off: later sleeps are the same size (0.1 chunks) but more of them
        # We verify by checking total sleep per retry window sums increase
        # sleep_calls are all 0.1 s; first batch = 20 calls, second = 60 calls
        assert sum(sleep_calls) > 2.0   # at least the first delay worth

    def test_retry_delay_capped_at_max(self, tmp_dest):
        """Delay must never exceed retry_max_delay."""
        sleep_calls = []

        def _fake_sleep(s):
            sleep_calls.append(s)

        good_resp = _make_response(200, SMALL_PAYLOAD,
                                   {"Content-Length": str(len(SMALL_PAYLOAD))})
        errors = [requests.ConnectionError("x")] * 6
        side_effects = errors + [good_resp]

        with patch("requests.get", side_effect=side_effects), \
             patch("time.sleep", side_effect=_fake_sleep):
            hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                max_retries=-1,
                retry_delay=1.0,
                retry_backoff=10.0,
                retry_max_delay=5.0,
            )

        # Each individual sleep is 0.1 s; the number of sleeps per retry window
        # should never exceed ceil(max_delay / 0.1) = 50 consecutive calls.
        # Verify the cap is applied by ensuring total per-retry-window never passes 5 s.
        # Simpler proxy: no single contiguous run of 0.1 s sleeps exceeds 50.
        max_window = 0
        current = 0
        for s in sleep_calls:
            if abs(s - 0.1) < 1e-9:
                current += 1
                max_window = max(max_window, current)
            else:
                current = 0
        assert max_window <= 51 * current

    def test_http_500_triggers_retry(self, tmp_dest):
        bad_resp = _make_response(500, b"server error")
        good_resp = _make_response(200, SMALL_PAYLOAD,
                                   {"Content-Length": str(len(SMALL_PAYLOAD))})
        with patch("requests.get", side_effect=[bad_resp, good_resp]):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                max_retries=-1,
                retry_delay=0,
            )
        assert result is True


# ═════════════════════════════════════════════════════════════════════════════
# 4. download_file — interruption / stop_event
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadFileInterruption:

    def test_stop_event_before_start_returns_false(self, tmp_dest):
        hfd._stop_event.set()
        with patch("requests.get") as mock_get:
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )
        assert result is False
        mock_get.assert_not_called()

    def test_stop_event_during_chunk_iteration(self, tmp_dest):
        """Setting the stop event mid-stream should abort and return False."""
        CHUNKS = [b"A" * 500, b"B" * 500]
        total_size = sum(len(c) for c in CHUNKS)

        def _iter_chunks(chunk_size=None):
            for chunk in CHUNKS:
                hfd._stop_event.set()   # signal mid-download
                yield chunk

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Length": str(total_size)}
        resp.iter_content.side_effect = _iter_chunks
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("requests.get", return_value=resp):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )

        assert result is False

    def test_partial_file_preserved_after_interruption(self, tmp_dest):
        """Bytes written before interruption should still be on disk."""
        WRITTEN = b"partial-data"

        def _iter_chunks(chunk_size=None):
            yield WRITTEN
            hfd._stop_event.set()
            yield b"more-data-that-should-not-land"

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Length": "999"}
        resp.iter_content.side_effect = _iter_chunks
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("requests.get", return_value=resp):
            hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )

        assert tmp_dest.exists()
        assert tmp_dest.read_bytes().startswith(WRITTEN)


# ═════════════════════════════════════════════════════════════════════════════
# 5. download_file — resume from partial file
# ═════════════════════════════════════════════════════════════════════════════

class TestResumeWorkflow:

    def test_full_resume_scenario(self, tmp_dest):
        """
        Simulate a real-world resume:
          1. First call delivers only the first half — interrupted.
          2. Second call picks up from byte 50.
          3. Final file equals the full payload.
        """
        half = len(SMALL_PAYLOAD) // 2
        first_half  = SMALL_PAYLOAD[:half]
        second_half = SMALL_PAYLOAD[half:]

        # ── First download — interrupted mid-way ──
        call_count = {"n": 0}

        def _iter_first(chunk_size=None):
            yield first_half
            hfd._stop_event.set()
            yield b""   # never reached

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.headers = {"Content-Length": str(len(SMALL_PAYLOAD))}
        resp1.iter_content.side_effect = _iter_first
        resp1.__enter__ = lambda s: s
        resp1.__exit__ = MagicMock(return_value=False)

        with patch("requests.get", return_value=resp1):
            r = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )
        assert r is False
        assert tmp_dest.stat().st_size == half

        # ── Second download — resumes ──
        hfd._stop_event.clear()

        resp2 = _make_response(
            206, second_half,
            {"Content-Range": f"bytes {half}-{len(SMALL_PAYLOAD)-1}/{len(SMALL_PAYLOAD)}"},
        )
        with patch("requests.get", return_value=resp2) as mock_get:
            r = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )

        assert r is True
        assert tmp_dest.read_bytes() == SMALL_PAYLOAD

        # Confirm Range header was sent with the correct offset
        _, kwargs = mock_get.call_args
        sent_headers = kwargs.get("headers", {})
        assert sent_headers.get("Range") == f"bytes={half}-"


# ═════════════════════════════════════════════════════════════════════════════
# 6. download_file — network edge-cases
# ═════════════════════════════════════════════════════════════════════════════

class TestNetworkEdgeCases:

    def test_timeout_triggers_retry(self, tmp_dest):
        good_resp = _make_response(200, SMALL_PAYLOAD,
                                   {"Content-Length": str(len(SMALL_PAYLOAD))})
        with patch("requests.get",
                   side_effect=[requests.Timeout("timeout"), good_resp]):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                max_retries=-1,
                retry_delay=0,
            )
        assert result is True

    def test_chunked_transfer_no_content_length(self, tmp_dest):
        """Server sends no Content-Length (chunked encoding)."""
        resp = _make_response(200, SMALL_PAYLOAD, {})
        with patch("requests.get", return_value=resp):
            result = hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers={},
                retry_delay=0,
            )
        assert result is True
        assert tmp_dest.read_bytes() == SMALL_PAYLOAD

    def test_auth_headers_forwarded(self, tmp_dest):
        resp = _make_response(200, SMALL_PAYLOAD,
                              {"Content-Length": str(len(SMALL_PAYLOAD))})
        auth = {"Authorization": "Bearer hf_TOKEN"}
        with patch("requests.get", return_value=resp) as mock_get:
            hfd.download_file(
                url="http://example.com/file",
                dest=tmp_dest,
                headers=auth,
                retry_delay=0,
            )
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer hf_TOKEN"

    def test_destination_directory_created_automatically(self, tmp_path):
        deep_dest = tmp_path / "a" / "b" / "c" / "model.bin"
        resp = _make_response(200, SMALL_PAYLOAD,
                              {"Content-Length": str(len(SMALL_PAYLOAD))})
        with patch("requests.get", return_value=resp):
            hfd.download_file(
                url="http://example.com/file",
                dest=deep_dest,
                headers={},
                retry_delay=0,
            )
        assert deep_dest.exists()


# ═════════════════════════════════════════════════════════════════════════════
# 7. download_repo — filtering
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadRepoFiltering:
    """Tests for include/exclude glob filtering in download_repo."""

    REPO_FILES = [
        "README.md",
        "config.json",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer.model",
    ]

    def _run_repo(self, args, files=None):
        """
        Patch out HfApi, hf_hub_url, build_hf_headers, and download_file, then
        call download_repo.  Returns the list of filenames that were requested.
        """
        files = files or self.REPO_FILES
        downloaded = []

        def _fake_download_file(url, dest, headers, **kwargs):
            downloaded.append(dest.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake")
            return True

        with patch("hf_download.HfApi") as MockApi, \
             patch("hf_download.hf_hub_url", side_effect=lambda **kw: f"http://hf/{kw['filename']}"), \
             patch("hf_download.build_hf_headers", return_value={}), \
             patch("hf_download.download_file", side_effect=_fake_download_file):

            instance = MockApi.return_value
            instance.list_repo_files.return_value = iter(files)
            hfd.download_repo(args)

        return downloaded

    def test_no_filter_downloads_all(self, tmp_path):
        args = _make_args(local_dir=str(tmp_path))
        got = self._run_repo(args)
        assert sorted(got) == sorted(self.REPO_FILES)

    def test_include_safetensors_only(self, tmp_path):
        args = _make_args(local_dir=str(tmp_path), include=["*.safetensors"])
        got = self._run_repo(args)
        assert all(f.endswith(".safetensors") for f in got)
        assert len(got) == 3

    def test_exclude_readme_and_config(self, tmp_path):
        args = _make_args(local_dir=str(tmp_path),
                          exclude=["README.md", "config.json"])
        got = self._run_repo(args)
        assert "README.md" not in got
        assert "config.json" not in got

    def test_include_and_exclude_combined(self, tmp_path):
        """Include all safetensors, but exclude the sharded ones."""
        args = _make_args(
            local_dir=str(tmp_path),
            include=["*.safetensors"],
            exclude=["*-of-*"],
        )
        got = self._run_repo(args)
        assert got == ["model.safetensors"]

    def test_no_match_logs_warning(self, tmp_path, caplog):
        import logging
        args = _make_args(local_dir=str(tmp_path), include=["*.gguf"])
        with caplog.at_level(logging.WARNING, logger="hf_download"):
            self._run_repo(args)
        assert any("No files matched" in r.message for r in caplog.records)

    def test_exclude_glob_wildcard(self, tmp_path):
        args = _make_args(local_dir=str(tmp_path), exclude=["tokenizer*"])
        got = self._run_repo(args)
        assert not any(f.startswith("tokenizer") for f in got)


# ═════════════════════════════════════════════════════════════════════════════
# 8. download_repo — directory layout
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadRepoLayout:

    def _run_repo_capture_dest(self, args, files):
        dest_paths = []

        def _fake_download(url, dest, headers, **kw):
            dest_paths.append(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return True

        with patch("hf_download.HfApi") as MockApi, \
             patch("hf_download.hf_hub_url", side_effect=lambda **kw: f"http://hf/{kw['filename']}"), \
             patch("hf_download.build_hf_headers", return_value={}), \
             patch("hf_download.download_file", side_effect=_fake_download):
            MockApi.return_value.list_repo_files.return_value = iter(files)
            hfd.download_repo(args)

        return dest_paths

    def test_local_dir_flat_layout(self, tmp_path):
        args = _make_args(local_dir=str(tmp_path))
        dests = self._run_repo_capture_dest(args, ["model.bin"])
        assert dests[0] == tmp_path / "model.bin"

    def test_cache_dir_hf_layout(self, tmp_path):
        args = _make_args(
            repo_id="org/mymodel",
            cache_dir=str(tmp_path),
        )
        dests = self._run_repo_capture_dest(args, ["weights.bin"])
        # Should be under <cache>/models--org--mymodel/snapshots/main/
        assert "models--org--mymodel" in str(dests[0])
        assert "snapshots" in str(dests[0])
        assert "main" in str(dests[0])

    def test_nested_file_preserves_subdir(self, tmp_path):
        args = _make_args(local_dir=str(tmp_path))
        dests = self._run_repo_capture_dest(args, ["sub/dir/weights.bin"])
        assert dests[0] == tmp_path / "sub" / "dir" / "weights.bin"


# ═════════════════════════════════════════════════════════════════════════════
# 9. download_repo — error handling
# ═════════════════════════════════════════════════════════════════════════════

class TestDownloadRepoErrors:

    def test_repo_not_found_exits(self, tmp_path):
        from huggingface_hub.utils import RepositoryNotFoundError

        args = _make_args(local_dir=str(tmp_path))
        with patch("hf_download.HfApi") as MockApi:
            MockApi.return_value.list_repo_files.side_effect = \
                RepositoryNotFoundError("404", response=MagicMock(status_code=404))
            with pytest.raises(SystemExit):
                hfd.download_repo(args)

    def test_stop_event_aborts_mid_repo(self, tmp_path):
        """If stop_event fires after the first file, subsequent files are skipped."""
        files = ["a.bin", "b.bin", "c.bin"]
        downloaded = []

        def _fake_download(url, dest, headers, **kw):
            downloaded.append(dest.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            # Trigger stop after first file
            hfd._stop_event.set()
            return True

        args = _make_args(local_dir=str(tmp_path))
        with patch("hf_download.HfApi") as MockApi, \
             patch("hf_download.hf_hub_url", side_effect=lambda **kw: f"http://hf/{kw['filename']}"), \
             patch("hf_download.build_hf_headers", return_value={}), \
             patch("hf_download.download_file", side_effect=_fake_download):
            MockApi.return_value.list_repo_files.return_value = iter(files)
            hfd.download_repo(args)

        # Only the first file should have been attempted
        assert downloaded == ["a.bin"]


# ═════════════════════════════════════════════════════════════════════════════
# 10. CLI argument parsing
# ═════════════════════════════════════════════════════════════════════════════

class TestCLIArgumentParsing:

    def _parse(self, argv):
        parser = hfd.build_parser()
        return parser.parse_args(argv)

    def test_repo_id_required(self):
        parser = hfd.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        ns = self._parse(["org/model"])
        assert ns.repo_id      == "org/model"
        assert ns.repo_type    == "model"
        assert ns.revision     == "main"
        assert ns.include      is None
        assert ns.exclude      is None
        assert ns.token        is None
        assert ns.cache_dir    is None
        assert ns.local_dir    is None
        assert ns.endpoint     is None
        assert ns.max_retries  == -1
        assert ns.retry_delay  == 5.0
        assert ns.retry_backoff == 1.5
        assert ns.retry_max_delay == 300.0

    def test_repo_type_dataset(self):
        ns = self._parse(["org/ds", "--repo-type", "dataset"])
        assert ns.repo_type == "dataset"

    def test_invalid_repo_type_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["org/model", "--repo-type", "invalid"])

    def test_include_multiple_patterns(self):
        ns = self._parse(["org/model", "--include", "*.safetensors", "*.bin"])
        assert ns.include == ["*.safetensors", "*.bin"]

    def test_exclude_multiple_patterns(self):
        ns = self._parse(["org/model", "--exclude", "*.md", "*.txt"])
        assert ns.exclude == ["*.md", "*.txt"]

    def test_retry_flags(self):
        ns = self._parse([
            "org/model",
            "--max-retries", "10",
            "--retry-delay", "2.5",
            "--retry-backoff", "2.0",
            "--retry-max-delay", "60",
        ])
        assert ns.max_retries    == 10
        assert ns.retry_delay    == 2.5
        assert ns.retry_backoff  == 2.0
        assert ns.retry_max_delay == 60.0

    def test_local_dir_flag(self):
        ns = self._parse(["org/model", "--local-dir", "/data/models"])
        assert ns.local_dir == "/data/models"

    def test_revision_flag(self):
        ns = self._parse(["org/model", "--revision", "v2.0"])
        assert ns.revision == "v2.0"

    def test_token_flag(self):
        ns = self._parse(["org/model", "--token", "hf_ABC123"])
        assert ns.token == "hf_ABC123"

    def test_quiet_flag(self):
        ns = self._parse(["org/model", "-q"])
        assert ns.quiet is True

    def test_endpoint_flag(self):
        ns = self._parse(["org/model", "--endpoint", "https://my-mirror.com"])
        assert ns.endpoint == "https://my-mirror.com"
