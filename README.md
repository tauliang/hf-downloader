# hf-downloader

A resilient command-line tool for downloading large models and datasets from
HuggingFace Hub. Built for multi-hundred-gigabyte models over unstable
connections — it resumes partial downloads seamlessly, retries forever with
exponential back-off, and accepts every flag that `huggingface-cli download`
does.



## Features

- **Resumable downloads** — uses HTTP `Range` headers to pick up exactly where
  a previous run left off. Partial files are never discarded.
- **Infinite retries** — keeps retrying on any network or server error, with
  configurable exponential back-off capped at a maximum delay.
- **Graceful interruption** — `Ctrl-C` / `SIGTERM` flushes in-flight data to
  disk and exits cleanly. Re-run the same command to continue.
- **Full `huggingface-cli` parity** — supports `--repo-type`, `--revision`,
  `--include`/`--exclude` glob filters, `--token`, `--cache-dir`,
  `--local-dir`, and `--endpoint`.
- **SHA-256 verification** — optional integrity check after each file lands.
- **Live progress bar** — shows percentage, bytes, speed, and ETA per file.



## Requirements

- Python 3.9 or later
- See `requirements.txt`


## Installation

```bash
# 1. Clone or copy this project
git clone https://github.com/your-org/hf-downloader.git
cd hf-downloader

# 2. Install dependencies
pip install -r requirements.txt
```

No package installation is required — run `hf_download.py` directly.



## Quick Start

```bash
# Download a full model (resumes automatically if interrupted)
python hf_download.py meta-llama/Llama-3-70B-Instruct

# Only the safetensors shards
python hf_download.py meta-llama/Llama-3-70B-Instruct --include "*.safetensors"

# A dataset
python hf_download.py HuggingFaceFW/fineweb --repo-type dataset

# Save to a specific directory
python hf_download.py bigscience/bloom --local-dir /mnt/nvme/bloom

# Authenticate with a token
python hf_download.py meta-llama/Meta-Llama-3-8B --token hf_XXXX
# or export HUGGING_FACE_HUB_TOKEN=hf_XXXX
```



## Usage

```
python hf_download.py <repo_id> [OPTIONS]
```

### Positional argument

| Argument  | Description                                          |
|-----------|------------------------------------------------------|
| `repo_id` | HuggingFace repo, e.g. `meta-llama/Llama-3-70B-Instruct` |

### HuggingFace options

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-type` | `model` | `model`, `dataset`, or `space` |
| `--revision` | `main` | Branch, tag, or commit SHA |
| `--include PATTERN …` | _(all)_ | Glob patterns for files to include |
| `--exclude PATTERN …` | _(none)_ | Glob patterns for files to exclude |
| `--token TOKEN` | _(env)_ | HuggingFace token (or `HUGGING_FACE_HUB_TOKEN`) |
| `--cache-dir PATH` | HF default | Override the HF cache directory |
| `--local-dir PATH` | _(cache)_ | Download flat into this directory |
| `--endpoint URL` | HF Hub | Custom Hub endpoint |
| `-q`, `--quiet` | off | Suppress info / progress output |

### Retry / resilience options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-retries N` | `-1` (∞) | Max attempts per file; `-1` = retry forever |
| `--retry-delay S` | `5.0` | Initial wait between retries (seconds) |
| `--retry-backoff X` | `1.5` | Exponential back-off multiplier |
| `--retry-max-delay S` | `300.0` | Maximum wait cap (seconds) |



## How resuming works

Every file is written in append mode (`ab`). Before each request the tool
checks the size of any existing partial file and sends a
`Range: bytes=<size>-` header. The server responds with `206 Partial Content`
and the remaining bytes are appended directly. If the server returns
`416 Range Not Satisfiable` the file is already complete.

On retry after a failure the same logic applies — the tool re-checks the
on-disk size and resumes from that point, so no data is ever re-downloaded.



## Authentication

Private models and gated repositories require a token:

```bash
# Option 1 — flag
python hf_download.py org/private-model --token hf_XXXX

# Option 2 — environment variable (recommended for scripts)
export HUGGING_FACE_HUB_TOKEN=hf_XXXX
python hf_download.py org/private-model
```

Generate a token at <https://huggingface.co/settings/tokens>.



## Running the tests

```bash
pytest test_hf_download.py -v
```

The test suite (51 tests, 10 classes) runs entirely offline — all network
calls are mocked. Coverage includes:

- Helper utilities (`_fmt_bytes`, `_sha256_of_file`, …)
- Happy-path downloads (200 and 206 responses)
- SHA-256 verification pass and fail
- Retry on connection errors, timeouts, and HTTP 5xx
- Back-off timing and `retry_max_delay` cap
- Interruption mid-stream (stop event), partial file preservation
- Full resume workflow: interrupt → verify on-disk state → resume → verify final bytes
- `--include` / `--exclude` glob filtering
- Cache-dir and `--local-dir` directory layout
- `RepositoryNotFoundError` exit handling
- All 14 CLI flags and their defaults



## Project structure

```
hf-downloader/
├── hf_download.py        # Main downloader — run this
├── test_hf_download.py   # pytest test suite
├── requirements.txt      # Runtime + test dependencies
└── README.md             # This file
```



## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
