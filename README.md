# civic-agent

Work in progress.

A CLI assistant that answers cancer-variant questions from CIViC evidence,
and an evaluation of three ways to do it: closed-book, a fixed retrieval
pipeline, and an agent that picks its own tool calls.

Not clinical advice.

## Setup

Python 3.13.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`: any OpenAI-compatible endpoint works, hosted or local.

Local model: Qwen3.8-27B (4-bit MLX) served by mlx-vlm, in its own venv.
The project only talks to it over HTTP.

```
python3.13 -m venv ~/.venvs/mlx-server
~/.venvs/mlx-server/bin/pip install mlx-vlm==0.7.8
~/.venvs/mlx-server/bin/python -m mlx_vlm.server --model mlx-community/Qwen3.8-27B-4bit --port 8080
```

Apple Silicon only. ~15 GB download on first start.

CIViC snapshot used for evaluation (release 01-Nov-2026, accepted evidence only):

```
python data/download_civic.py
```

Writes the TSVs and a manifest.json (URLs, sha256, download time) to data/raw/01-Nov-2026/.

Tests:

```
python -m unittest discover tests
```
