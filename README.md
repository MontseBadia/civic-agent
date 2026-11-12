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
The project only talks to it over HTTP. Pinned to one revision: the
server loads whatever model each request names, and a Hugging Face name
pulls the latest revision.

```
python3.13 -m venv ~/.venvs/mlx-server
~/.venvs/mlx-server/bin/pip install mlx-vlm==0.7.8
~/.venvs/mlx-server/bin/hf download mlx-community/Qwen3.8-27B-4bit --revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818
```

The download prints the local snapshot path. Set `LLM_MODEL` in `.env` to
that path and start the server with it, offline:

```
HF_HUB_OFFLINE=1 ~/.venvs/mlx-server/bin/python -m mlx_vlm.server --model <snapshot path> --port 8080 --prefill-step-size 512
```

Apple Silicon only, ~15 GB download. On a 24 GB Mac the default prefill
step (2048) runs out of GPU memory on ~10k-token prompts.

CIViC snapshot used for evaluation (release 01-Nov-2026, accepted evidence only):

```
python data/download_civic.py
```

Writes the TSVs and a manifest.json (URLs, sha256, download time) to data/raw/01-Nov-2026/.

Templated questions and reference answers (dev and test):

```
python -m evaluation.generate --snapshot data/raw/01-Nov-2026 --seed 7 --out data/questions
```

Run one condition over a question file (server up):

```
python -m evaluation.run --condition B --questions <file.jsonl> --config configs/greedy.json --label dev
```

Writes runs/<time>_dev_B.jsonl. The config holds every setting that changes results.

Tests:

```
python -m unittest discover tests
```
