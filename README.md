# CIViC Agent

This project has two parts:

1. **A CLI assistant** that answers cancer-variant questions from CIViC evidence
2. **An evaluation** comparing three approaches to answering those questions,
   to investigate the value of evidence retrieval and model-controlled tool use

### Research questions

For the evaluation phase, the following two questions are answered:

> 1. Does curated CIViC evidence improve accuracy over closed-book
>    answering (B vs A)?

> 2. Does letting the model choose its own lookups improve results
>    over a fixed pipeline (C vs B)?

### Approaches compared

| Condition | Gets CIViC evidence | Who picks the lookups |
|---|---|---|
| A, closed-book | No | None |
| B, fixed pipeline | Yes | Fixed code |
| C, agent | Yes | The model |

Example question: "Which therapies have CIViC evidence items supporting
sensitivity in Melanoma with BRAF V600K?"

*This is not clinical advice.*

## Evaluation

Protocol, committed before the test run: [docs/protocol.md](docs/protocol.md).

Test results:
- [Summary](results/test/summary.json): metrics, McNemar tests, cost
- [Per-question results](results/test/per_question.jsonl): scoring per question and condition
- [Failure analysis](results/test/failure_tags.csv): manually classified wrong answers

The run logs these are scored from (`runs/`) are kept locally.

## Repository structure

```text
agent/          the assistant: snapshot lookups, tools, LLM client, prompts, tool loop, conditions
evaluation/     question generator, runner, scorer
data/           download script, question set (snapshot goes to data/raw/, not committed)
configs/        run settings (greedy.json is the frozen one)
docs/           evaluation protocol
results/test/   scored test results
tests/          unit tests on a made-up snapshot, no model or API calls
```

## Install

Using Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`: any OpenAI-compatible endpoint works, hosted or local.
For the local server below:
- `LLM_BASE_URL=http://localhost:8080/v1`
- `LLM_API_KEY` empty
- `LLM_MODEL` the snapshot path (set after the download)

## Local model

Qwen3.8-27B (4-bit MLX) served by mlx-vlm, in its own venv.

The project only talks to it over an OpenAI-compatible HTTP API. Pinned
to one revision: the server loads whatever model each request names, and
a Hugging Face name silently pulls the latest revision. So the server
runs from the downloaded snapshot, offline.

```bash
python3.13 -m venv ~/.venvs/mlx-server
~/.venvs/mlx-server/bin/pip install mlx-vlm==0.7.8
~/.venvs/mlx-server/bin/hf download mlx-community/Qwen3.8-27B-4bit \
  --revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818
```

The download prints the local snapshot path. Set `LLM_MODEL` in `.env`
to that path and start the server with it, offline:

```bash
HF_HUB_OFFLINE=1 ~/.venvs/mlx-server/bin/python -m mlx_vlm.server \
  --model <snapshot path> --port 8080 --prefill-step-size 512
```

Apple Silicon only, ~15 GB download. On a 24 GB Mac the default prefill
step (2048) runs out of GPU memory on ~10k-token prompts.

## Data

CIViC snapshot used for evaluation (release 01-Nov-2026, accepted evidence only):

```bash
python data/download_civic.py
```

Writes the TSVs and a `manifest.json` (URLs, sha256, download time) to
`data/raw/01-Nov-2026/`.

## Questions

Question set is already in `data/questions/`. Two types of questions used:
- Dev questions: used to tune prompts
- Test questions: run once, after the protocol was committed

The split is made by gene, so no gene appears in both splits.

Templated questions and reference answers (dev and test) were generated
with the command below. The generator doesn't overwrite existing files.

```bash
python -m evaluation.generate \
  --snapshot data/raw/01-Nov-2026 --seed 7 --out data/questions
```

Hand-made questions (`data/questions/hand_dev.jsonl`, `hand_test.jsonl`)
are written by hand.

## Experiments

Each condition is run over one or more question files (server up):

```bash
for c in A B C; do
  python -m evaluation.run --condition $c \
    --questions data/questions/dev.jsonl data/questions/hand_dev.jsonl \
    --config configs/greedy.json --label dev
done
```

The above writes one log per condition, `runs/<time>_dev_<condition>.jsonl`. The
config holds every setting that changes results.

Time per question on the local model:
- A -> about 10 s
- B -> about 20 s
- C -> about 2 min (50 test questions: C about 2 hours).

The three runs are then scored together (one log per condition):

```bash
python -m evaluation.score \
  --questions data/questions/dev.jsonl data/questions/hand_dev.jsonl \
  --runs runs/<A>.jsonl runs/<B>.jsonl runs/<C>.jsonl \
  --snapshot data/raw/01-Nov-2026 --out results/dev
```

Writes `per_question.jsonl` and `summary.json` to `results/dev/` and prints the report.

Test run: same commands with `--label test`, `--out results/test` and
the test files `data/questions/test.jsonl` and `data/questions/hand_test.jsonl`.

`results/test/` already holds the committed results and the scorer
doesn't overwrite, so score into another folder to reproduce them.

## Tests

Tests are run with:

```bash
python -m unittest discover tests
```