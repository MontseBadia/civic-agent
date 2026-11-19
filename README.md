# CIViC Agent

![tests](https://github.com/MontseBadia/civic-agent/actions/workflows/tests.yml/badge.svg)

This project has two parts:

1. **A CLI assistant** that answers cancer-variant questions from CIViC evidence
2. **An evaluation** comparing three approaches to answering those questions,
   to investigate the value of evidence retrieval and model-controlled tool use

*Research software. Not clinical advice.*

## CLI Assistant

Example question:

```text
"Which therapies have CIViC evidence items supporting sensitivity in Melanoma with BRAF V600K?"
```

Example answer (agent response):

```text
Drugs: Vemurafenib, Dabrafenib, Trametinib
Explanation: CIViC evidence items support sensitivity of vemurafenib (EID1399, EID1400, EID4180), dabrafenib (EID2505), and trametinib (EID2506) in melanoma with BRAF V600K. Combination items (EID6179, EID6939, EID6941, EID6965) are excluded because they are combination therapies.

Evidence cited
EID      Disease   Type        Significance          Direction  Level  Citation                          Link
EID1399  Melanoma  Predictive  Sensitivity/Response  Supports   B      McArthur et al., 2014 (24508103)  https://civicdb.org/links/evidence_items/1399
EID2505  Melanoma  Predictive  Sensitivity/Response  Supports   B      Long et al., 2011 (21343559)      https://civicdb.org/links/evidence_items/2505
```

*Output shortened: the full table has five rows, one per cited item.*

The evidence table comes straight from the CIViC snapshot, not from the
model. Setup and usage are under How To below.

## Evaluation

### Research Questions

For the evaluation phase, the following two questions are answered:

> 1. Does curated CIViC evidence improve accuracy over closed-book
>    answering (B vs A)?

> 2. Does letting the model choose its own lookups improve results
>    over a fixed pipeline (C vs B)?

### Approaches Compared

| Condition | Gets CIViC evidence | Who picks the lookups |
|---|---|---|
| A, closed-book | No | None |
| B, fixed pipeline | Yes | Fixed code |
| C, agent | Yes | The model |

### Results

Protocol, committed before the test run: [docs/protocol.md](docs/protocol.md).

Write-up: [docs/writeup.md](docs/writeup.md)

Test results:

| | A | B | C |
|---|---:|---:|---:|
| Templated (40) | 7 | 35 | 39 |
| Hand-made (10) | 2 | 8 | 10 |
| Mean time per templated question | 11 s | 19 s | 139 s |

- [Summary](results/test/summary.json): metrics, McNemar tests, cost
- [Per-question results](results/test/per_question.jsonl): scoring per question and condition
- [Failure analysis](results/test/failure_tags.csv): manually classified wrong answers
- [Manual review](results/test/manual_review.csv): explanation tags for 10 questions per condition

The run logs these are scored from (`runs/`) are kept locally.

Conclusion: Evidence helps (B vs A, p < 0.001). The agent's gain over the fixed pipeline is not significant (C vs B, p = 0.125) and costs about 7x the time. Two of C's four templated wins involve evidence statements that do not explicitly support the exact variant, although the structured CIViC records qualify.

## Repository Structure

```text
agent/          the assistant: CLI, snapshot lookups, tools, LLM client, prompts, tool loop, conditions
evaluation/     question generator, runner, scorer
data/           download script, question set (snapshot goes to data/raw/, not committed)
configs/        run settings (greedy.json is the frozen one)
docs/           evaluation protocol, write-up
results/test/   scored test results
tests/          unit tests on a made-up snapshot, no model or API calls
```

## How To

### Setup

Using Python 3.13. Install dependencies and configure model endpoint:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`: any OpenAI-compatible endpoint works, hosted or local. Reproducing the reported evaluation requires the pinned local model and configuration described under Local Model.
For the local server:
- `LLM_BASE_URL=http://localhost:8080/v1`
- `LLM_API_KEY` empty
- `LLM_MODEL` the snapshot path (set after the download)

### Local Model

Qwen3.8-27B (4-bit MLX) served locally through `mlx-vlm==0.7.8` in its own venv.

The assistant communicates with the model through an OpenAI-compatible HTTP API. Pinned
to one revision: the server loads whatever model each request names, and
a Hugging Face name silently pulls the latest revision. So the server
runs from the downloaded snapshot, offline.

```bash
python3.13 -m venv ~/.venvs/mlx-server
~/.venvs/mlx-server/bin/pip install mlx-vlm==0.7.8 huggingface_hub==2.4.0
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

### CIViC Data

Download the CIViC snapshot used for evaluation (release 01-Nov-2026, accepted evidence only):

```bash
python data/download_civic.py
```

The above writes the TSVs and a `manifest.json` (URLs, sha256, download timestamps) to
`data/raw/01-Nov-2026/`.

### Run CLI Assistant

One question per call, same settings and snapshot as the evaluation (server up):

```bash
python -m agent.cli --condition C --config configs/greedy.json \
  "Which therapies have CIViC evidence items supporting sensitivity in Melanoma with BRAF V600K?"
```

In the command above, `--condition` picks A, B or C (see Approaches Compared); C is the agent.

Prints the answer and one row per cited evidence item, taken from the
snapshot. Each question is logged to `runs/cli/`.

### Evaluation Questions

The question set is in `data/questions/`, split by gene into dev and test:
- Dev questions: used to tune prompts
- Test questions: run once, after the protocol was committed

Templated questions and reference answers (dev and test) were generated
with the command below. `data/questions` already holds the questions, so use a new folder to regenerate those:

```bash
python -m evaluation.generate \
  --snapshot data/raw/01-Nov-2026 --seed 7 --out data/questions_regenerate
```

Hand-made questions (`data/questions/hand_dev.jsonl`, `hand_test.jsonl`)
are written by hand.

### Run Evaluation

The frozen run configuration is found in `configs/greedy.json` and the evaluation protocol is documented in `docs/protocol.md`.

Run each condition over one or more question files (server up):

```bash
for c in A B C; do
  python -m evaluation.run --condition $c \
    --questions data/questions/test.jsonl data/questions/hand_test.jsonl \
    --config configs/greedy.json --label test
done
```

The above writes one log per condition, `runs/<time>_test_<condition>.jsonl`. The
config holds every setting that changes results. C takes about 2 hours for the 50 test questions.

Score the three runs together (one log per condition). `results/test/` already holds the committed results, so use a new folder to reproduce results:

```bash
python -m evaluation.score \
  --questions data/questions/test.jsonl data/questions/hand_test.jsonl \
  --runs runs/<time>_test_A.jsonl runs/<time>_test_B.jsonl runs/<time>_test_C.jsonl \
  --snapshot data/raw/01-Nov-2026 --out results/regenerate
```

The above writes `per_question.jsonl` and `summary.json` to `results/regenerate/` and prints the report, containing aggregate metrics, statistical comparisons and per-question results.

### Run Tests

Tests are run with:

```bash
python -m unittest discover tests
```