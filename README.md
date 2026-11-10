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

Tests:

```
python -m unittest discover tests
```
