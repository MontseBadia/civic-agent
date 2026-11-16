# Protocol

## 1. Purpose

Two questions will be answered:
- Does curated evidence beat closed-book answers? That's B vs A
- Does letting the model pick its own tools improve results over a fixed pipeline? That's C vs B

## 2. Frozen Setup

Setup is frozen before starting final run:
- Code: as of commit 6d7fc70
- Model: mlx-community/Qwen3.8-27B-4bit, revision
  10c35caafbb80f7dc6a7a432cdd11af10a6d4818, loaded from local
  snapshot path with HF_HUB_OFFLINE=1
- Server: mlx-vlm 0.7.8, --prefill-step-size 512
- Settings: configs/greedy.json (temperature 0, top_p 1.0, top_k 0, seed
  7, max_tokens 1024)
- prompt_hash 284dcde65f23, tools_hash 5e1f4807dc0d
- C loop limits: 6 tool calls, stop after 2 duplicate calls, 35,000-char
  evidence budget, 40,000-char result guard
- Snapshot: CIViC 01-Nov-2026, accepted evidence, manifest.json sha256 c2b56d0f5a8bf4fe0e19b0d005a643a790f0e00767c95d802e740362d0f7bb77
- Greedy reproducible across a server restart

## 3. Conditions

Three conditions are compared:
1. A closed-book: against the model internal knowledge, one JSON-mode call, no records
2. B pipeline: the model has access to CIViC following a fixed pipeline. There is an extraction call, then fixed lookups (using same tool functions as C), then a synthesis call. It follows fixed rules written by code:
  - no gene -> unknown_gene
  - several matches -> ambiguous unless exactly one match has the searched name as its own name, not alias
  - no variant or no rows -> no_evidence
3. C agentic: the model is allowed to choose among four tools (search_gene, search_variant, list_diseases, get_evidence), using a hand-written loop, then one JSON-mode call for the answer

Same model, settings, questions and answer format for the three conditions.

## 4. Question Set

Questions are split between dev and test:
- Dev: those have been used for config purposes
- Test: 40 templated questions (17 T1 sensitivity, 17 T2 resistance, 6 no evidence; 7 of those are in protein form) + 10 hand-made (2 disease wording, 2 gene alias, 2 ambiguous, 2 unknown gene, 2 free wording)
- Files: data/questions/test.jsonl, hand_test.jsonl, split.json (generated with seed 7 by evaluation/generate.py; hand-made written by hand and checked against snapshot)
- Gene split dev/test
- Each question's correct answer is built using filters: accepted, not flagged, predictive, supports, sensitivity/resistance, single drug or substitutes, single-variant profiles, exact disease, 1-5 drugs

## 5. Procedure

- One run per condition on test, in order A, B, C: python -m evaluation.run --condition X --questions data/questions/test.jsonl data/questions/hand_test.jsonl --config configs/greedy.json --label test
- Scored once with evaluation/score.py into results/test
- No retries of answers: unparseable answer counts as wrong
- Server error status or timeout: question counts as wrong
- A run that stops before every question has an answer, on server crash: rerun the condition from start, keep both logs, report it under Changes
- No question dropped or edited after the run

## 6. Correct answers

The following counts as correct answer:
- Answerable: not abstained, drug set equals the reference set (lowercased, generic names, no synonyms)
- Expected abstention: abstain true, expected reason, no drugs
- Server error, timeout or unparseable: wrong

## 7. Metrics

- Primary: per-question correctness, accuracy per condition
- Secondary, all automatic: drug precision/recall (micro), abstention counts, cited ID precision/recall, citation validity and invented-ID rate, claim support, retrieved ID recall (B, C) and lookup accuracy (B, C)
- Cost: time per question, model calls, tool calls, prompt and completion tokens
- Templated and hand-made reported separately; accuracy also by kind and hand made group
- One line per metric definition:
   -> Accuracy: share of questions answered correctly, under the rules in "Correct Answers"
   -> Drug precision: of all drugs given, across all questions, the share that are in the reference. Drugs given where an abstention was expected count as wrong
   -> Drug recall: of all reference drugs on answerable questions, the share that were given
   -> Cited ID precision / recall: overlap between cited evidence IDs and the reference IDs. Cited Does Not Support rows for the question's own profile and disease don't count
   -> Citation validity: share of cited IDs that exist and belong to the question's profile and disease. Any citation on a question without a profile is invalid
   -> Invented-ID rate: share of cited IDs not found in the snapshot
   -> Claim support: share of given drugs backed by a cited row with the right profile, disease, direction, significance and therapy.
   -> Retrieved-ID recall (B, C): share of reference rows the condition actually saw
   -> Lookup accuracy (B, C): share of questions where some get_evidence call used the question's profile and disease

## 8. Comparisons

- B vs A, C vs B, exact two-sided McNemar on per-question correctness,
  templated and hand-made separately
- Hand-made questions are suggestive only
- Primary tests: B vs A and C vs B on templated questions. Hand-made
  tests exploratory: p reported, no claims from them
- Level: 0.05 each, two-sided; the two answer separate questions
- C vs B read with cost next to it: a tie in accuracy means agency costs
  more for the same result

## 9. Predictions

Written before the test run:
- templated: A 10/40, B 38/40, C 40/40. B beats A clearly; C only slightly above B
- hand-made: A 2/10, B 8/10, C 10/10. C beats both A and B clearly

## 10. Failure Analysis and Manual Review

Every wrong answer gets one category, the earliest failing stage:
- format_error
- extraction_error
- retrieval_error
- abstention_error
- interpretation_error
- other

They can get sub-tags as well:
- alias_miss
- direction_misread
- right_drug_wrong_name

Manual review: 10 questions picked at random using seed 7 from all 50, same 10 for A, B and C. Picked before the run: test-04, test-05, test-07, test-10, test-21, test-24, test-26, test-35, test-38, hand-test-02.

Model explanations get an explanation tag: unsupported_claim, dropped_text_condition, clinical_advice, applied to right and wrong answers

## 11. Known Limitations

1. Reference answers come from CIViC, not biological truth. Disagreeing with CIViC might still be right
2. Some CIViC therapy names are drug classes (e.g. "mTOR Inhibitor"). Prompting for generic drug names can't be matched (dev-12 question)
3. Greedy config frozen. Model recommended sampling config gave no gain on dev, but only one draw was tested
4. Combined molecular profiles are out (AND/OR/NOT), 562 of 4,974
  accepted evidence items
5. Combination and Sequential therapies are out of templated questions
6. Assertions are not used
7. Not using drug synonyms, so A closed-book might look weaker than it is
8. Disease needs to match exact CIViC name. A Melanoma question does not pick up Skin Melanoma for instance
9. Questions with more than 5 drugs are skipped
10. Reference answers taken from structured fields
11. Only therapy questions (predictive, sensitivity/response, resistance). Prognostic and diagnostic evidence, reduced sensitivity and adverse response are not tested
12. One model (local Qwen) and one snapshot (01-Nov-2026) only
13. Small test set: 40 templated and 10 hand-made questions. McNemar has little power; C vs B on the hand-made cases is suggestive
14. Strict drug match: one extra or missing drug makes the answer wrong
15. B abstains by fixed code
16. B has no mapping from disease wording to CIViC names. On hand-made questions that word the disease differently, B abstains with no_evidence; C has the tool list_diseases to recover.

## Changes