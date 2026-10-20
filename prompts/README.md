# Prompt skeletons

CosMet's two components are prompted exactly as in the original study (TSE):
a system prompt with instructions and few-shot examples, and a user prompt
with the text to process. The files here are the *skeletons*: the structure of
the two system prompts and of the two user prompts, with the example blocks replaced by
`<<... -- withheld>>` placeholders. The public examples of the original study (training sets T1 and
T2 in `prompt.py` of the CosMet repository) are referenced, not duplicated. As in the original
study, the Splitter prompt is a one-line instruction followed by examples, and the Analyzer prompt
consists of worked examples only: its output format (the `measurement` block with triggering event,
functional user and, per step, `FU – DG – OOI – movement type`) is defined by the examples and is the
"output schema" referred to in Sec. V-B of the paper. Instructions, field names and movement types
are in English; the company examples are in Italian.

## Few-shot examples

| Component         | Original study (GPT-4, 8k context) | This deployment (gpt-oss-20B, 131k context) |
|-------------------|-----------------------------------:|--------------------------------------------:|
| Sentence Splitter | T1: 16 example sentences (English) | 30 = the 16 of T1 + 14 Italian sentences from the calibration set |
| COSMIC Analyzer   | T2: 7 worked use cases (English)   | 12 = the 7 of T2 + 5 Italian worked use cases from the calibration set |

Rationale: the original study had to leave >5,200 of GPT-4's 8,192 tokens for
the use case and the answer, which capped T2 at 7 examples. That constraint
disappeared with gpt-oss-20B; the Analyzer prompt now runs to 8-12k tokens.
Adding examples beyond 12 gave no measurable gain on the calibration set.

## Language

Kiranet's use cases are written in **Italian**. Instructions, the output
schema and the COSMIC vocabulary (E/X/R/W, Entry/Exit/Read/Write) stay in
English as in the original study; the added examples, the sub-processes
produced by the Splitter, and the names of functional users, data groups and
objects of interest are in Italian, as they appear in the use case. Each
Italian example is paired with its accepted manual COSMIC analysis, prepared
by a senior company measurer and reviewed by the first three authors (Sec. V-A).
BERTScore on the Italian labels uses `bert-base-multilingual-cased`.

## Selection material

A calibration set of 32 use cases (~190 CFP; Sec. V-B) from a product completed and
measured by the company in 2024, DISJOINT from the five portfolio projects,
so nothing in the evaluation was seen in the prompts. On this set we chose:
- the 14 + 5 Italian examples (coverage of the company template's recurring
  patterns: validation messages, lookups on codes, batch exchanges with
  external systems);
- the reasoning effort: `low` produced more decomposition errors; `high`
  more than doubled the latency (median 7.0 s vs 3.1 s per functional process
  on the calibration set, `data/calibration_log.csv`) without a clear accuracy
  benefit over `medium`; hence `medium` (Sec. V-B).

## Freezing

Prompts, checkpoint revision, vLLM version and request parameters were
frozen on 19 Dec 2025 (end of Phase 0) and used unchanged for all 10
evaluation runs on all five projects. The files in this folder are public templates: same instruction, structure and output format as the
operational prompts, with the example blocks replaced by placeholders. They are NOT identical to the
prompts actually evaluated (which contain the 14 + 5 confidential Italian examples) and are not released
as such; no digest of the operational prompt files is provided.

## Prompt sizes and token counts

Approximately 3,000 tokens (Splitter) and 8,000–12,000 tokens (Analyzer) are `usage.prompt_tokens`
as reported by vLLM, i.e. the gpt-oss tokenizer (`o200k_harmony`) applied to the whole Harmony-formatted
request: system prompt (instructions, schema, examples) PLUS the user prompt with the use case or the
split process. The range for the Analyzer reflects the length of the input process. The output budget
(`max_tokens` 8,192) is shared between reasoning and final answer; the maximum observed completion was
well below it (`analysis/timing.py`: no `finish_reason = length` in the 10,000 evaluation calls).
