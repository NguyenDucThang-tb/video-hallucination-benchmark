# Experimental Protocol

## Scope

Experiment 1 measures hallucination on VidHalluc, VideoHallucer, and
EventHallusion. Experiment 2 measures whether mitigation changes ordinary
video understanding on TempCompass, TVBench, VideoMME, and MVBench.

No row named "Our Method" is emitted until a separately specified and
implemented method exists. SEASON is listed as SEASON, not as our method.

## Deterministic input

Every video is decoded once through `src.data.sampler.sample_video`. Eight
indices are produced by rounding an inclusive linear spacing from frame 0 to
frame N-1. When N < 8, nearest indices repeat. The manifest records path,
indices, actual frame count, FPS, duration, and policy. Methods receive the
already sampled array; they cannot resample the source video.

TCD takes a chronological subset of positions from these eight frames. It
does not open the video again and does not shuffle or reverse frames.

## Generation

- Greedy decoding
- `do_sample=false`
- `temperature=0`
- one beam
- identical prompt and maximum output length within each comparison

Any benchmark-specific deviation is written to prediction metadata and is a
separate protocol, never silently mixed into the 8-frame table.

## Evaluation

- VidHalluc reports BQA, MCQ, STH, TSH and macro-average over available tasks.
- VideoHallucer reports strict pair accuracy: both basic and hallucination
  questions in a pair must be correct.
- EventHallusion reports binary accuracy per split and overall. Description
  judging requires `OPENAI_API_KEY`; it is skipped without that variable.
  Main results use the corrected `mix` reference (`event_info.unexpected`).
- Temporal understanding average is mean(TempCompass, TVBench).
- Conventional understanding average is mean(VideoMME, MVBench).

Missing or unparseable outputs remain present in audit data and are not
silently removed. Tables report valid, missing, and parser-error counts.

## Reproducibility and resume

Each output records model checkpoint, upstream commits, method/sampling/
generation config, runtime, peak GPU memory when available, raw text, parser
state, and errors. Resume skips only records with no error and a valid parse.

Current pinned shallow-clone commits:

- VideoHallucer: `8b785d1680465911cd2ce80c9f652837c0ba2abd`
- EventHallusion: `aa544c21c7cd93b4685423cb94f77ab441f754bc`
- VidHalluc: `e753864f5c2500c38523f97992355e2352bf8732`
