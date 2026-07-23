# Historical CER report status

The previous file mixed literature numbers, local experiments, estimates, and
unverified Atlas claims, so it must not be used as release evidence.

No reproducible CER result is committed because the corresponding source audio,
portable manifest, exact model revision, and prediction file are not all
available in the repository.

Generate a new evidence-bearing JSON and Markdown report with:

```bash
python modules/02_asr_model_training/scripts/03_evaluate_baseline.py \
  --manifest /path/to/test.jsonl \
  --data-root /path/to/audio-root \
  --output reports/baseline-evaluation.json
```

A valid report must contain all predictions, failed samples, corpus-weighted
edit counts, model identity, and the manifest/artifact provenance described in
`docs/TRAINING_AND_EVALUATION.md`.
