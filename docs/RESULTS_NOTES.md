# RESULTS NOTES

- The new paper-ready writer emits:
  - runs/{DATE}__{dataset}__{arm}__{model}__seed{S}__t{TEMP}__mt{MAXT}/
  - config.json, metrics.json (mean ± 95% CI), traces.json
  - per-turn artifacts in he/ and gsm8k/
- Harness versions are pinned via src/utils/harness_parity.py
- Cost estimation uses placeholder pricing in src/utils/cost.py and can be extended with real logs

