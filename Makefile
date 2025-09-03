.PHONY: env data eval analyze report setup smoke reproduce figures tests

# Backward-compatible env target
env:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || true
	python -c "import sys,platform;print(sys.version);print(platform.platform())"

data:
	@echo "Datasets are CSVs. Use local files or GitHub blob URLs (auto-converted to raw)."

eval:
	export DEMO_MODE=1; python -m src.main run --dataset data/math20.csv --max-turns 2 --out outputs/demo.json --provider demo

analyze:
	python scripts/analyze_results.py --raw experimental-results/raw --out experimental-results

report: analyze
	@echo "Report updated at experimental-results/analysis.md"

# New: paper/eval rig targets
setup:
	@echo "==> Creating venv and upgrading pip"
	@/bin/bash -lc 'set -euo pipefail; if [ ! -d .venv ]; then python3.11 -m venv .venv || python3 -m venv .venv; fi'
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; python -m pip install -U pip wheel setuptools pip-tools'
	@echo "==> Locking requirements (pip-compile)"
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; (pip-compile --generate-hashes -q -o requirements.lock.txt requirements.txt || true)'
	@echo "==> Minimal dev deps for smoke"
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; python -m pip install -U pytest pytest-cov pyyaml pandas'
	@echo "setup: OK"

smoke:
	@echo "==> Smoke: dataset availability checks"
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; python scripts/check_datasets.py || true'
	@echo "==> Smoke: running baseline/full_system (demo mode)"
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; DEMO_MODE=1 PROVIDER=demo RUN_ID=baseline OPENAI_TEMPERATURE=0.2 SEEDS=1,2,3 python -m src.main run --dataset humaneval --subset subset_20 --max-turns 2 --out runs/smoke/heval20_traces.json --provider $$PROVIDER'
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; DEMO_MODE=1 PROVIDER=demo RUN_ID=full_system OPENAI_TEMPERATURE=0.2 SEEDS=1,2,3 python -m src.main run --dataset data/smoke/gsm8k16.csv --max-turns 2 --out runs/smoke/gsm8k16_traces.json --provider $$PROVIDER'
	@echo "==> Smoke: generating plot and table"
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; python scripts/analyze_results.py --raw runs/smoke --out reports/smoke && ls -1 reports/smoke/figures | head -5'
	@echo "smoke: OK"

figures:
	@echo "==> Ensuring system diagram exists"
	@/bin/bash -lc 'test -f reports/figures/system_diagram.svg && echo OK || (echo MISSING && exit 1)'
	@echo "figures: OK"

reproduce:
	@echo "==> Reproduce (demo seeds only)"
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; DEMO_MODE=1 PROVIDER=demo RUN_ID=baseline python -m src.main run --dataset humaneval --subset subset_100 --max-turns 2 --out runs/repro/heval100.json --provider $$PROVIDER'
	@/bin/bash -lc 'set -euo pipefail; . .venv/bin/activate; DEMO_MODE=1 PROVIDER=demo RUN_ID=full_system python -m src.main run --dataset data/smoke/gsm8k16.csv --max-turns 3 --out runs/repro/gsm8k16.json --provider $$PROVIDER'
	@echo "reproduce: OK"

# Unit + style + harness tests (light)
.tests:
	@/bin/bash -lc '. .venv/bin/activate; pytest -q'

tests: .tests
	@echo "tests: OK"
