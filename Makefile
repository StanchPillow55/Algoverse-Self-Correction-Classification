.PHONY: env data eval analyze report

env:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .
	python -c "import sys,platform;print(sys.version);print(platform.platform())"

data:
	@echo "Datasets are CSVs. Use local files or GitHub blob URLs (auto-converted to raw)."

eval:
	export DEMO_MODE=1; python -m src.main run --dataset data/math20.csv --max-turns 2 --out outputs/demo.json --provider demo

analyze:
	python scripts/analyze_results.py --raw experimental-results/raw --out experimental-results

report: analyze
	@echo "Report updated at experimental-results/analysis.md"
