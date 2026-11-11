.PHONY: run test clean report arch
run:        ## full pipeline (ingest -> ... -> evaluate)
	python run_pipeline.py
test:
	python -m unittest discover -s tests -v
arch:       ## re-render architecture diagram (needs Graphviz `dot`)
	python -c "from pathlib import Path; from src.reporting.diagrams import render_architecture as r; r(Path('reports/figures'))"
report:     ## build the 3-page PDF report from the latest results
	python -m src.reporting.final_report
clean:
	rm -rf mlruns models/*.pkl models/run_registry.json data/raw/* data/interim/* data/processed/* data/features/* data/errors/*
