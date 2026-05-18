.PHONY: app benchmark cost report summary excel

app:
	streamlit run app.py

benchmark:
	python experiments/sequential_learner.py

cost:
	python experiments/cost_benchmark.py

report:
	python scripts/export_summary.py

excel:
	python scripts/export_excel.py
