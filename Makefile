.PHONY: train app test clean install setup

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

setup: install
	mkdir -p data/raw data/processed models reports
	@echo "✅ Project ready. Run 'make train' next."

# ── ML pipeline ───────────────────────────────────────────────────────────────
train:
	python -m src.train

evaluate:
	python -m src.evaluate

# ── App ───────────────────────────────────────────────────────────────────────
app:
	streamlit run app/main.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Clean ─────────────────────────────────────────────────────────────────────
clean-models:
	rm -f models/*.joblib models/*.json
	@echo "Models cleared."

clean-data:
	rm -f data/processed/*.csv
	@echo "Processed data cleared."

clean-reports:
	rm -f reports/*.png reports/*.json
	@echo "Reports cleared."

clean: clean-models clean-data clean-reports
	@echo "✅ All artifacts cleared. Run 'make train' to rebuild."
