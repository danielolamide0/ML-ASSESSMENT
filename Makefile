.PHONY: install test train train-ablation predict notebook all clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

train:
	$(PY) -m severity_triage.train

train-ablation:
	$(PY) -m severity_triage.train --exclude-internal-scores --model-path models/ablation_no_internal_scores/severity_model.joblib

predict:
	$(PY) -m severity_triage.predict --input data/holdback_dataset.csv --output predictions/holdback_predictions.csv --explain

notebook:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace severity_analysis.ipynb --ExecutePreprocessor.timeout=900

# Full reproducible run: tests, model, ablation, predictions, notebook
all: test train train-ablation predict notebook

clean:
	rm -rf models/*.joblib models/*.json models/*.npy models/ablation_no_internal_scores predictions/*.csv predictions/*.json reports/figures/*.png
