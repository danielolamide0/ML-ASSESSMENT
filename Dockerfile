FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY data ./data
COPY models ./models

# Score a file: docker run --rm -v $PWD/out:/app/predictions severity-triage
ENTRYPOINT ["python", "-m", "severity_triage.predict"]
CMD ["--input", "data/holdback_dataset.csv", "--output", "predictions/holdback_predictions.csv", "--explain"]
