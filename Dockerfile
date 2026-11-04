FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /Users/mircoperna/Desktop/ALS
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 'pyspark==4.1.2'

COPY Holdout_2.py als_model.py configuration.py repository_my_sql.py offline_repository.py ./
COPY data_manager.py predictor.py utils.py model.py fm.py nfm.py ./
COPY test_positive_interactions_ids_ratio_1.csv ./
COPY models/ ./models/
COPY saved_als_model_2/ ./saved_als_model_2/
COPY saved_indexer_2/ ./saved_indexer_2/

ENV PYTHONPATH=/Users/mircoperna/Desktop/ALS \
    HOLDOUT_OFFLINE_DATA_DIR=/offline_data \
    ALS_CONFIG_PATH=/offline_data/config.json
WORKDIR /results
CMD ["python", "/Users/mircoperna/Desktop/ALS/Holdout_2.py"]
