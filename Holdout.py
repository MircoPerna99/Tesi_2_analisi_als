from als_model import RecommendedSystem
from repository_my_sql import RepositoryMySql
from data_manager import DataManager
from predictor import Predictor
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
import torch
from utils import collate
import numpy as np
import pandas as pd


TEST_FRACTION = 0.20
SEED = 42
CANDIDATE_POOL = 500
TOP_K_VALUES = [10, 20, 50, 100]
GAT_THRESHOLD = 0.50


def take_proteins_unique(dataframe):
    return dataframe[["protein_id", "protein_structure"]].drop_duplicates().values.tolist()


def decode_if_bytes(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def split_positive_holdout(interactions, test_fraction=TEST_FRACTION, seed=SEED):
    positives = interactions[interactions["amount_interactions"] > 0].copy()
    positives["amount_interactions"] = 1.0

    test_parts = []
    train_parts = []
    rng = np.random.default_rng(seed)

    for _, group in positives.groupby("drug_id"):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        if len(group) < 2:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(len(group) * test_fraction)))
        n_test = min(n_test, len(group) - 1)
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else positives.iloc[0:0].copy()
    return train, test


def train_als_for_holdout(model, train_data):
    spark_train = model.spark.createDataFrame(train_data)
    indexer_model = StringIndexer(
        inputCol="protein_id",
        outputCol="protein_id_index",
    ).fit(spark_train)
    indexed_train = indexer_model.transform(spark_train)

    als = ALS(
        maxIter=10,
        regParam=model._config["reg_param"],
        rank=model._config["rank"],
        alpha=model._config["alpha"],
        userCol="drug_id",
        itemCol="protein_id_index",
        ratingCol="amount_interactions",
        coldStartStrategy="drop",
    )

    model.indexer_model = indexer_model
    model.als_model = als.fit(indexed_train)


def recommend_for_drugs(model, drug_ids, candidate_pool=CANDIDATE_POOL):
    subset = model.spark.createDataFrame([(int(drug_id),) for drug_id in drug_ids], ["drug_id"])
    recommendations = model.als_model.recommendForUserSubset(subset, candidate_pool)
    recommendations = recommendations.select(
        "drug_id",
        explode("recommendations").alias("rec"),
    ).select(
        "drug_id",
        "rec.protein_id_index",
        "rec.rating",
    )

    converter = IndexToString(
        inputCol="protein_id_index",
        outputCol="protein_id",
        labels=model.indexer_model.labels,
    )
    return converter.transform(recommendations).select("drug_id", "protein_id", "rating").toPandas()


def make_gat_predictions(interactions, predictor, data_manager, batch_size=500):
    proteins_to_analize = take_proteins_unique(interactions)
    data_manager.protein_graph_construct(proteins_to_analize)
    dataset = data_manager.prepare_set(interactions.values.tolist())
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    prediction = predictor.make_prediction(test_loader)
    interactions_for_merge = interactions.copy()
    interactions_for_merge["drug_structure"] = interactions_for_merge["drug_structure"].apply(decode_if_bytes)
    drug_ids_frame = interactions_for_merge[["drug_id", "drug_structure"]].drop_duplicates()

    return pd.merge(
        prediction,
        drug_ids_frame,
        on="drug_structure",
        how="inner",
    )[["drug_id", "protein_id", "prediction"]].rename(
        columns={"prediction": "gat_prediction"}
    )


def add_pipeline_scores(candidates, repo, predictor, data_manager):
    ids = candidates[["drug_id", "protein_id"]].drop_duplicates().values.tolist()
    interactions_with_structures = repo.take_structures(ids)
    gat_predictions = make_gat_predictions(interactions_with_structures, predictor, data_manager)

    scored = pd.merge(
        candidates.rename(columns={"rating": "als_prediction"}),
        gat_predictions,
        on=["drug_id", "protein_id"],
        how="inner",
    )
    scored["pipeline_prediction"] = scored["als_prediction"] * scored["gat_prediction"]
    return scored


def remove_train_positives(candidates, train_data):
    train_pairs = train_data[["drug_id", "protein_id"]].drop_duplicates()
    return pd.merge(
        candidates,
        train_pairs.assign(is_train_positive=True),
        on=["drug_id", "protein_id"],
        how="left",
    ).query("is_train_positive != True").drop(columns=["is_train_positive"])


def dcg_at_k(hits):
    return sum(hit / np.log2(index + 2) for index, hit in enumerate(hits))


def ranking_metrics_at_k(scored, test_data, score_col, k):
    test_by_drug = (
        test_data.groupby("drug_id")["protein_id"]
        .apply(lambda values: set(values.tolist()))
        .to_dict()
    )
    ranked = scored.sort_values(["drug_id", score_col], ascending=[True, False])

    hits = []
    recalls = []
    precisions = []
    ndcgs = []
    reciprocal_ranks = []

    for drug_id, true_proteins in test_by_drug.items():
        recommendations = ranked[ranked["drug_id"] == drug_id]["protein_id"].head(k).tolist()
        if not recommendations:
            hits.append(0.0)
            recalls.append(0.0)
            precisions.append(0.0)
            ndcgs.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        hit_list = [1.0 if protein_id in true_proteins else 0.0 for protein_id in recommendations]
        n_hits = sum(hit_list)
        ideal_hits = [1.0] * min(len(true_proteins), k)
        ideal_dcg = dcg_at_k(ideal_hits)
        first_hit_rank = next((i + 1 for i, hit in enumerate(hit_list) if hit), None)

        hits.append(1.0 if n_hits > 0 else 0.0)
        recalls.append(n_hits / len(true_proteins))
        precisions.append(n_hits / k)
        ndcgs.append(dcg_at_k(hit_list) / ideal_dcg if ideal_dcg else 0.0)
        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    return {
        "score": score_col,
        "k": k,
        "drugs_evaluated": len(test_by_drug),
        "hit_rate": float(np.mean(hits)),
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "ndcg": float(np.mean(ndcgs)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def evaluate_rankings(scored, test_data):
    rows = []
    score_columns = ["als_prediction", "gat_prediction", "pipeline_prediction"]
    scored_with_gat_filter = scored[scored["gat_prediction"] >= GAT_THRESHOLD].copy()

    for score_col in score_columns:
        for k in TOP_K_VALUES:
            rows.append(ranking_metrics_at_k(scored, test_data, score_col, k))

    for k in TOP_K_VALUES:
        metric = ranking_metrics_at_k(scored_with_gat_filter, test_data, "pipeline_prediction", k)
        metric["score"] = f"pipeline_prediction_gat_filter_{GAT_THRESHOLD}"
        rows.append(metric)

    return pd.DataFrame(rows)


repo = RepositoryMySql()
predictor = Predictor()
model = RecommendedSystem()
data_manager = DataManager()

data = repo.read_interactions_for_als()
train_data, test_data = split_positive_holdout(data)
print(f"Train positivi: {len(train_data)}")
print(f"Test positivi nascosti: {len(test_data)}")

train_als_for_holdout(model, train_data)
predictor.load_model("GAT")

known_proteins = set(model.indexer_model.labels)
raw_test_size = len(test_data)
test_data = test_data[test_data["protein_id"].isin(known_proteins)].copy()
print(f"Test positivi valutabili: {len(test_data)}")
print(f"Test positivi cold-start esclusi: {raw_test_size - len(test_data)}")

test_drug_ids = sorted(test_data["drug_id"].unique().tolist())
candidates = recommend_for_drugs(model, test_drug_ids)
candidates = remove_train_positives(candidates, train_data)
print(f"Candidati ALS dopo rimozione positivi train: {len(candidates)}")

scored_candidates = add_pipeline_scores(candidates, repo, predictor, data_manager)
metrics = evaluate_rankings(scored_candidates, test_data)

scored_candidates.to_csv("holdout_ranked_candidates.csv", index=False)
metrics.to_csv("holdout_metrics.csv", index=False)

print("Metriche holdout ranking:")
print(metrics)
