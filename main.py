from als_model import RecommendedSystem
from repository_my_sql import RepositoryMySql
from data_manager import DataManager
from predictor import Predictor
import torch
from utils import collate
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GAT_THRESHOLD = 0.50

pairs_to_check = [
    (15, "P05026"), (15, "P49721"), (118, "P07919"), (125, "P15954"), (24, "P35625"),
    (146, "Q15375"), (9, "P51451"), (131, "Q9UM73"), (134, "O00206"), (63, "P00156"),
    (52, "Q16665"), (417, "P43405"), (24, "Q53HL2"), (24, "Q9H6I2"), (15, "Q9UBE0"),
    (24, "Q9H7Z6"), (24, "O15151"), (9, "Q9UGP5"), (15, "Q9BZ29"), (24, "P78536"),
    (15, "Q14524"), (24, "Q9NVV9"), (24, "Q9H3D4"), (15, "O15144"), (24, "Q15796"),
    (9, "P42338"), (24, "P15927"), (1841, "P49356"), (9, "P56381"), (45, "P56381"),
    (9, "P13489"), (24, "P62888"), (15, "P19525"), (9, "P09622"), (118, "P14927"),
    (15, "P14927"), (24, "P42892"), (24, "Q99496"), (24, "P35222"), (9, "P35222"),
    (52, "Q9GZT9"), (1523, "P00403"), (1066, "P00403"), (125, "P00403"), (9, "P35221"),
    (15, "Q16877"), (24, "Q15843"), (417, "Q05655"), (1094, "Q92731"), (565, "Q92731"),
    (640, "Q92731"), (417, "P40763"), (15, "Q7L804"), (24, "Q96I99"), (325, "Q16623"),
    (15, "P63165"), (24, "P18146"), (417, "Q14289"), (9, "Q96IV0"), (9, "O00459"),
    (15, "Q9UII2"), (15, "Q13418"), (24, "P32969"), (1063, "P53611"), (1066, "P20674"),
    (15, "Q8TF76"), (24, "Q9BQD3"), (24, "P04637"), (77, "P36888"), (1523, "P14854"),
    (15, "Q9Y5K5"), (24, "Q12888"), (9, "O00443"), (424, "Q8IUR0"), (15, "P43405"),
    (77, "P43405"), (24, "P51451"), (417, "P51451"), (24, "Q02543"), (325, "P60880"),
    (9, "P12694"), (15, "P12694"), (9, "P21953"), (15, "P21953"), (24, "P62913"),
    (24, "Q9UI42"), (15, "P54652"), (24, "Q9H0R8"), (15, "P54619"), (9, "P55072"),
    (118, "P31930"), (1107, "P31930"), (561, "P31930"), (15, "Q9NRR6"), (15, "Q9UBT2"),
    (15, "Q8TAA3"), (15, "P59998"), (15, "Q99436"), (118, "P00156"), (15, "P00156"),
    (561, "P00156"), (24, "P02511"), (24, "Q96CA5"), (24, "P62266"), (24, "Q9H8S9"),
    (24, "Q13625"), (417, "P29353"), (24, "Q02878"), (9, "Q9UPN6"), (24, "P22681"),
    (417, "P22681"), (424, "Q9Y5R8"), (24, "P61218"), (286, "P35916"), (59, "P35916"),
    (1742, "P06401"), (9, "Q8WWV3"), (286, "P35968"), (59, "P35968"), (9, "O14949"),
    (1066, "O14949"), (15, "O14949"), (24, "Q13620"), (24, "P62753"), (24, "O15164"),
    (1066, "P07919"), (1094, "P03372"), (565, "P03372"), (640, "P03372"), (1742, "P03372"),
    (146, "P07333"), (24, "Q13191"), (417, "Q13191"), (134, "Q9Y6Y9"), (24, "P26368"),
    (24, "P62829"), (15, "Q9Y478"), (1841, "P49354"), (24, "P30876"), (1282, "Q5FBB7"),
    (15, "P54760"), (567, "P54760"), (24, "Q13490"), (1107, "Q9UDW1"), (9, "Q9UDW1"),
    (1066, "Q9UDW1"), (15, "Q9UDW1"), (1063, "Q92696"), (15, "O95149"), (15, "Q9UGP5"),
    (15, "Q99418"), (24, "P16333"), (417, "P16333"), (15, "Q04917"), (45, "P25705"),
    (24, "O94972"), (24, "Q9NRG4"), (9, "Q08334"), (1523, "P15954"), (1066, "P15954"),
    (15, "Q12906"), (24, "P27694"), (15, "O15530"), (15, "P49327"), (9, "P08238"),
    (15, "P08238"), (118, "P08574"), (15, "P08574"), (561, "P08574"), (9, "P14616"),
    (15, "P14616"), (1282, "P30153")
]

df_target_pairs = pd.DataFrame(pairs_to_check, columns=["drug_id", "protein_id"])

def take_proteins_unique(dataframe):
    proteins = dataframe[['protein_id','protein_structure']].drop_duplicates().values.tolist()
    return proteins 

def decode_if_bytes(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value

def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)

def build_real_test_set(data, model, train_weight=0.8, test_weight=0.2, seed=42):
    spark_data = model.spark.createDataFrame(data)
    _, test = spark_data.randomSplit([train_weight, test_weight], seed=seed)
    return test.toPandas()

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

def calculate_model_aucs(test_data, model, repo, predictor, data_manager):
    evaluation_pairs = test_data[["drug_id", "protein_id"]].drop_duplicates()
    known_proteins = set(model.indexer_model.labels)
    evaluation_pairs = evaluation_pairs[
        evaluation_pairs["protein_id"].isin(known_proteins)
    ].copy()

    als_predictions = model.predict_for_pairs(
        evaluation_pairs[["drug_id", "protein_id"]].values.tolist()
    ).toPandas().rename(columns={"prediction": "als_prediction"})

    evaluation_with_structures = repo.take_structures(
        evaluation_pairs[["drug_id", "protein_id"]].values.tolist()
    )
    evaluation_with_structures = pd.merge(
        evaluation_with_structures,
        test_data[["drug_id", "protein_id", "amount_interactions"]],
        on=["drug_id", "protein_id"],
        how="inner",
    )

    gat_predictions = make_gat_predictions(
        evaluation_with_structures,
        predictor,
        data_manager,
    )

    scores = pd.merge(
        test_data[["drug_id", "protein_id", "amount_interactions"]],
        als_predictions,
        on=["drug_id", "protein_id"],
        how="inner",
    )
    scores = pd.merge(
        scores,
        gat_predictions,
        on=["drug_id", "protein_id"],
        how="inner",
    )
    scores["pipeline_prediction"] = scores["als_prediction"] * scores["gat_prediction"]

    aucs = {
        "ALS": safe_auc(scores["amount_interactions"], scores["als_prediction"]),
        "GAT": safe_auc(scores["amount_interactions"], scores["gat_prediction"]),
        "Pipeline": safe_auc(scores["amount_interactions"], scores["pipeline_prediction"]),
    }
    return aucs, scores

repo = RepositoryMySql()
# data = RepositoryMySql().read_interactions_for_als()
predictor = Predictor()

# print(data.count)
model = RecommendedSystem()
data_manager = DataManager()

model.load_pipeline()
predictor.load_model("GAT")

# print(model.als_model.itemFactors.count())
# print(model.als_model.userFactors.count())
drug_ids = [
    15, 118, 125, 24, 146, 9, 131, 134, 63, 52, 
    417, 1841, 45, 1523, 1066, 1094, 565, 640, 325, 
    1063, 77, 424, 1107, 561, 286, 59, 1742, 1282, 567
]
# df = model.predict_for_pairs(pairs_to_check)
# df = df.toPandas().sort_values(by='prediction')
# print(df)

df = model.calculate_recommended_for_drug(drug_ids,500)
# df = model.calculate_recommended_proteins(50)
df_check = df.toPandas()
# # 2. Facciamo il merge (equivalente della join)
df_found_ALS = pd.merge(
    df_check, 
    df_target_pairs, 
    on=["drug_id", "protein_id"], 
    how="inner"
)

# 3. Vediamo i risultati
print(f"Coppie trovate: {len(df_found_ALS)}")
print(df_found_ALS)

ids = df.toPandas().values.tolist()
interactions_with_structures =  repo.take_structures(ids)
prediction = make_gat_predictions(interactions_with_structures, predictor, data_manager)
print(prediction)
print(interactions_with_structures)

df_final = pd.merge(
    df_check.rename(columns={"rating": "als_prediction"}),
    prediction,
    on=["drug_id", "protein_id"],
    how="inner",
)
df_final["pipeline_prediction"] = df_final["als_prediction"] * df_final["gat_prediction"]
df_final = df_final[df_final["gat_prediction"] >= GAT_THRESHOLD].copy()
df_final = df_final.sort_values(
    by=["pipeline_prediction", "gat_prediction", "als_prediction"],
    ascending=False,
)
print(f"Coppie dopo filtro GAT >= {GAT_THRESHOLD}: {len(df_final)}")
print(df_final)

evaluation_data = repo.read_interactions_for_als_2()
test_data = build_real_test_set(evaluation_data, model)
aucs, auc_scores = calculate_model_aucs(
    test_data,
    model,
    repo,
    predictor,
    data_manager,
)
print(f"Test set reale: {len(test_data)} righe")
print("AUC sul test set reale:")
for model_name, auc in aucs.items():
    print(f"{model_name}: {auc}")
auc_scores.to_csv("auc_scores.csv", index=False)

# # Visualizziamo le prime righe
# print(df_final.head())
# print(df_final)

# # 2. Facciamo il merge (equivalente della join)
# df_found = pd.merge(
#     df_final, 
#     df_target_pairs, 
#     on=["drug_id", "protein_id"], 
#     how="inner"
# )

# # 3. Vediamo i risultati
# print(f"Coppie trovate: {len(df_found)}")
# print(df_found.head(20))

# #prediction = completedataframe_with_names(prediction) 
# # print(prediction)
# # prediction.to_excel(f"risultati_50.xlsx", index=False)
# # prediction.to_excel(f"risultati_{drug_id}.xlsx", index=False)
