from als_model import RecommendedSystem
from repository_my_sql import RepositoryMySql
from data_manager import DataManager
from predictor import Predictor
import torch
from utils import collate, complete_dataframe_with_names
import numpy as np
import pandas as pd

def take_proteins_unique(dataframe):
    proteins = dataframe[['protein_id','protein_structure']].drop_duplicates().values.tolist()
    return proteins 

repo = RepositoryMySql()
# data = RepositoryMySql().read_interactions_for_als()
predictor = Predictor()

# print(data.count)
model = RecommendedSystem()
data_manager = DataManager()

model.load_pipeline()
drug_id = 1054
# df = model.calculate_recommended_for_drug(drug_id,50)
df = model.calculate_recommended_proteins(50)
ids = df.toPandas().values.tolist()
interactions_with_structures =  repo.take_structures(ids)
proteins_to_analize = take_proteins_unique(interactions_with_structures)
data_manager.protein_graph_construct(proteins_to_analize)
dataset = data_manager.prepare_set(interactions_with_structures.values.tolist())
predictor.load_model("GAT")
test_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, collate_fn=collate)

prediction = predictor.make_prediction(test_loader)
print(prediction)
prediction = complete_dataframe_with_names(prediction) 
print(prediction)
prediction.to_excel(f"risultati_50.xlsx", index=False)
# prediction.to_excel(f"risultati_{drug_id}.xlsx", index=False)