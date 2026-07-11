from als_model import RecommendedSystem
from repository_my_sql import RepositoryMySql
from data_manager import DataManager
from predictor import Predictor
import torch
from utils import collate
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

data = repo.read_interactions_for_als_3()
print(data.count())
model.init_data(data)
model.hyperparameter_tuning()

