from model import CPI_GAT_NFM_GLOBAL, CPI_GCN_NFM_GLOBAL
import torch
from utils import predicting_rec

class Predictor():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available() 
                        else "cpu")
    
    def define_model(self, model_type):
        if(model_type == 'GCN'):
            return CPI_GCN_NFM_GLOBAL()
        else:
            return CPI_GAT_NFM_GLOBAL()
    
    def define_model_path(self, model_type):
        if(model_type == 'GCN'):
            return "/Users/mircoperna/Desktop/ALS/models/model_CPI_GAT_NFM_GLOBAL_human_1.model"
        else:
            return "/Users/mircoperna/Desktop/ALS/models/model_CPI_GAT_NFM_GLOBAL_human_1.model"
    
    def load_model(self, model_type = 'GCN'):
        self.model_p = self.define_model(model_type)
        self.model_p.to(self.device)
        path = self.define_model_path(model_type)
        self.model_p.load_state_dict(torch.load(path , map_location=self.device))
        # test_T, test_P = predicting(model_p, self.device, loader)
        
    def make_prediction(self, loader):
       return predicting_rec(self.model_p, self.device, loader)
        