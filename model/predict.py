import joblib
from model import NeuralNetwork
from preprocess import FeatureBuilder
import os 
import torch
import numpy as np
import pandas as pd

BASEDIR=os.path.dirname(__file__)
model_state_path=os.path.join(BASEDIR,"model_weights.pkl")
prerpocessor=os.path.join(BASEDIR,"preprocessor_artifacts.pkl")

#load model
device=torch.device("cpu")
model=NeuralNetwork()
model.load_state_dict(joblib.load(model_state_path))
model.to(device)
model.eval()

#load preprocessing
prprocessartifacts=joblib.load(prerpocessor)
bnes=prprocessartifacts["bnes"]
ohes=prprocessartifacts["ohes"]

#Feature Builder
featurebuiler=FeatureBuilder()


def predict(sample:dict):
    sample=pd.DataFrame([sample])
    sample=featurebuiler.transform(sample)
    sample=prprocessartifacts["ct"].transform(sample)
    cols=[col for col in range(sample.shape[1]) if col not in prprocessartifacts["index_remove"]]
    sample=sample[:,cols]
    sample = torch.tensor(sample, dtype=torch.float32).to(device)
    with torch.no_grad():
        output=model(sample)
        output = torch.clamp(output, 1, 5)
        return int(torch.round(output).item())
    

if __name__== "__main__":
    sample = {
    "channelname": "Inbound",
    "category": "Order Related",
    "subcategory": "Delayed",
    "issuereportedat": "2023-02-08 10:44:00",
    "issueresponded": "2023-02-08 11:14:00",
    "surveyresponsedate": "2023-08-02 00:00:00",
    "agentname": "Stanley Hogan",
    "supervisor": "Harper Wong",
    "manager": "Emily Chen",
    "tenurebucket": ">90",
    "agentshift": "Split",
    "customerremarks":"Very Bad"
    }
    
    print(predict(sample))