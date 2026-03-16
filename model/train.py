import pandas as pd 
import numpy as np 
import torch 
from torch import nn,optim
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from model import NeuralNetwork
from preprocess import FeatureBuilder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

model=NeuralNetwork()


df1=pd.read_csv("../Data/final_data.csv")

x=df1.drop("csatscore",axis=1)
y=df1["csatscore"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y)

bnes=['category',
        'subcategory',
        'agentname',
        'sentiment_score',
        'issuemonth',
        'issuerespondedmonth']
ohes=['channelname', 
      'tenurebucket', 
      'agentshift', 
      'responsehourscategory']
index_remove=[0,3,8,13]
ct=ColumnTransformer(
            [
                ("ohe",OneHotEncoder(),ohes),
                ("bnes",BinaryEncoder(),bnes),
            ],
            remainder="passthrough"
        )

class Data(Dataset):
    def __init__(self,x,y):
        super().__init__()
        x=ct.fit_transform(x)
        cols=[col for col in range(x.shape[1]) if col not in index_remove]
        x=x[:,cols]
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y.values,dtype=torch.float32)
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.x.shape[0]

feature_builder = FeatureBuilder()
x_train = feature_builder.fit_transform(x_train)
x_test = feature_builder.transform(x_test)
train_data=Data(x_train,y_train)
test_data=Data(x_test,y_test)
    
train_dataloader=DataLoader(
    dataset=train_data,
    batch_size=16,
    drop_last=False,
    shuffle=True
)
test_dataloader=DataLoader(
    dataset=test_data,
    shuffle=False,
    drop_last=True,
    batch_size=16,
)

device=torch.device("cpu")
loss_fn=nn.L1Loss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

def predict():
    model.eval()
    running_test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.float().to(device)
            y = y.float().unsqueeze(1).to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            running_test_loss += loss.item()

            pred = torch.clamp(pred, 1, 5)
            pred_round = torch.round(pred)

            all_preds.extend(pred_round.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    avg_test_loss = running_test_loss / len(test_dataloader)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_acc = (all_preds == all_targets).mean() * 100

    return avg_test_loss, test_acc, all_preds, all_targets


epochs = 20
trainloss = []
trainaccs = []
testlosses = []
testaccs = []

model.to(device)

for i in range(epochs):
    model.train()
    running_train_loss = 0.0
    total, correct = 0, 0

    for x, y in tqdm(train_dataloader, total=len(train_dataloader)):
        x = x.float().to(device)
        y = y.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        pred = model(x)
        loss = loss_fn(pred, y)

        running_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = torch.clamp(pred, 1, 5)
        pred_round = torch.round(pred)

        total += y.size(0)
        correct += (pred_round == y).sum().item()

    avg_train_loss = running_train_loss / len(train_dataloader)
    train_acc = correct / total * 100

    test_loss, test_acc, y_pred, y_actual = predict()

    trainaccs.append(train_acc)
    trainloss.append(avg_train_loss)
    testaccs.append(test_acc)
    testlosses.append(test_loss)

    print(
        f"Epoch: {i+1}/{epochs}\t| "
        f"Train loss: {avg_train_loss:.4f}\t| "
        f"Test loss: {test_loss:.4f}\t| "
        f"Train acc: {train_acc:.4f}\t| "
        f"Test acc: {test_acc:.4f}"
    )
    
test_loss, test_acc, y_pred, y_actual = predict()

print("Rounded Accuracy:", test_acc)
print("MAE:", mean_absolute_error(y_actual, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_actual, y_pred)))
print("Within ±1:", (np.abs(y_actual - y_pred) <= 1).mean() * 100)


artifacts = {
    "ct": ct,               
    "bnes": bnes,           
    "ohes": ohes,              
    "index_remove": index_remove,  
}

joblib.dump(artifacts, "preprocessor_artifacts.pkl")    
print("Training successfull")