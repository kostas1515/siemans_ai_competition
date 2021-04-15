import numpy as np 
import pandas as pd 
import os

dataset="trainingdata_c"
directory='../synth_vote_dataset'
path = os.path.join(directory,dataset)
l_0=0
l_1=0
k=0
for filename in os.listdir(path):
    if  filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(path,filename))
        l_0 = l_0 + np.array(df['l_0'])
        l_1 = l_1 + np.array(df['l_1'])
        k=k+1
l0_final=l_0/k
l1_final=l_1/k
final_dataset=pd.DataFrame()
final_dataset["x_i1"] = df["x_i1"]
final_dataset["x_i2"] = df["x_i2"]
final_dataset["l_0"] = l0_final
final_dataset["l_1"] = l1_final
final_dataset.to_csv(os.path.join(directory,dataset)+".csv")