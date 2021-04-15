import pandas as pd
import numpy as np
import tpot
from tpot import TPOTClassifier
import os

root="./data"
# datasets = ["trainingdata_a.csv","trainingdata_b.csv","trainingdata_c.csv"]
datasets = ["trainingdata_a.csv"]
for dataset in datasets:
    checkpoint_folder=os.path.join('checkpoints',dataset.split('.')[0])
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)
    path = os.path.join(root,dataset)
    data_train=pd.read_csv(path)
    x_train=np.array([data_train['x_i1'],data_train['x_i2']]).T
    y_train=np.array([data_train['l_i']]).T.ravel()
    pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=10,
                                         verbosity=2,scoring='f1',n_jobs=-1,periodic_checkpoint_folder=checkpoint_folder)
    pipeline_optimizer.fit(x_train, y_train)
    pipeline_optimizer.export(checkpoint_folder+"_pipeline_final.py")
    

