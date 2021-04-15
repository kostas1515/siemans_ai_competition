import os
from subprocess import call
from tqdm import tqdm
dataset='trainingdata_c'
directory = "../refined_scripts"
for filename in tqdm(os.listdir(os.path.join(directory,dataset))):
    if  filename.endswith(".py"):
        exe = os.path.join(directory,dataset,filename)
        call(["python",exe])