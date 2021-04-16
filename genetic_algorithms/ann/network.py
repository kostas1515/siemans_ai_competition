import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np 

data=pd.read_csv("./synth_vote_dataset/trainingdata_a.csv")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2) 

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        return torch.sigmoid(x)

k=0
net = Net()
net=net.cuda()
criterion=torch.nn.MSELoss(reduction='mean')
net=net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
input=np.array([data['x_i1'],data['x_i2']]).T
input = torch.tensor(input,device='cuda',dtype=torch.float)
gt= np.array([data['l_0'],data['l_1']]).T
gt = torch.tensor(gt,device='cuda',dtype=torch.float)
loss=torch.tensor(100)
while loss.item()>0.001:
    k=k+1
    optimizer.zero_grad()
    out=net(input)
    loss = criterion(out,gt)
    loss.backward()
    optimizer.step()
    if k>10000:
        break
k=0
criterion=torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
while loss.item()>1000:
    k=k+1
    optimizer.zero_grad()
    out=net(input)
    loss = criterion(out,gt)
    loss.backward()
    optimizer.step()
    if k>10000:
        break
k=0
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
while loss.item()>500:
    k=k+1
    optimizer.zero_grad()
    out=net(input)
    loss = criterion(out,gt)
    loss.backward()
    optimizer.step()
    if k>10000:
        break
    
torch.save(net.state_dict(), './neta.pt')