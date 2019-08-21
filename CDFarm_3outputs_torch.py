# CDFarm_3outputs_torch
#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

#%%
df = pd.read_excel(r'N:\agpo\work1\Shang\ForPyomoBook\nn_CDFarm_torch_multiobj_test.xlsx', header = 0)
df_array = df.to_numpy()

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection="3d")
#%%
fig = plt.figure()
ax = plt.axes(projection="3d")

# Data for a three-dimensional line
zdata = df_array[:, 6]
xdata = df_array[:, 0]
ydata = df_array[:, 2]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
plt.show()


#%%
# create a class for custom dataloader
class DatasetCDFarm(Dataset): 

    def __init__(self, file_path):
        df = pd.read_excel(file_path, header = 0)
        df_array = df.to_numpy() # transform fd to numpy array
        self.len = df_array.shape[0]
        # First three columns contains the features X
        x = torch.from_numpy(df_array[:, 0:3]) # prices of barley, rapeseed, wheat 
        self.x = (x - torch.mean(x))/torch.sqrt(torch.var(x))
        # The rest columns contains the labels Y
        y = torch.from_numpy(df_array[:, 3:7]) 
        self.y = (y - torch.mean(y))/torch.sqrt(torch.var(y))# profit
        print('x:',self.x)
        print('y:', self.y)
    #  PyTorch gives you the freedom to pretty much do anything with the Dataset class,
    #  so long as you override two of the subclass functions:  

    def __getitem__(self, index):  
        # returns the data and labels 
        return self.x[index], self.y[index]

    def __len__(self): 
        # return the size of the dataset, so that torch can divide data into batches
        return self.len

# Load train and test data
train_dataset = DatasetCDFarm(r'N:\agpo\work1\Shang\ForPyomoBook\nn_CDFarm_torch_multiobj_train.xlsx')
train_loader = DataLoader(dataset=train_dataset, 
                        batch_size = 4, 
                        shuffle = True) # shuffle: mix the data randomly


# create Net class: construct the Neural Network
class Net(nn.Module): 
    #class initialization 

    def __init__(self, input_size, hidden1_size,hidden2_size, output_size): 
        super(Net, self).__init__() # super fconstructor creates an instance of the base nn.Module 
    
        self.fc1 = nn.Linear(input_size, hidden1_size) # first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # output layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)
    #define how data flows in this network, needs to be defined for each network
    def forward(self, x):
        # pass data through the net
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out) 
        out = self.relu2(out)
        out = self.fc3(out)
        return out
# print this network architecture:
MyNet = Net(3, 50, 20, 4)
print(MyNet)


# Construct our loss function and an Optimizer. The call to model.parameters()
criterion = nn.MSELoss()
optimizer = optim.Adam(MyNet.parameters(), lr=0.01)
epochs = 100
loss_values = []    

for epoch in range(epochs):
    cum_loss = 0
   
    MyNet.train()
    for batch_id, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs = Variable(inputs).float()
        labels = Variable(labels).float()

        # print(epoch, batch_id, "inputs", inputs.data, "labels", labels.data)
        # Forward pass
                        
        out = MyNet(inputs)
        # print('out', type(out),out)
        # print('labels', type(labels),labels)

        loss = criterion(out, labels)
        print(epoch, batch_id, loss.data)
        cum_loss += loss.data

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_values.append(cum_loss/len(train_dataset))

plt.plot(loss_values)




test_dataset = DatasetCDFarm(r'N:\agpo\work1\Shang\ForPyomoBook\nn_CDFarm_torch_multiobj_test.xlsx')
test_loader = DataLoader(dataset=test_dataset) 
#%%
def test():
    MyNet.eval()
    loss = 0
    for inputs, labels in test_loader:
        inputs = Variable(inputs).float()
        labels = Variable(labels).float()
        print('labels:', labels)
        out = MyNet(inputs)
        print('out:', out)
        loss += criterion(out, labels).item()
        print('loss:', loss)
    Average_loss = loss/len(test_loader.dataset)
    print("Average loss:", Average_loss)

test()

print('Square rooted loss:', torch.sqrt(loss))


#%%
