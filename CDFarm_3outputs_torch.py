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
### see how inputs and outputs are related
df = pd.read_excel('nn_CDFarm_torch_multiobj_train.xlsx', header = 0)
df_array = df.to_numpy()

for i in range(3, 7):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df_array[:, 0]
    y = df_array[:, 1]
    z = df_array[:, 2]

    c = df_array[:, i] 

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())

    plt.title(str('y'+ str(i)))

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylabel('x3')

    fig.colorbar(img)
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
train_dataset = DatasetCDFarm('nn_CDFarm_torch_multiobj_train.xlsx')
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




test_dataset = DatasetCDFarm('nn_CDFarm_torch_multiobj_test.xlsx')
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
