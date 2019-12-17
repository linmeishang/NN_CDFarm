# CDFarm_3outputs+classification_torch
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
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from hyperopt import fmin, tpe, hp
import pickle
import os
import sys
import hyperopt.pyll.stochastic
#%%
### see how data looks like
# df = pd.read_excel(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_cornersolution_train.xlsx', header = 0)
# df_array = df.to_numpy()
# print(df_array)

# for i in range(3, 13):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     x = df_array[:, 0]
#     y = df_array[:, 1]
#     z = df_array[:, 2]

#     c = df_array[:, i] 

#     img = ax.scatter(x, y, z, c=c, cmap='viridis', alpha=0.5)

#     plt.title(str('y'+ str(i)))

#     plt.xlabel('x1')
#     plt.ylabel('x2')
#     plt.ylabel('x3')

#     fig.colorbar(img)
#     plt.show()


#%%
# Data preparation: access to feasture and lables and normalize them
def datapreparation(file_path):

    df = pd.read_excel(file_path, header = 0)
    df = df.mask(df < 0.1, 0) # convert very small numbers into 0 using df.mask
    df_array = df.to_numpy() # first transform df to numpy array    

    # Access features and normalize them
    x = torch.from_numpy(df_array[:, 0:3]) # transform numpy array to tensors
    picklename = "scaling_metrics.pkl"
    if not os.path.exists(picklename):
        x_mean = torch.mean(x)
        x_sd = torch.sqrt(torch.var(x))
        export = (x_mean,x_sd)
        pickle.dump( export, open( picklename, "wb" ) )
    elif os.path.exists(picklename):
        (x_mean,x_sd) = pickle.load( open( picklename, "rb" ) )
    else:
        print("Error in Scaling Metrics")
        sys.exit()

    x = (x - x_mean)/x_sd # normalization
    

    # Access lables and normalize them
    y = torch.from_numpy(df_array[:, 3:13])
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    y[y==0]= -5
    # print('Type and shape of x', x.type, x.size)
    # print('Type and shape of y', y.type y.size)

    #print(x, y)
    print('Type and shape of x', x.type, x.shape)
    print('Type and shape of y', y.type, y.shape)
    return [x, y]
#%%

train_file = datapreparation(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_cornersolution_train.xlsx')
validation_file = datapreparation(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_cornersolution_validation.xlsx')
test_file = datapreparation(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_cornersolution_test.xlsx')
print(train_file[0]) # X
print(trian_file[1]) # y

#%%
# create a class for custom dataloader
class DatasetCDFarm(Dataset): 

    def __init__(self, file):
        
       
        self.x = file[0]
        self.y = file[1]  
      
        
        # print('x:',self.x)
        # print('y:', self.y)
        
        print('shape of x: ', self.x.shape)
        print('shape of y: ', self.y.shape)
      
    #  PyTorch gives you the freedom to pretty much do anything with the Dataset class,
    #  so long as you override two of the subclass functions:  

    def __getitem__(self, index):  
        # returns the data and labels 
        return self.x[index], self.y[index]

    def __len__(self): 
        # return the size of the dataset, so that torch can divide data into batches
        self.len = self.x.shape[0]
        return self.len

#%%
# Load train and validation data
train_dataset = DatasetCDFarm(train_file)
train_loader = DataLoader(dataset=train_dataset, 
                        batch_size = 4, 
                        shuffle = True) # shuffle: mix the data randomly_dataset = DatasetCDFarm(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_cornersolution_train.xlsx')
validation_dataset = DatasetCDFarm(validation_file)
validation_loader = DataLoader(dataset=validation_dataset, 
                        batch_size = len(validation_dataset)) 
test_dataset = DatasetCDFarm(test_file)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset)) 

#%%
next(iter(train_loader))
#%%
for i, batch in enumerate(train_loader):
    print(i, batch)
#%%
# create Net class: construct the Neural Network
class Net(nn.Module): 
    #class initialization 

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size): 
        super(Net, self).__init__() # super fconstructor creates an instance of the base nn.Module 
    
        self.fc1 = nn.Linear(input_size, hidden1_size) # first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # output layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)
    #define how data flows in this network, needs to be defined for each network
    def forward(self, x):
        # pass data through the net
        out1 = self.fc1(x)
        out2 = self.relu1(out1)
        out3 = self.fc2(out2) 
        out4 = self.relu2(out3)
        out = self.fc3(out4)
        return out
        
# print this network architecture:
MyNet = Net(3, 100, 100, 10)
print(MyNet)
# MyNet.forward(x)
# print('shape of out:', out.shape)
# print('out_reg and cla',  out_cla.shape)


#%%
# Construct our loss function and an Optimizer. The call to model.parameters()
# Define the process of train_model
def train(epochs, train_loader, validation_loader, MyNet, optimizer, criterion):
    for epoch in range(epochs):
        train_loss = 0.0
        validation_loss = 0.0
        
        MyNet.train()
        for batch_id, data in enumerate(train_loader):
            optimizer.zero_grad() 
            inputs, labels = data
            inputs = Variable(inputs).float()
            labels = Variable(labels).float()
            # print('train inputs:', inputs)
            # print('train labels:', labels)
            out = MyNet(inputs)
            # print('train out:', out)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader) 

        MyNet.eval()
        for data in validation_loader:
            inputs, labels = data
            inputs = Variable(inputs).float()
            labels = Variable(labels).float()
            out = MyNet(inputs)
            #print('validation out:', out)
            loss = criterion(out, labels)
            
            
            validation_loss += loss.item()
        validation_loss /= len(validation_loader) 

        print("Epoch: {}/{}..".format(epoch+1, epochs),
                        "Training loss: {:.3f}..".format(train_loss/len(train_loader)),
                        "Validation loss: {:.3f}..".format(validation_loss/len(validation_loader)))
        

    return MyNet
#%%

#%%
def objective(params):
    # print(lr)
    # sys.exit()
    # lr, hidden1_size, hidden2_size =  params
    lr = params["lr"]
    hidden1_size = int(params["hidden1_size"])
    hidden2_size = int(params["hidden2_size"])

    input_size = 3
    output_size = 10
    MyNet = Net(input_size, hidden1_size, hidden2_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(MyNet.parameters(), lr=lr)
    epochs = 20

    train(epochs, train_loader, validation_loader, MyNet, optimizer, criterion)
    
    for data in validation_loader:
        inputs, labels = data
        inputs = Variable(inputs).float()
        labels = Variable(labels).float()       
        
        # print(inputs)
        # print(labels)
        result = MyNet(inputs)
        # print(result)
        pred=result.data.numpy()
        # print('shape of pred:', pred.shape)
        # print(len(pred),len(labels))
        r2 = r2_score(pred,labels)
        print(r2)
    return 1-r2



#%%



#%%
# def f (lr,hidden1_size,hidden2_size):
#     print(lr,hidden1_size,hidden2_size)
# print(type(sample["hidden2_size"]))
# params = f(**sample)
# print(params)

#%%
def test():
    MyNet.eval()
    loss = 0
    for inputs, labels in test_loader:
        inputs = Variable(inputs).float() # or inputs = Variable(torch.FloatTensor(inputs)) 
        labels = Variable(labels).float()
        print('labels:', labels)
        out = MyNet(inputs)
        print('out:', out)

        loss = criterion(out[:, 0:10], labels[:, 0:10])
        loss += loss.item()
        print('loss:', loss)

        pred=out.data.numpy()
        r2 = r2_score(pred,labels)
        

    Average_loss = loss/len(test_loader.dataset)
    print("Average loss:", Average_loss)
    print('Square rooted loss:', torch.sqrt(loss))
    print('R squared:', r2)






# %%
def main():

    criterion = nn.MSELoss()
    optimizer = optim.Adam(MyNet.parameters(), lr=0.01)
    epochs = 50

    train(epochs, train_loader, validation_loader, MyNet, optimizer, criterion)
    
    space={
        'lr': hp.uniform('lr', 0, 1),
        'hidden1_size': hp.quniform('hidden1_size', 50,100,1),
        'hidden2_size': hp.quniform('hidden2_size', 50,100,1)
        }

    # for i in range(10):
    #     sample = hyperopt.pyll.stochastic.sample(space)
    #     print(sample)

    best = fmin(
        fn= objective,
        space = space,
        algo=tpe.suggest,
        max_evals=10)
    print(best)

    params = {'hidden1_size': best['hidden1_size'], 'hidden2_size': best['hidden2_size'], 'lr': best['lr']}
    print(params)
    objective(params)

    test()

    print('model is trained')

#%%
main()

# %%
# save the trained NN
torch.save(MyNet, 'MyNet.pkl')                     # save entire net
torch.save(MyNet.state_dict(), 'MyNet_params.pkl') # only save paramters

#%%
## Restore the saved NN
def restore_net():
    net2 = torch.load('MyNet.pkl')
    return net2

restore_net()
#%%
