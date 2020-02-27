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

from sklearn.metrics import r2_score  # for regression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from hyperopt import fmin, tpe, hp
import pickle
import os
import sys
import hyperopt.pyll.stochastic
#%%
### see how data looks like
# df = pd.read_excel(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_multiobj+classification_train.xlsx', header = 0)
# df_array = df.to_numpy()

# for i in range(3, 8):
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
def datapreparation(file_path):

    df = pd.read_excel(file_path, header = 0)
    df_array = df.to_numpy() # first transform df to numpy array    

    # Access features and normalize them
    x = torch.from_numpy(df_array[:, 0:3]) # transform numpy array to tensors # prices of barley, rapeseed, wheat 
    picklename = "scaling_metrics_01.pkl"
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

    # Access lables and normalize y1-y4
    y_reg = torch.from_numpy(df_array[:, 3:7]) # get data of y1-y4
    y_reg = (y_reg - torch.min(y_reg)) / (torch.max(y_reg) - torch.min(y_reg)) 
    y_cla = torch.from_numpy(df_array[:, 7]).reshape(y_reg.shape[0],1) # get data of y5
    y = torch.cat((y_reg, y_cla), 1) 
    
    print('Type and shape of x', x.type, x.shape)
    print('Type and shape of y', y.type, y.shape)
    return [x, y]

train_file = datapreparation(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_multiobj+classification_train.xlsx')
validation_file = datapreparation(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_multiobj+classification_validation.xlsx')
test_file = datapreparation(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_multiobj+classification_test.xlsx')
print(train_file[0]) # outcome of the datapreparation function X
print(train_file[1]) # outcome of the datapreparation function y
    
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

# Load train, validation and test data
train_dataset = DatasetCDFarm(train_file)
train_loader = DataLoader(dataset=train_dataset, 
                        batch_size = 4, 
                        shuffle = True) 
validation_dataset = DatasetCDFarm(validation_file)
validation_loader = DataLoader(dataset=validation_dataset, 
                        batch_size = len(validation_dataset)) 
test_dataset = DatasetCDFarm(test_file)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset)) 



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
        out5 = self.fc3(out4)
        out_reg = out5[:, 0:4] # y1-y4 regression problem, y5 is classification problem 
        out_cla1 = torch.sigmoid(out5[:, 4]).reshape(out5.shape[0],1) # transform y5 with sigmoid function 
        # out_cla2 = out_cla1 #(out_cla1>0.5).float()
        out = torch.cat((out_reg, out_cla1), 1)
        return out
        
# print this network architecture:
MyNet = Net(3, 90, 50, 5)
print(MyNet)


#%%
def train(epochs, train_loader, validation_loader, MyNet, optimizer, criterion1, criterion2):
  
    Train_Losses = [] # empty list to store train losses through epochs
    Val_Losses = [] # empty list to store validation losses through epochs
    
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

            out_reg, labels_reg = out[:, 0:4], labels[:, 0:4]
            out_cla, labels_cla = out[:, 4], labels[:, 4]

            loss_reg = criterion1(out_reg, labels_reg)
            loss_cla = criterion2(out_cla, labels_cla)
            # print('Regression loss: ', loss_reg)
            # print('Classification loss: ', loss_cla)

            loss = loss_reg * 1000 + loss_cla # loss function
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader) 
        Train_Losses.append(train_loss) # store train_loss to Train_Losses

        MyNet.eval()
        for data in validation_loader:
            inputs, labels = data
            inputs = Variable(inputs).float()
            labels = Variable(labels).float()
            out = MyNet(inputs)
            #print('validation out:', out)

            out_reg, labels_reg = out[:, 0:4], labels[:, 0:4]
            out_cla, labels_cla = out[:, 4], labels[:, 4]

            loss_reg = criterion1(out_reg, labels_reg)
            loss_cla = criterion2(out_cla, labels_cla)
            
            loss = loss_reg * 1000 + loss_cla
            validation_loss += loss.item()
        validation_loss /= len(validation_loader) 
        Val_Losses.append(validation_loss) # # store validation_loss to Val_Losses

        print("Epoch: {}/{}..".format(epoch+1, epochs),
                        "Training loss: {:.3f}..".format(train_loss/len(train_loader)),
                        "Validation loss: {:.3f}..".format(validation_loss/len(validation_loader)))
        
    plt.plot(Train_Losses, label='Training losses')
    plt.plot(Val_Losses, label='Validation losses')
    plt.legend()
    plt.show()


    return MyNet
#%%
def test(test_loader, criterion1, criterion2):
    MyNet.eval()
    loss = 0
    for inputs, labels in test_loader:
        inputs = Variable(inputs).float() # or inputs = Variable(torch.FloatTensor(inputs)) 
        labels = Variable(labels).float()
        # print('labels:', labels)
        out = MyNet(inputs)
        # print('out:', out)
        out_reg, labels_reg = out[:, 0:4], labels[:, 0:4]
        out_cla, labels_cla = out[:, 4], labels[:, 4]

        loss_reg = criterion1(out_reg, labels_reg)
        loss_cla = criterion2(out_cla, labels_cla)

    
        loss = loss_reg * 1000 + loss_cla
        loss += loss.item()
        # print('loss:', loss)

        pred_reg=out_reg.data.numpy()
        pred_cla=out_cla.data.numpy()
        r2 = r2_score(pred_reg, labels_reg)
        

        fpr, tpr, thresholds = roc_curve(labels_cla, pred_cla, pos_label=0)
        # Print ROC curve
        plt.plot(fpr,tpr)
        plt.show() 
        # Print AUC
        auc = np.trapz(tpr,fpr)
             

    Average_loss = loss/len(test_loader.dataset)
    print("Average loss:", Average_loss)
    print('Square rooted loss:', torch.sqrt(loss))
    print('R squared:', r2)
    print('AUC:', auc)  # good auc should be close to 1 
#%%

#%%
def main():

    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()
    optimizer = optim.Adam(MyNet.parameters(), lr=0.01)
    epochs = 20

    train(epochs, train_loader, validation_loader, MyNet, optimizer, criterion1, criterion2)
    
    
    # space={
    #     'lr': hp.uniform('lr', 0, 1),
    #     'hidden1_size': hp.quniform('hidden1_size', 50,100,1),
    #     'hidden2_size': hp.quniform('hidden2_size', 50,100,1)
    #     }

    # # for i in range(10):
    # #     sample = hyperopt.pyll.stochastic.sample(space)
    # #     print(sample)

    # best = fmin(
    #     fn= objective,
    #     space = space,
    #     algo=tpe.suggest,
    #     max_evals=10)
    # print(best)

    # params = {'hidden1_size': best['hidden1_size'], 'hidden2_size': best['hidden2_size'], 'lr': best['lr']}
    # print(params)
    # objective(params)

    test(test_loader, criterion1, criterion2)

    print('model is trained')

#%%
main()

#%%
# Construct our loss function and an Optimizer. The call to model.parameters()


# loss_values = []    

# for epoch in range(epochs):
#     cum_loss = 0
   
#     MyNet.train()
#     for batch_id, data in enumerate(train_loader):
#         # get the inputs
#         inputs, labels = data

#         # wrap them in Variable
#         inputs = Variable(inputs).float()
#         labels = Variable(labels).float()

#         # print(epoch, batch_id, "inputs", inputs.data, "labels", labels.data)
#         # Forward pass
                        
#         out = MyNet(inputs)
#         # print('out', type(out),out)
#         # print('labels', type(labels),labels)

#         loss1 = criterion1(out[:, 0:4], labels[:, 0:4])
#         loss2 = criterion2(out[:, 4], labels[:, 4])

#         loss = loss1 + 0.1*loss2
#         print(epoch, batch_id, loss.data)
#         cum_loss += loss.data

#         # Zero gradients, perform a backward pass, and update the weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     loss_values.append(cum_loss/len(train_dataset))

# plt.plot(loss_values)


# #%%

# test_dataset = DatasetCDFarm(r'N:\agpo\work1\Shang\ForPyomoBook\MLmodel\nn_CDFarm_torch_multiobj+classification_test.xlsx')
# test_loader = DataLoader(dataset=test_dataset) 
# #%%
# def test():
#     MyNet.eval()
#     loss = 0
#     for inputs, labels in test_loader:
#         inputs = Variable(inputs).float()
#         labels = Variable(labels).float()
#         print('labels:', labels)
#         out = MyNet(inputs)
#         print('out:', out)

#         loss1 = criterion1(out[:, 0:4], labels[:, 0:4])
#         loss2 = criterion2(out[:, 4], labels[:, 4])
#         loss = loss1 + 0.1*loss2

#         loss += loss.item()
#         print('loss:', loss)
#     Average_loss = loss/len(test_loader.dataset)
#     print("Average loss:", Average_loss)

# test()

# print('Square rooted loss:', torch.sqrt(loss))


#%%
