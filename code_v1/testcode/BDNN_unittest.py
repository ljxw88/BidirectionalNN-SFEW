import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset,DataLoader,TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
# np.random.seed(5)

#load in data(SFEW.xlsx), please change this path if its incorrect
data = pd.read_excel('/Users/michael/Desktop/COMP4660/Assignment1/faces-emotion (images)/SFEW.xlsx')

#data preprocessing, scale data and normalize.
data = data.fillna(0)
column_names = ['Name','Label','LPQ feature 1','LPQ feature 2','LPQ feature 3','LPQ feature 4','LPQ feature 5','PHOG feature 1','PHOG feature 2','PHOG feature 3','PHOG feature 4','PHOG feature 5']
data.columns = column_names
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

feature_LPQ = ['LPQ feature 1','LPQ feature 2','LPQ feature 3','LPQ feature 4','LPQ feature 5']
feature_PHOG = ['PHOG feature 1','PHOG feature 2','PHOG feature 3','PHOG feature 4','PHOG feature 5']
feature_ALL = feature_LPQ + feature_PHOG
# display(data)
msk = np.random.rand(len(data)) < 0.8
scaler = StandardScaler()
scaler.fit(data[feature_ALL])
# print(scaler.mean_)
data[feature_ALL] = scaler.transform(data[feature_ALL])
# augment invertibility
data['ExtraNode'] = data[feature_ALL].mean(1)

train_data = data[msk].copy()
test_data = data[~msk].copy()
# print(len(train_data),len(test_data))

train_data_LPQ = train_data[feature_LPQ]
test_data_LPQ = test_data[feature_LPQ]

train_data_PHOG = train_data[feature_PHOG]
test_data_PHOG = test_data[feature_PHOG]

train_data_ALL = train_data[feature_LPQ+feature_PHOG]
test_data_ALL = test_data[feature_LPQ+feature_PHOG]

train_target = train_data['Label']
train_extra = train_data['ExtraNode']
test_target = test_data['Label']

x_LPQ = torch.Tensor(train_data_LPQ.values).float()
x_PHOG = torch.Tensor(train_data_PHOG.values).float()
x_ALL = torch.Tensor(train_data_ALL.values).float()
y = torch.Tensor(train_target.values).long()
y_extra = torch.Tensor(train_extra.values).long()

test_x_LPQ = torch.Tensor(test_data_LPQ.values).float()
test_x_PHOG = torch.Tensor(test_data_PHOG.values).float()
test_x_ALL = torch.Tensor(test_data_ALL.values).float()
test_y = torch.Tensor(test_target.values).long()

all_accuracy_hidden = []
all_loss_hidden = []
start = 20
end = 100
for n_unit in range(start,end+1):
#definition of hyperparameters
    num_classes = 7+1
    learning_rate = 0.02
    hidden = n_unit
    num_epochs = 100
    batch_size = 64
    drop_out = 0.1
    data_select = 3

    input = x_ALL
    test_input = test_x_ALL
    if (data_select==1):
        input = x_LPQ
        test_input = test_x_LPQ
    elif (data_select==2):
        input = x_PHOG
        test_input = test_x_PHOG
    elif (data_select==3):
        input = x_ALL
        test_input = test_x_ALL
    else:
        None
    dataset_combine = TensorDataset(input,y)
    train_loader = DataLoader(dataset = dataset_combine, batch_size=batch_size, shuffle=True, num_workers=2)

    def assignTensor(x, y):
        x = torch.nn.Parameter(torch.tensor(y.detach().numpy()))

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net,self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            self.predict = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net = Net(input.shape[1], hidden, num_classes)
    net_bidirection = Net(num_classes, hidden, input.shape[1])
    # print(net)
    # print(net_bidirection)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(net_bidirection.parameters(), lr=learning_rate)
    criterion2 = torch.nn.BCEWithLogitsLoss()

    all_losses = []
    all_accuracy = []
    for epoch in range(num_epochs):
        for step,(x_,y_) in enumerate(train_loader):
            #Reversely assign weight of backward model,Train the forward model
            assignTensor(net.predict.weight, net_bidirection.hidden.weight.T)
            assignTensor(net.hidden.weight, net_bidirection.predict.weight.T)
            b_x = Variable(x_)
            b_y = Variable(y_)
            b_y_extra = Variable(y_extra)
            output = net(b_x)
            loss = criterion(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Reversely assign weight of forward model, Train the backward model
            assignTensor(net_bidirection.hidden.weight, net.predict.weight.T)
            assignTensor(net_bidirection.predict.weight, net.hidden.weight.T)
            init_input_bidirection = torch.zeros((len(b_y),num_classes))
            for i in range(len(b_y)):
                init_input_bidirection[i,int(b_y[i])] = 1
                init_input_bidirection[i,num_classes-1] = b_y_extra[i]
            input_bidirection = Variable(torch.tensor(init_input_bidirection))
            output_bidirection = net_bidirection(input_bidirection)
            loss_bidirection = criterion2(output_bidirection,b_x)
            optimizer2.zero_grad()
            loss_bidirection.backward()
            optimizer2.step()

        all_losses.append(loss.item())
        test_out = net(input)
        pred = torch.max(F.softmax(test_out,dim=1),1)[1].numpy()
        accuracy = sum(pred == y.numpy())/len(pred)
        all_accuracy.append(accuracy)
        # if (epoch + 1) % 10 == 0:
        #     print('Training Epoch: [%d/%d], Loss: %.4f, Accuracy: %.2f%%'
        #           % (epoch + 1, num_epochs, loss.item(), accuracy*100.))

    output = net.forward(test_input)
    accuracy = 0
    pred = torch.max(F.softmax(output,dim=1),1)[1].numpy()
    accuracy = sum(pred == test_y.numpy())/len(pred)
    # print("accuracy of neural net: "+str(accuracy*100.)+"%")
    all_accuracy_hidden.append(all_accuracy[-1])
    all_loss_hidden.append(all_losses[-1])
    print(str(n_unit)+"th test completed, final loss:"+str(all_accuracy[-1])+"")

#plot accuracy&loss,  confusion matrix.
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_xlabel('epoch')
ax1.set_ylabel('training accuracy')
ax1.plot(list(np.arange(start,end+1)),all_accuracy_hidden)

ax2.set_xlabel('epoch')
ax2.set_ylabel('training loss')
ax2.plot(list(np.arange(start,end+1)),all_loss_hidden)
plt.show()
