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
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
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
scaler = StandardScaler()
scaler.fit(data[feature_ALL])
# print(scaler.mean_)
data[feature_ALL] = scaler.transform(data[feature_ALL])
# augment invertibility
data['ExtraNode'] = data[feature_ALL].mean(1)

k = 5
kf = KFold(n_splits=k,shuffle=True)
accuracy_KFold = []
fold = 1
tried = ''
plt.figure()
for train_index, test_index in kf.split(data):
    def randomcolor():
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            color += colorArr[np.random.randint(0,14)]
        return "#"+color
    color = randomcolor()
    print('processing {}th fold'.format(fold));fold+=1
    train_data = data.copy().loc[train_index]
    test_data = data.copy().loc[test_index]
    print("train data size:{}, test data size:{}".format(len(train_data),len(test_data)))

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

    #definition of hyperparameters
    num_classes = 7+1
    learning_rate = 0.02
    hidden = 70
    num_epochs = 50
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
            input_bidirection = Variable(init_input_bidirection.clone().detach().requires_grad_(True))
            output_bidirection = net_bidirection(input_bidirection)
            loss_bidirection = criterion2(output_bidirection,b_x)
            optimizer2.zero_grad()
            loss_bidirection.backward()
            optimizer2.step()

        all_losses.append(loss.item())

        #try training accuracy
        # tried = 'training'
        # test_out = net(input)[:,0:7]
        # pred = torch.max(F.softmax(test_out,dim=1),1)[1].numpy()
        # accuracy = sum(pred == y.numpy())/len(pred)

        #try testing accuracy
        tried = 'testing'
        test_out = net(test_input)[:,0:7]
        pred = torch.max(F.softmax(test_out,dim=1),1)[1].numpy()
        accuracy = sum(pred == test_y.numpy())/len(pred)

        all_accuracy.append(accuracy)

    output = net.forward(test_input)[:,0:7]
    accuracy = 0
    pred = torch.max(F.softmax(output,dim=1),1)[1].numpy()
    accuracy = sum(pred == test_y.numpy())/len(pred)
    print("accuracy of neural net: "+str(accuracy*100.)+"%")
    accuracy_KFold.append(accuracy)
    # fig = plt.figure(figsize=(8,3))
    x = np.arange(1,num_epochs+1).reshape([-1,1])
    polynomial = PolynomialFeatures(degree = 50)
    x_transformed = polynomial.fit_transform(x)
    poly_linear_model = Lasso(max_iter=10000)
    poly_linear_model.fit(x_transformed, all_accuracy)
    xx_transformed = polynomial.fit_transform(x)
    y = poly_linear_model.predict(xx_transformed)
    plt.plot(x,y,color = color,label=str(fold-1)+'th fold')


print("{}-FOLD cross validation average accuracy: {}%".format(k,np.mean(accuracy_KFold)*100.))
plt.xlabel('epoch(s)')
plt.ylabel(tried+' accuracy')
plt.legend()
plt.show()
