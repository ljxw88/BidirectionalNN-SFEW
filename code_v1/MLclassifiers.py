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

data = pd.read_excel('/Users/michael/Desktop/COMP4660/Assignment1/faces-emotion (images)/SFEW.xlsx')
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
test_target = test_data['Label']


clf = RandomForestClassifier(n_estimators=100,criterion="gini")
clf.fit(train_data_LPQ, train_target)
prediction_sklearn = clf.predict(test_data_LPQ)
accuracy_sk = 0
for i in range(len(prediction_sklearn)):
    if prediction_sklearn[i]==list(test_target)[i]:
        accuracy_sk += 1
accuracy_sk /= len(prediction_sklearn)
print("accuracy of random forest classifier: "+str(accuracy_sk*100.)+"%")

clf = SVC()
clf.fit(train_data_LPQ, train_target)
prediction_sklearn = clf.predict(test_data_LPQ)
accuracy_sk = 0
for i in range(len(prediction_sklearn)):
    if prediction_sklearn[i]==list(test_target)[i]:
        accuracy_sk += 1
accuracy_sk /= len(prediction_sklearn)
print("accuracy of SVM classifier: "+str(accuracy_sk*100.)+"%")
