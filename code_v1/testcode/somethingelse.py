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

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

hist = [21.3,19.7,24.,26.8,24.]
label = ['Random Forest','SVM','BPNN','BDNN','BiLSTM']
X = np.arange(1,len(hist)+1)

plt.bar(label, height=hist, label=label ,color='gray', )
plt.ylabel('Accuracy')


for a,b in zip(X,hist):
    plt.text(a-1, b+0.05, '%.1f' % b, ha='center', va='bottom',fontsize=11)

plt.show()
