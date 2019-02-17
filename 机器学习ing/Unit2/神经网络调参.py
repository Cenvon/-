import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state =42)
# mlp = MLPClassifier(solver='lbfgs',random_state=0).fit(X_train,y_train)

#修改hidden_layer_sizes =[10]
# mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10]).fit(X_train,y_train)

#使用两个隐层hidden_layer_sizes = [10,10] 每层10个单元隐单元
# mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)

#接着上一步，这次使用tanh非线性
mlp = MLPClassifier(solver='lbfgs',activation='tanh',random_state=0,hidden_layer_sizes=[10,10]).fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp,X_train,fill= True,alpha=.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train) 
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
