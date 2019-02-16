import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

x,y = mglearn.datasets.load_extended_boston()
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

ls = Lasso().fit(x_train,y_train)
# print(ls.score(x_test,y_test))
ls001 = Lasso(alpha=0.01,max_iter=100000).fit(x_train,y_train)
ls00001 = Lasso(alpha=0.0001,max_iter=100000).fit(x_train,y_train)
ls01 = Lasso(alpha=0.1,max_iter=100000).fit(x_train,y_train)

plt.plot(ls.coef_,'s',label = "alpha = 1")
plt.plot(ls001.coef_,'^',label = "alpha = 0.01")
plt.plot(ls00001.coef_,'v',label = "alpha = 0.0001")
plt.plot(ls01.coef_,"o",label="alpha = 0.1")

# plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend(ncol = 2,loc = (0,1.05))
plt.show()