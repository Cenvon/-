import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

x,y = mglearn.datasets.load_extended_boston()
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)

lr = LinearRegression().fit(x_train,y_train)
ridge = Ridge().fit(x_train,y_train)
# print(ridge.score(x_train,y_train))
# print(ridge.score(x_test,y_test))
ridge10 = Ridge(alpha=10).fit(x_train,y_train)
ridge01 = Ridge(alpha=0.1).fit(x_train,y_train)

plt.plot(ridge.coef_,'s',label = "alpha = 1")
plt.plot(ridge.coef_,'^',label = "alpha = 10")
plt.plot(ridge.coef_,'v',label = "alpha = 0.1")
plt.plot(lr.coef_,"o",label="LinearRegression")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()
