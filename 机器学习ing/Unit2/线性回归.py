import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
#p36
#mglearn.plots.plot_linear_regression_wave()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x,y = mglearn.datasets.make_wave(n_samples=60)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42)
lr = LinearRegression().fit(x_train,y_train)

# print(lr.coef_)##斜率
# print(lr.intercept_)##截距
"""
###保存文件
import pickle
f = open("test.pickle","wb")
pickle.dump(lr,f)
f.close()
###读取文件
with open("test.pickle","rb") as f:
    lr1 = pickle.load(f)
print(lr1.coef_)
print(lr1.intercept_)  
"""
print(lr.score(x_test,y_test))
print(lr.score(x_train,y_train))
