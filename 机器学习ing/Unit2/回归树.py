#回归决策树
###错误：无法打开""parsers.pyx":找不到文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

ram_prices = pd.read_csv("data/ram_price.csv")

# plt.semilogy(ram_prices.data,ram_prices.price)
# plt.xlabel("Year")
# plt.ylabel("Price in $/Mbyte")
# plt.show()

data_train = ram_prices[ram_prices._data<2000]
data_test = ram_prices[ram_prices._data>2000]

X_train = data_train.data[:,np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train,y_train)
linear_reg = LinearRegression().fit(X_train,y_train)
X_all = ram_prices._data[:,np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.data,data_train.price,label="Train Data")
plt.semilogy(data_test.data,data_test.price,label="Test Data")
plt.semilogy(ram_prices.data,price_tree,label="Tree Prediction")
plt.semilogy(ram_prices.data,price_lr,label="Linear Prediction")
plt.legend()
plt.show()