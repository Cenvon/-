import mglearn 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

x, y = mglearn.datasets.make_wave(n_samples=40)
# 将wave数据集分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 创建1000个数据点，在-3到3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9],axes):
    # 利用1个，3个和9个邻居分别进行预测
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(x_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(x_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(x_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
            "{} neighbor(s)\n train score:{:.2f} test score:{:.2f}".format(
                    n_neighbors, reg.score(x_train, y_train),
                    reg.score(x_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model prediction", "Training data/target", "Test data/target"], loc="best")
plt.show()