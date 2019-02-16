from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state = 0)
# gbrt = GradientBoostingClassifier(random_state=0)
# gbrt.fit(X_train,y_train)

# print(gbrt.score(X_train,y_train))#1.0
# print(gbrt.score(X_test,y_test))#0.958041958041958

#上面训练精度为100%，可能存在过拟合，为了降低过拟合，故我们以限制最大深度来加强预剪枝。
gbrt = GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)

print(gbrt.score(X_train,y_train))#0.9906103286384976
print(gbrt.score(X_test,y_test))#0.972027972027972      对比前面一个明显提高了

#降低学习率来加强预剪枝
# gbrt = GradientBoostingClassifier(random_state=0,learning_rate=0.01)
# gbrt.fit(X_train,y_train)

# print(gbrt.score(X_train,y_train))#0.9882629107981221
# print(gbrt.score(X_test,y_test))#0.965034965034965      依然较第一个有所提升


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importances")
    plt.ylabel("Feature")
    plt.show()

plot_feature_importances_cancer(gbrt)
