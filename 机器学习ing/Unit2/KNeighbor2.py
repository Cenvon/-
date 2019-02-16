# from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# cancer=load_breast_cancer()
# X_train,X_test,y_train,y_test = train_test_split(
#     cancer.data,cancer.target,stratify = cancer.target,random_state = 66
# )
# training_accuracy = []
# test_accuracy = []
# neighbors_settings = range(1,11)
# for n_neighbors in neighbors_settings:
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train,y_train)
#     training_accuracy.append(clf.score(X_train,y_train))
#     test_accuracy.append(clf.score(X_test,y_test))
# plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
# plt.plot(neighbors_settings,test_accuracy,label="test accuracy")
# plt.plot.xlabel("Accuracy")
# plt.plot.ylable("n_neighbor")
# plt.legend()

from sklearn.datasets import load_breast_cancer 

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(
cancer.data,cancer.target,stratify=cancer.target,random_state=66)

trainning_accuracy = []
test_accuracy = []
#n_neighbors取值从1到10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    #构建模型
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    #记录训练集精度
    trainning_accuracy.append(clf.score(X_train,y_train))
    #记录泛化精度
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,trainning_accuracy,label="trainning accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

