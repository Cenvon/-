from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state = 0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
import numpy as np
X_new = np.array([[5,2.9,1,0.2]])
# print("X_new.shape:{}".format(X_new.shape))
pre = knn.predict(X_new)
print("Prediction :{}".format(pre))
print("Predicted target name:{}\n".format(iris_dataset['target_names'][pre]))
y_pred = knn.predict(X_test)
print("Test set predictions :\n{}".format(y_pred))
print("Test set score:{:.2f}".format(np.mean(y_pred == y_test)))