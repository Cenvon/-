from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
# from mglearn.datasets import make_blobs
from sklearn.datasets import make_blobs
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train,y_train)
# print("Test set predictions:{}".format(clf.predict(X_test)))
# print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))
fig,axes = plt.subplots(1,3,figsize = (10,3))
for n_neighbors,ax in zip([1,3,9],axes):
    #fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill = True,ax=ax,eps=0.5,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
