import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
"""
LogisticRegression默认应用L2正则化
本次主要对正则化参数C的调整
每次的训练通过下列两个打印出训练的得分
    print(logreg.score(x_train,y_train))
    print(logreg.score(x_test,y_test))

"""
cancer = load_breast_cancer()
cancer=load_breast_cancer(return_X_y=False)
x_train,x_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify = cancer.target,random_state = 66
)

#调整C的参数
logreg = LogisticRegression().fit(x_train,y_train)#C=1
logreg100 = LogisticRegression(C=100).fit(x_train,y_train)#C=100
logreg001 = LogisticRegression(C=0.01).fit(x_train,y_train)#C=0.01

plt.plot(logreg.coef_.T,'o',label = "c=1")
plt.plot(logreg100.coef_.T,'^',label = "c=100")
plt.plot(logreg001.coef_.T,'v',label = "c=0.01")
plt.hlines(0,0,cancer.data.shape[1])
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5,5)
plt.legend()
plt.show()
