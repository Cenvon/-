import matplotlib.pyplot as plt
import numpy as np
import mglearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC#svc代表支持向量分类器
#svm是一种分类算法，不是一种回归算法
#决策边界可视化
x,y = mglearn.datasets.make_forge()
fig,axes = plt.subplots(1,2,figsize = (10,3))
for model,ax in zip([LinearSVC(),LogisticRegression()],axes):
    clf = model.fit(x,y)
    mglearn.plots.plot_2d_separator(clf,x,fill=False,eps=0.5,ax=ax,alpha=.7)
    mglearn.discrete_scatter(x[:,0],x[:,1],y,ax=ax)
axes[0].legend()
plt.show()

#mglearn.plots.plot_linear_svc_regularization()
#mglearn函数库的显示函数都可以加一个plt.show()用来直接显示图片