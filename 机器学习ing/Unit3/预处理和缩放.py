#预处理和缩放
import mglearn
# mglearn.plots.plot_scaling()
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,
                            cancer.target,random_state = 1)
scl=MinMaxScaler()
scl.fit(X_train)
X_train_scale = scl.transform(X_train)
# print(X_train_scale.max(axis=0))
# print(X_train_scale.min(axis=0))
X_test_scale = scl.transform(X_test)
print(X_test_scale.min(axis=0))
print(X_test_scale.max(axis=0))
"""
输出：
[ 0.0336031   0.0226581   0.03144219  0.01141039  0.14128374  0.04406704
  0.          0.          0.1540404  -0.00615249 -0.00137796  0.00594501
  0.00430665  0.00079567  0.03919502  0.0112206   0.          0.
 -0.03191387  0.00664013  0.02660975  0.05810235  0.02031974  0.00943767
  0.1094235   0.02637792  0.          0.         -0.00023764 -0.00182032]

[0.9578778  0.81501522 0.95577362 0.89353128 0.81132075 1.21958701
 0.87956888 0.9333996  0.93232323 1.0371347  0.42669616 0.49765736
 0.44117231 0.28371044 0.48703131 0.73863671 0.76717172 0.62928585
 1.33685792 0.39057253 0.89612238 0.79317697 0.84859804 0.74488793
 0.9154725  1.13188961 1.07008547 0.92371134 1.20532319 1.63068851]
 
 你可能发现对训练集缩放后的最大值和最小值不是1和0，甚至有些数据超出了0~1之外，
 由此可见，transform的方法总是减去训练集的最小值。然后除以训练集的范围，而这两
 个值可能与测试集的最小值和范围并不相同。
"""