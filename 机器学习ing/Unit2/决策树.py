"""
决策树是广泛应用于分类和回归任务的模型。本质上，它从一层层的if/else问题中进行学习，
并得出结论。
1.构造树
学习决策树就是学习一系列if/else问题，使我们能够最快速度得到正确答案。在机器学习中
这些问题叫做"测试"

2.控制树的复杂度
通常来说，构造决策树直到所有叶节点都是纯的叶节点，这会导致模型非常复杂，并且对训练
数据高度过拟合。
防止过拟合有两种常见的策略：
    1>预剪枝：及早停止树的增长
    2>后剪枝：先构造树，但随后删除或折叠信息量很少的节点
sklearn的决策树在DecisionTreeRegressor类和DecisionTreeClassifier类中实现
sklearn只实现了预剪枝，没有实现后剪枝

3.分析决策树
我们可以利用tree模块的export_graphviz函数来将树可视化。函数会生成一个以.dot后缀的
文件用来保存图形的文本文件格式

4.树的特征重要性
利用一些有用的属性总结树的工作原理，最常用的就是特征重要性，它为每个特征对树的决策
的重要性进行排序。对于每个特征来说，它都是一个介于0和1之间的数字，其中0表示"根本没有"，
1表示"完美预测目标值"特征重要性求和始终为1。

5.回归决策树
虽然前面讨论的是分类的决策树，不过对于回归的决策树来说，所有内容都是类似的，在
DecisionTreeRegression中实现。回归树的用法和分析与分类树非常相似。但是将基于树
的模型用于回归时，我们想要指出它的一个特殊性质。DecisionTreeRegression（以及其
它基于树的回归模型）不能外推，也不能在训练数据范围之外进行预测。

6.优点，缺点和参数
决策树两个优点：1，得到的模型很容易可视化，非专家也很容易理解（至少对于较小的树而言
）；2，完全不受数据缩放的影响
缺点：就算做了预剪枝，它也经常会过拟合，泛化性很差

"""
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(
    cancer.data,cancer.target,stratify = cancer.target,random_state = 42
)
#默认将树完全展开树的训练精度是1，但是测试精度却只有0.937
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X_train,y_train)
# print(tree.score(X_train,y_train))
# print(tree.score(X_test,y_test))

#最大深度设置为4，训练精度是0.988，测试精度有0.951
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train,y_train)
# print(tree.score(X_train,y_train))
# print(tree.score(X_test,y_test))

#决策树可视化
export_graphviz(tree,out_file="tree.dot",class_names=["malignant","benign"],feature_names=cancer.feature_names,impurity=False,filled=True)
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)





