"""
核支持向量机(通常称SVM)可以推广到更复杂模型的拓展，这些模型无法被输入空间的超平面定
义。虽然SVM可以同时用于分类和回归，但我们只介绍分类的情况，它在SVC中实现。类似的概念
也适用于支持向量回归，后者在SVR中实现。

1.核技巧
    它的原理是直接计算扩展特征表示中数据点之间的距离(更准确说是内积)，而不用对实际扩
    展进行计算。

    对于支持向量机，将数据映射到更高维空间中有两种常用的方法：一种是多核式，在一定阶
    内计算原始特征所有可能的多项式(比如feature**2*feature**5)；另一种是径向函数核，
    也叫高斯核。一种对高斯核的解释是它考虑所有的阶数的所有可能的多项式，但阶数越高，
    特征的重要性越小。

2.理解SVM
    在训练过程中，SVM嘘唏每个训练数据点对于表示两个类别之间的决策边界的重要性。通常
    只有一部分训练数据点对于定义决策边界来说很重要：位于类别之间边界上的点。这些点叫
    作支持向量。

3.SVM调参
    gamma:小的gamma值表示决策边界变化很慢(我的理解是边界线更趋于平直)，生成的是复杂
    度较低的模型，而大的gamma值则会生成更为复杂的模型。

    c:c是正则化参数，，与线性模型中类似，它限制每个点的重要性(确切说每个点的dual_coef_)

4.为SVM预处理数据
    解决这个问题的一种方法是对每个特征进行缩放，使其大致都位于同一范围。核SVM常用的
    缩放方法就是将所有特征缩放到0到1之间。第三章学习MinMaxScale预处理方法来实现这一点
    
5.优缺点和参数
    *优点：允许决策边界很复杂，即使数据只有几个特征。它在低维数据和高维数据上的表现都很
    好。
    *缺点：对样本个数的缩放表现不好。数据量太大对运行时间和内存使用方面可能是一个挑战
    而且对于预处理数据和调参都得小心谨慎。此外，SVM模型很难检查。
    *参数：正则化参数c
           RBF参数gamma：高斯核宽度的倒数
            各类核相关参数。。。



"""