import numpy as np
from scipy import sparse

#创建一个二维Numpy数组，对角线是1，其余为0
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

#将NumPy数组转换成CSR格式的SciPy稀疏矩阵
#只保留非零元素
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))