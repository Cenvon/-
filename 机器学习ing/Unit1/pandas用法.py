import pandas as pd
from IPython.display import display
#创建关于人的简单数据集
data = {'name':["Jhon","Anna","Peter","Linda"],
        'location':["NewYork","Paris","Berlin","London"],
        'age':[24,13,53,33]}
data_pandas = pd.DataFrame(data)
display(data_pandas)
print("\n\n")
#选择年龄大于30的所有行
display(data_pandas[data_pandas.age>30])