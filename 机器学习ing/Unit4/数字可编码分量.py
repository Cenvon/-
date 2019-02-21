import pandas as pd
from IPython.display import display
demo = pd.DataFrame({
    'Int':[0,1,2,1],
    'Categorical':['socks','fox','socks','box']
})

pd.get_dummies(demo)
demo['Int'] = demo['Int'].astype(str)
pd.get_dummies(demo,columns=['Int','Categorical'])


