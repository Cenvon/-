from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import mglearn
X,y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)

bins = np.linspace(-3,3,11)
wbin = np.digitize(X,bins=bins)

encoder = OneHotEncoder(sparse=False)
encoder.fit(wbin)
xbin = encoder.transform(wbin)

linebin = encoder.transform(np.digitize(line,bins=bins))

reg = LinearRegression().fit(xbin,y)
plt.plot(line,reg.predict(linebin),label = 'linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(xbin,y)
plt.plot(line,reg.predict(linebin),label = 'decision tree binned')
plt.plot(X[:,0],y,'o',c='k')
plt.vlines(bins,-3,3,linewidth =1,alpha = .2)
plt.legend(loc = 'best')
plt.xlabel("Regression output")
plt.ylabel("Input feature")
plt.show()