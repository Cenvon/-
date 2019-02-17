import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

s = mglearn.datasets.make_signals()
a = np.random.RandomState(0).uniform(size=(100,3))
x= np.dot(s,a.T)

nmf = NMF(n_components=3,random_state=42)
s_ = nmf.fit_transform(x)

pca = PCA(n_components=3)
h= pca.fit_transform(x)

models = [x,s,s_,h]

names= ['Observations',
        'True sources',
        'NMF',
        'PCA']

fig, axes = plt.subplots(4,figsize=(8,4),gridspec_kw={'hspace':.5},subplot_kw={'xticks':(),'yticks':()})
for model,name,ax in zip(models,names,axes):
    ax.set_title(name)
    ax.plot(model[:,:3],'-')

plt.show()
