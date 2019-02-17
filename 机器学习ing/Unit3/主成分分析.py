import mglearn
# mglearn.plots.plot_pca_illustration()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
fig,axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.date[cancer.target==0]
benign = cancer.date[cancer.target==1]

ax = axes.ravel()

for i in range(30):
    _,bins = np.histogram(cancer.data[:,i],bins = 50)
    ax[i].hist(malignant[:,i],bins=bins,color = mglearn.cm3(0),alpha = .5)
    ax[i].hist(benign[:,i],bins=bins,color = mglearn.cm2(2),alpha = .5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequence")
ax[0].legend(["malignant","benign"],loc="best")
fig.tight_layout()
plt.show()