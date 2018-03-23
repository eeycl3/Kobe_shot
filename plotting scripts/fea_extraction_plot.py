import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


X_train = pd.read_csv('X_train_rfe_40.csv', sep=',')
X_test = pd.read_csv('X_test_rfe_20.csv', sep=',')
y_train = pd.read_csv('y_train.csv', sep=',')
y_test = pd.read_csv('y_test.csv', sep=',')
y_train= y_train["shot_made_flag"]
classLable = ["shot missed", "shot made"]
colors=["r", "b"]
"""
pca = PCA(n_components=2, svd_solver="full")
pca_X_train= pca.fit_transform(X_train)
plt.subplot("111")
for l, color in zip(np.unique(y_train), colors):
    plt.scatter(pca_X_train[y_train==l, 0], pca_X_train[y_train==l, 1], c = color)

lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2, eigen_solver='dense')
lle_X_train = lle.fit_transform(X_train)
plt.subplot("112")
for l, color in zip(np.unique(y_train), colors):
    plt.scatter(lle_X_train[y_train==l, 0], lle_X_train[y_train==l, 1], c = color)

isomap = Isomap(n_neighbors=5, n_components=2)
isomap_X_train = isomap.fit_transform(X_train)
plt.subplot("113")
for l, color in zip(np.unique(y_train), colors):
    plt.scatter(isomap_X_train[y_train==l, 0], isomap_X_train[y_train==l, 1], c = color)
"""
tsne = TSNE(n_components=2)
tsne_X_train = tsne.fit_transform(X_train)
print(len(tsne_X_train))
for l, color,name in zip(np.unique(y_train), colors, classLable):
    plt.scatter(tsne_X_train[y_train==l, 0], tsne_X_train[y_train==l, 1], c = color, label = name )
plt.legend()
plt.title("TSNE reduction for visualization on training data")
plt.xlabel("TSNE component 1")
plt.ylabel("TSNE component 2")
plt.show()