#!/usr/bin/env python


from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr

import warnings
from joblib import load

from sklearn.decomposition import PCA
from sklearn import preprocessing

from embedder import xclass

#=======================================================================

#print(xclass.iloc[0:49, 0])

xclass.drop(labels='Environment code', axis=1, inplace=True)
#print(xclass.iloc[:, -135:].head())
#print(xclass.iloc[:, -135:-128].head())

'''
this selects only the features from the xclass df and standardizes them
standard scalar was used bc it worked better than other stadardization method - trial and error
following the standardization, the features were fit and dim reduction was applied - fit_transform
'''
#features = xclass.iloc[:, -135:-128]
features = xclass.iloc[:, -135:]
X_std = pd.DataFrame(preprocessing.StandardScaler().fit_transform(features))
#print(X_std)
# df.to_csv('scaled_features.csv')
# df.to_csv('/home/hnscolati/orion-kl-ml/data/scaled_features.csv')

model = PCA().fit(X_std)

reduced_pcs = PCA()
reduced_features = reduced_pcs.fit_transform(X_std)
# print(reduced_features)
# print(reduced_features.shape)
# print(type(reduced_features))


# feature_space = model.inverse_transform(reduced_features)
# print(feature_space)

x = model.singular_values_
#print(x)

#print(list(xclass.columns.values))
# print(np.arange(features))
# components = np.arange(model.n_components)
# print(components)

#components = np.asarray(model.explained_variance_ratio_)
#print(components)


cov_mat = X_std.cov()   #pd.DataFrame(X_std).cov()
corr_mat = X_std.corr()
#print(cov_mat)
# print(corr_mat)
# print(cov_mat.shape)


'''
makes a df series of the just the first eigenvalue of each eigenvector and orders their 
absolute values in decescening order from greatest to least
'''
pc1 = pd.Series.abs(cov_mat.iloc[0,:])
#print(pc1)
ordered = pc1.sort_values(axis=0, ascending=False)
#print(ordered)


'''
implementation of kelvin's spearman r script to the okl data set
'''

lit_features = X_std.iloc[:, 0:6]
#print(lit_features)
print(lit_features.shape)
embeddings = X_std.iloc[:, -129:]
embeddings_init = X_std.iloc[:, -129:-119] #takes first 10 embedding features
print(embeddings_init)
print(embeddings.shape)

rho, pval = spearmanr(lit_features, embeddings_init)

plt.imshow(rho, cmap='BrBG', annot=True)
plt.colorbar()
plt.title('Correlation between Literature and Embedding Features')
plt.xlabel('First 10 Embedding Features')
plt.ylabel('Literautre Features')
plt.show()



#===================================================================
'''
#plot cumulative var ratio vs total number of features
fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)
ax1.plot(np.cumsum(model.explained_variance_ratio_))
ax2.plot(np.cumsum(model.explained_variance_ratio_))
ax2.set_xlim(-1,15)
# individual plot titles
ax1.set_title('All Principal components', style='italic')
ax2.set_title('Most important components', style='italic')
# common axis labels
fig.supxlabel('Number of components')
fig.supylabel('Cumulative variance ratio')
plt.show()

# # heat maps
# feat_corr = pd.DataFrame(X_std).corr()
# #parameter_corr = xclass.iloc[:, -135:-128].corr()

# plt.figure()
# sns.heatmap(feat_corr, cmap='cubehelix', cbar=True, square=True, annot=True, xticklabels=True, yticklabels=True)
# #sns.heatmap(parameter_corr, cmap='cubehelix', cbar=True, square=True, annot=True, xticklabels=True, yticklabels=True)
# plt.title('Feature Correlation')
# plt.show()

#scree plot of best principal components
plt.bar(range(135), model.explained_variance_ratio_, align='center')
plt.xlabel('principal components')
plt.ylabel('Explained variance ratio')
plt.xlim(-1,15)
plt.xticks(np.arange(-1, 15, 1))
plt.show()

# plt.bar(range(135), np.cumsum(model.explained_variance_ratio_), align='center')
# plt.xlabel('principal components')
# plt.ylabel('Cumulative variance')
# plt.xlim(-1,15)
# plt.xticks(np.arange(-1, 15, 1))
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)

plt.bar(range(135), model.explained_variance_ratio_, align='center', color='rosybrown', label='individual explained variance')
plt.step(range(135), np.cumsum(model.explained_variance_ratio_), where='mid', color='burlywood', label='cumulative explained variance')
plt.xlabel('principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='center right')
plt.xlim(-1,7)
plt.xticks(np.arange(-1, 7, 1))
plt.show()
'''
