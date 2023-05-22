#!/usr/bin/env python


from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from umda.data import load_pipeline
from joblib import load

from astrochem_embedding import VICGAE
import torch

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.utils import resample


#==========================================================================================================

def bootstrap_bill_turner(data: Tuple[np.ndarray], seed: int, n_samples: int = 500, replace: bool = True, noise_scale: float = 0.5, molecule_split: float = 0.2, test_size: float = 0.2):
    """
    This function specifically splits the training set into train
    and validation sets within molecule classes. The idea behind this
    is to prevent data leakage.
    Parameters
    ----------
    data : Tuple[np.ndarray]
        [description]
    seed : int
        [description]
    n_samples : int, optional
        [description], by default 500
    replace : bool, optional
        [description], by default True
    noise_scale : float, optional
        [description], by default 0.5
    molecule_split : float, optional
        [description], by default 0.2
    """
    true_X, true_y = data
    indices = np.arange(len(true_y))
    rng = np.random.default_rng(seed)
    # shuffle the molecules
    rng.shuffle(indices)
    split_num = int(len(indices) * molecule_split)
    test_indices = indices[:split_num]
    train_indices = indices[split_num:]
    # print('test', test_indices)
    # print('train', train_indices)
    test_indices.sort(); train_indices.sort()
    sets = list()
    indices = list()
    for index_set, train in zip([train_indices, test_indices], [True, False]):
        if train:
            num_samples = int(n_samples * (1 - test_size))
        else:
            num_samples = int(n_samples * test_size)
        resampled_indices = resample(index_set, n_samples=num_samples, replace=replace, random_state=seed)
        resampled_indices.sort()
        resampled_X, resampled_y = true_X[resampled_indices], true_y[resampled_indices]
        reshuffled_indices = np.arange(resampled_y.size)
        rng.shuffle(reshuffled_indices)
        resampled_y += rng.normal(0., noise_scale, size=resampled_y.size)
        sets.append(
            (resampled_X[reshuffled_indices], resampled_y[reshuffled_indices])
        )
        indices.append(resampled_indices[reshuffled_indices])
    return sets, np.concatenate(indices), test_indices, train_indices

#=========================================================================================================

pd.set_option('display.max_rows', None, 'display.max_column', None)

xclass = pd.read_csv (r'data/raw/orionkl_parameters_xclass.csv')

#  sets index to environment and makes new df of specific environments to be embedded or plotted
# xclass.set_index('Molecule', inplace=True)
# xclass = xclass.loc[['OD', 'HDO']]
xclass.set_index('Environment', inplace=True)
#xclass.reset_index(inplace=True)


#madex = pd.read_csv (r'data/raw/orionkl_parameters_madex.csv')


df = pd.DataFrame(xclass)
#print(xclass)
embedder = VICGAE.from_pretrained()
#embedder = load_pipeline()

# creates new data frame or column using pandas
xclass['SMILES vector'] = ''

#check if embedder works
#xclass['SMILES vector'][0] = embedder(xclass['SMILES'][0])

j = 0
removed = []
for i in xclass['SMILES']:
	try:
		xclass['SMILES vector'][j] = embedder.embed_smiles(i).numpy()[0]
		#print(xclass['SMILES vector'][j])
		j += 1
	except:
		#print(i)
		removed.append(i)
		xclass.drop(labels=j, axis=0, inplace=True)
		j +=1

#print(xclass['SMILES vector'])
print('molecules not accepted by embedder:', removed)

for i in range(len(xclass['SMILES vector'][0])): 
	xclass['column %s' % i] = xclass['SMILES vector'].str[i]

xclass.drop('SMILES vector', axis=1, inplace=True)	

# splits up df into environments with embedded vectors
hot_core = xclass.loc['Hot Core']
south = xclass.loc['Hot Core (S)']
compact = xclass.loc['Compact Ridge']
plateau = xclass.loc['Plateau']
extended = xclass.loc['Extended Ridge']


#print(xclass.iloc[:, -130:])

# quick print function to help find the index of a column within the pandas dataframe when searched by column name
# print(df.columns.get_loc('column 0'))

i_regress = GradientBoostingRegressor()
norwegian_ridgeback = Ridge()
carl = GaussianProcessRegressor(normalize_y=True)

# splits the input data into training and testing sets - x and y arrays stay correlated to one another
# tuple in function must be given as np array so convert the df to np array, i.e. to.numpy()
boot_data, dummy, test_ind, train_ind = bootstrap_bill_turner([xclass.iloc[:, -130:].to_numpy(), np.log10(xclass['N_tot'].to_numpy())], seed=54664)
# print('test', test_ind)
# print(type(test_ind))
# print('train', train_ind)
# print(type(train_ind))


# test_i = test_ind.iloc[:]
# print(test_i)
# train_i = train_ind.iloc[:]
# print(train_i)

# print(len(boot_data[0][0][0]))
# print(len(boot_data[0][0][1]))
# print(len(boot_data[0][1][0]))
# print(len(boot_data[0][1][1]))

train_x = boot_data[0][0]
train_y = boot_data[0][1]

test_x = boot_data[1][0]
test_y = boot_data[1][1]


''' inputs training set into regressor - this is the actual machine learning part of the script
    stored value is the result of the training set data fit with the gradient boosting regressor '''
#result = i_regress.fit(train_x, train_y)
#result = norwegian_ridgeback.fit(train_x, train_y)
result = carl.fit(train_x, train_y)

# fit is used to predict column densities
# predict function is given the SMILES vector since this is what the model was trained on
# trained on correlating the SMILES vectors with the column densities, Ntot
N_obs = np.log10(xclass['N_tot']).to_numpy()
Labels = xclass['Molecule'].to_numpy()

pred_y = result.predict(xclass.iloc[:, -130:])
pred_train = result.predict(train_x)
pred_test = result.predict(test_x)

env1 = result.predict(hot_core.iloc[:, -130:])
env2 = result.predict(south.iloc[:, -130:])
env3 = result.predict(compact.iloc[:, -130:])
env4 = result.predict(plateau.iloc[:, -130:])
env5 = result.predict(extended.iloc[:, -130:])


# MSE and r^2 of the bootstrap N_col test data vs the predicted N_tot from bootstrap data
mean2 = metrics.mean_squared_error(test_y, pred_test)
r2 = metrics.r2_score(test_y, pred_test)
print('MSE:', mean2)
print('r-squared:', r2)

# all N_col from paper vs predicted paper N_tot from regressor 
mean_obs = metrics.mean_squared_error(N_obs, pred_y)
r2_obs = metrics.r2_score(N_obs, pred_y)
print('MSE:', mean_obs)
print('r-squared:', r2_obs)


test_obs  = np.delete(N_obs, train_ind)
train_obs = np.delete(N_obs, test_ind)

p_test  = np.delete(pred_y, train_ind)
p_train = np.delete(pred_y, test_ind)

# prints out paper and predicted col densities
# df = pd.DataFrame([Labels,N_obs,pred_y])
# print(df)

# scatter of training and test data split, all environments included
plt.scatter(test_obs, p_test, color='#1c9099', label='test')
plt.scatter(train_obs, p_train, color='#a6bddb', label='training')
plt.title('observed training and test split - all environments - GPR')
plt.xlabel(r'Observed $N_{tot}$')
plt.ylabel(r'Predicted $N_{tot}$')
plt.annotate(r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(N_obs, pred_y)))
			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(N_obs, pred_y))), (11.75, 17.5))
plt.legend(loc='upper left')
plt.xlim([11.5,19.5])
plt.ylim([11.5,19.5])
plt.show()


# scatter of all environments - individually color coded
plt.scatter(np.log10(hot_core['N_tot']).to_numpy(), env1, color='#d0d1e6', label='Hot Core')
plt.scatter(np.log10(south['N_tot']).to_numpy(), env2, color='#a6bddb', label='Hot Core (S)')
plt.scatter(np.log10(compact['N_tot']).to_numpy(), env3, color='#67a9cf', label='Compact Ridge')
plt.scatter(np.log10(plateau['N_tot']).to_numpy(), env4, color='#1c9099', label='Plateau')
plt.scatter(np.log10(extended['N_tot']).to_numpy(), env5, color='#016c59', label='Extended Ridge')
plt.title('Individual Environments - GPR')
plt.xlabel(r'Observed $N_{tot}$')
plt.ylabel(r'Predicted $N_{tot}$')
plt.annotate(r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(N_obs, pred_y)))
			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(N_obs, pred_y))), (12, 16))
plt.legend(loc='upper left')
plt.xlim([11.5,19.5])
plt.ylim([11.5,19.5])
plt.show()


# scatters of individual environments
fig, axs = plt.subplots(2,3)
axs[0,0].scatter(np.log10(hot_core['N_tot']).to_numpy(), env1, color='#d0d1e6', label=r'$R^2$ ='            
            + str('%.3f'%(metrics.r2_score(np.log10(hot_core['N_tot']).to_numpy(), env1)))
            + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(hot_core['N_tot']).to_numpy(), env1)))) 
axs[0,0].set_title('Hot Core')
axs[0,0].legend(fontsize='small', loc='upper left')
axs[0,1].scatter(np.log10(south['N_tot']).to_numpy(), env2, color='#a6bddb', label=r'$R^2$ ='           
             + str('%.3f'%(metrics.r2_score(np.log10(south['N_tot']).to_numpy(), env2)))
             + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(south['N_tot']).to_numpy(), env2)))) 
axs[0,1].set_title('Hot Core (S)')
axs[0,1].legend(fontsize='small', loc='upper left')
axs[0,2].scatter(np.log10(compact['N_tot']).to_numpy(), env3, color='#67a9cf', label=r'$R^2$ ='            
             + str('%.3f'%(metrics.r2_score(np.log10(compact['N_tot']).to_numpy(), env3)))
             + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(compact['N_tot']).to_numpy(), env3)))) 
axs[0,2].set_title('Compact Ridge')
axs[0,2].legend(fontsize='small', loc='upper left')
axs[1,0].scatter(np.log10(plateau['N_tot']).to_numpy(), env4, color='#1c9099', label=r'$R^2$ ='            
             + str('%.3f'%(metrics.r2_score(np.log10(plateau['N_tot']).to_numpy(), env4)))
             + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(plateau['N_tot']).to_numpy(), env4)))) 
axs[1,0].set_title('Plateau')
axs[1,0].legend(fontsize='small', loc='upper left')
axs[1,1].scatter(np.log10(extended['N_tot']).to_numpy(), env5, color='#016c59', label=r'$R^2$ ='            
             + str('%.3f'%(metrics.r2_score(np.log10(extended['N_tot']).to_numpy(), env5)))
             + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(extended['N_tot']).to_numpy(), env5)))) 
axs[1,1].set_title('Extended Ridge')
axs[1,1].legend(fontsize='small', loc='upper left')
fig.delaxes(axs[1,2])
fig.tight_layout()


for axs in fig.get_axes():
	axs.label_outer()

# plt.plot([], [], '', label='Hot Core: \n' + r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(np.log10(hot_core['N_tot']).to_numpy(), env1)))
# 			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(hot_core['N_tot']).to_numpy(), env1)))
# 			 + '\nHot Core (S): \n' + r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(np.log10(south['N_tot']).to_numpy(), env2)))
# 			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(south['N_tot']).to_numpy(), env2)))
# 			 + '\nCompact Ridge: \n' + r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(np.log10(compact['N_tot']).to_numpy(), env3)))
# 			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(compact['N_tot']).to_numpy(), env3)))
# 			 + '\nPlateau: \n' + r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(np.log10(plateau['N_tot']).to_numpy(), env4)))
# 			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(plateau['N_tot']).to_numpy(), env4)))
# 			 + '\nExtended Ridge: \n' + r'$R^2$ ='            + str('%.3f'%(metrics.r2_score(np.log10(extended['N_tot']).to_numpy(), env5)))
# 			 + '\nMSE = ' + str('%.3f'%(metrics.mean_squared_error(np.log10(extended['N_tot']).to_numpy(), env5))))
# plt.legend(loc='lower right')

plt.show()

