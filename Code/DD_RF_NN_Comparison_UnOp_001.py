# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:00:38 2024

@author: Dina Dinh
"""

#%% Basic packages

import pandas as pd 
import numpy as np  
import os
import math
import matplotlib.pyplot as plt

#%% Preprocessing packages

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.impute import SimpleImputer
#%% Model packages

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
#%% Sets the working directory input and output
dir1=os.path.realpath(__file__)
main_dir = os.path.dirname(os.path.dirname(dir1))
Ex0=pd.read_csv(main_dir+'/Data/YY_varying_withGrid_withoutSurrounding_06_25.csv')

#%% Split dataset into training and testing sets based on grid IDs
def gridid_train_test_split(x_array, y_array, test_size, grid_id_col_index):
    
    grid_id_col_index = grid_id_col_index - 1 # makes numbering of columns start at 0 instead of 1.
    
    start_grid_id = min(x_array[:, grid_id_col_index])
    end_grid_id = max(x_array[:, grid_id_col_index])
    grid_cell_count = end_grid_id - start_grid_id
    
    test_number_of_cells = math.ceil(grid_cell_count * test_size) # math.ceil rounds up to the nearest whole number.
    
    test_grid_ids = np.random.randint(start_grid_id, end_grid_id, test_number_of_cells) # Picking grid IDs from our data to use in the test set.
    
    grid_id_array = x_array[:, grid_id_col_index]
    
    test_indices = np.where(np.in1d(grid_id_array, test_grid_ids))[0] # finding which rows of our data have grid IDs that match the ones we randomly picked for test.
    
    X_test = x_array[test_indices]
    y_test = y_array[test_indices]
    
    total_records = len(x_array)
    mask = np.ones(total_records, dtype=bool) # mask starts with all True.
    mask[test_indices] = False # tells us which rows aren't used in the test set by setting the rows that are used in the test set to be False.
   # training set created by taking the rows from our data that are marked True in the mask.
    X_train = x_array[mask] 
    y_train = y_array[mask]

    return X_train, X_test, y_train, y_test, test_grid_ids
#%%
cols = list(Ex0)
cols.insert(0, cols.pop(cols.index('VRYieldVOl'))) # makes VRYieldVOl the first colmun.
cols.insert(1, cols.pop(cols.index('GridId'))) # moves gridID to the second column.

Ex1=Ex0.loc[:,cols] # applies column arrangement to Ex1
#%%
Ex1.dropna(subset=['VRYieldVOl'])
Ex2=Ex1[Ex1['VRYieldVOl']>0]
#%%
imputer = SimpleImputer(strategy='mean') # Inputs the mean as the missing value
Ex2=pd.DataFrame(imputer.fit_transform(Ex2), columns=Ex2.columns) # applies the imputer to our dataset
#%%
Ex3 = pd.get_dummies(Ex2)
all_zero_columns = Ex3.columns[Ex3.all(axis=0) == 0]
Ex3 = Ex3.drop(columns=all_zero_columns)
#%%
X=Ex3.iloc[:,1:len(Ex3.columns)].values # creates a matrix without the column names for the X-variables.
y=Ex3.iloc[:,0].values.flatten() # list of the response variable.

#%% Get grid id col index for train/test split
grid_col_index = cols.index('GridId')

#%% Optimizer and loss function
optimizer='rmsprop'
loss_m='mean_squared_error'


#%% Inititalize vectors to store the mse and set the number of repetitions
rf_mse_list= []
nn_mse_list=[]
best_rf_model = None
best_rf_mse = 50
best_nn_model = None
best_nn_mse = 50
reps=20

#%% Scales and encodes the data (Not needed for regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # standardize the X variables

#%% Main Loop
for i in range(reps):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=None)
    
    # Random forest regressor 
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    RF_y_pred = rf.predict(X_test)
    rf_mse = metrics.mean_squared_error(y_test, RF_y_pred)
    rf_mse_list.append(rf_mse) # adds current mse to the list with each iteration
    if rf_mse < best_rf_mse:
        best_rf_mse = rf_mse
        best_rf_model = rf
    
        # Neural Network with some architecture
    nn=Sequential()
    nn.add(Dense(50, activation='relu', input_shape=[X_train.shape[1]]))
    nn.add(Dense(20, activation='relu'))
    nn.add(Dense(1))
    nn.compile(optimizer=optimizer, loss=loss_m, metrics=['accuracy'])
    nn.fit(X_train, y_train)
    NN_y_pred = nn.predict(X_test)
    
  #
    c=metrics.mean_squared_error(y_test, NN_y_pred)
    nn_mse_list.append(c)
    if c < best_nn_mse:
     best_nn_mse=c
     best_nn_model=nn

    print("Finished Itteration",i+1)
  #    
        
#%%     
Results=pd.DataFrame({'Random_Forest':rf_mse_list,"Neural Network":nn_mse_list})      

#%% Computes and Reports mean mse
mean_rf_mse = sum(rf_mse_list) / len(rf_mse_list)
mean_nn_mse = sum(nn_mse_list) / len(nn_mse_list)

#%%
fig, ax = plt.subplots()
ax.boxplot(Results)
ax.set_title('Side by Side Boxplot of MSE for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('mse')
xticklabels=['Random Forest','Neural Network']
ax.set_xticklabels(xticklabels)
ax.yaxis.grid(True)
plt.show()

#%% Prediction on best model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
RF_pred=rf.predict(X_test)
NN_pred=nn.predict(X_test)

#%% R2, RMSE, and MAE for each model
r2_RF = metrics.r2_score(y_test, RF_pred)
rmse_RF = np.sqrt(metrics.mean_squared_error(y_test, RF_pred, squared=False))
mae_RF = metrics.mean_absolute_error(y_test, RF_pred)

r2_NN = metrics.r2_score(y_test, NN_pred)
rmse_NN = np.sqrt(metrics.mean_squared_error(y_test, NN_pred, squared=False))
mae_NN = metrics.mean_absolute_error(y_test, NN_pred)

mean_y_test = np.mean(y_test)
color_values = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))

#%% RF Scatterplot of pred vs true
plt.figure(figsize=(16, 4))
# Random Forest
plt.subplot(1, 3, 1)
plt.scatter(y_test, RF_pred, c=color_values, cmap='viridis', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Measured Yield')
plt.ylabel('Predicted Yield')
plt.title('Measured vs Predicted Yield (Random Forest)')
plt.text(0.05, 0.95, f'RMSE: {rmse_RF:.2f}\nMean Y: {mean_y_test:.2f}\nMAE: {mae_RF:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

#%% NN Scatterplot of pred vs true
plt.figure(figsize=(16, 4))
# NN
plt.subplot(1, 3, 1)
plt.scatter(y_test, NN_pred, c=color_values, cmap='viridis', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Measured Yield')
plt.ylabel('Predicted Yield')
plt.title('Measured vs Predicted Yield (Neural Network)')
plt.text(0.05, 0.95, f'RMSE: {rmse_NN:.2f}\nMean Y: {mean_y_test:.2f}\nMAE: {mae_NN:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()
