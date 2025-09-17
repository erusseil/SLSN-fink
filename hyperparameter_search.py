from itertools import product
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
from sklearn.model_selection import train_test_split
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import KFold
import kernel as kern


features = pd.read_parquet(kern.classified_path)
features['label_SLSN'] = 0
features.loc[features['label'].isin(kern.SLSN_types), 'label_SLSN'] = 1
X = features[kern.features_to_use]
y = features['label_SLSN']

param_grid = {
    'reg_lambda': [0, 1, 10, 50],
    'reg_alpha': [0, 1, 10, 50],
    'max_depth': [5, 6, 7, 8, 9, 10],
    'max_delta_step': [0, 1, 5, 10],
    'learning_rate':[0.05, 0.2, 0.5, 1]
}

kf = KFold(n_splits=10, shuffle=True)
scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)

# Get the keys and values
keys = param_grid.keys()
values = param_grid.values()

# Create combinations
combinations = list(product(*values))

# Turn each combination into a dictionary
param_combinations = [dict(zip(keys, combo)) for combo in combinations]

params_scores = []
for idx, params in enumerate(param_combinations):

    if idx%10 == 0:
        print(idx, "/", len(param_combinations))

    f2, f1, f05 = [], [], []

    for idx, (ktrain_index, ktest_index) in enumerate(kf.split(np.unique(features['objectId']))):

        # We make sure that a given objectId can only be in train or test
        train_index = features[features['objectId'].isin(np.unique(features['objectId'])[ktrain_index])].index
        test_index = features[features['objectId'].isin(np.unique(features['objectId'])[ktest_index])].index
        
        clf = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic',
                                scale_pos_weight=scale_pos_weight, **params)
    
        
        clf.fit(X.loc[train_index], y.loc[train_index])
    
        pred = clf.predict(X.loc[test_index])
        true = y.loc[test_index]
    
        f05.append(fbeta_score(true, pred, beta=0.5))
        f1.append(fbeta_score(true, pred, beta=1))
        f2.append(fbeta_score(true, pred, beta=2))

    params_scores.append([np.mean(f05), np.mean(f1), np.mean(f2)])

summary = pd.DataFrame(data = {"hyperparameters":param_combinations,
                     "f0.5":np.array(params_scores)[:, 0],
                     "f1":np.array(params_scores)[:, 1],
                     "f2":np.array(params_scores)[:, 2]})

summary.to_parquet(kern.hyperparameter_path)