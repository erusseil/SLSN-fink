import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import KFold
import seaborn as sns
import kernel as kern


# Split the full feature extraction into sub files
everything = pd.read_parquet(kern.full_extraction_path)
valid_mask = everything.isnull().sum(axis=1)==0
extracted = everything[valid_mask]
classified_mask = extracted['label']!=''
classified = extracted[classified_mask]
unclassified = extracted[~classified_mask]

extracted.to_parquet(kern.extracted_path)
classified.to_parquet(kern.classified_path)
unclassified.to_parquet(kern.unclassified_path)

features = classified.sample(frac=1)

# Test the model performances using a K-fold procedure
# Create a 0/1 label column
features['label_SLSN'] = 0

features.loc[features['label'].isin(kern.SLSN_types), 'label_SLSN'] = 1

X = features[kern.features_to_use]
y = features['label_SLSN']

scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)
kf = KFold(n_splits=10, shuffle=True, random_state=0)

search_hyper = pd.read_parquet(kern.hyperparameter_path)

hyper_str = "f1"
optimal_hyper = search_hyper.iloc[np.argmax(search_hyper[hyper_str])]['hyperparameters']


full_cm = []
TP, TN, FP, FN = [], [], [], []
feature_importance = []

for idx, (ktrain_index, ktest_index) in enumerate(kf.split(np.unique(features['objectId']))):

    # We make sure that a given objectId can only be in train or test
    train_index = features[features['objectId'].isin(np.unique(features['objectId'])[ktrain_index])].index
    test_index = features[features['objectId'].isin(np.unique(features['objectId'])[ktest_index])].index

    clf = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic',
                            scale_pos_weight=scale_pos_weight, **optimal_hyper)

    clf.fit(X.loc[train_index], y.loc[train_index])
    feature_importance.append(clf.feature_importances_)
    
    pred = clf.predict(X.loc[test_index])
    true = y.loc[test_index]
    full_cm.append(confusion_matrix(true, pred, labels=[0, 1]))

    TP.append(test_index[(pred==1) & (true==1)])
    TN.append(test_index[(pred==0) & (true==0)])
    FP.append(test_index[(pred==1) & (true==0)])
    FN.append(test_index[(pred==0) & (true==1)])

TP = np.concatenate(TP).ravel()
TN = np.concatenate(TN).ravel()
FP = np.concatenate(FP).ravel()
FN = np.concatenate(FN).ravel()

mean_matrix = np.mean(full_cm, axis=0)
std_matrix = np.std(full_cm, axis=0)

# SAVE MATRIX
plt.figure(figsize=(7, 6))
ax = sns.heatmap(mean_matrix, annot=False, fmt="", xticklabels=["Non-SLSN", "SLSN"],
            yticklabels=["Non-SLSN", "SLSN"], linewidths=3,
           cmap=kern.fink_gradient, cbar=True)

# Add custom annotations
for i in range(mean_matrix.shape[0]):
    for j in range(mean_matrix.shape[1]):
        value = f"{round(mean_matrix[i, j])}"
        uncert = f"± {round(std_matrix[i, j])}"
        ax.text(j+0.5, i+0.45, value, ha='center', va='center',
                fontsize=28, color='white', fontweight='bold')
        ax.text(j+0.5, i+0.60, uncert, ha='center', va='center',
                fontsize=15, color='white')

plt.xlabel("Predicted label", labelpad=15)
plt.ylabel("True label", labelpad=15)
plt.savefig(f"kfold_confusion.png", bbox_inches='tight')

# Finally we train a model with the full dataset, this is our definitive model. 
clf = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic',
                            scale_pos_weight=scale_pos_weight, **optimal_hyper)
clf.fit(X, y)
joblib.dump(clf, kern.classifier_path)