### In this file, we assume preprocessing has already happenned.
### We will compare the performance of different classifiers on the preprocessed data.

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

period_features_glove_50 = pd.read_csv('preprocessed-data/period_features_glove_50.csv')
period_features_glove_200 = pd.read_csv('preprocessed-data/period_features_glove_200.csv')
period_features_bert = pd.read_csv("preprocessed-data/period_features_bert.csv")

period_features_test_glove_50 = pd.read_csv('preprocessed-data/period_features_test_glove_50_no_retweets_etc.csv')
period_features_test_glove_200 = pd.read_csv('preprocessed-data/period_features_test_glove_200_no_retweets_etc.csv')
period_features_test_bert = pd.read_csv("preprocessed-data/period_features_test_bert.csv")

period_features = [period_features_glove_50, period_features_glove_200, period_features_bert]
period_features_test = [period_features_test_glove_50, period_features_test_glove_200, period_features_test_bert]

XX = [pf.drop(columns=["EventType", "MatchID", "PeriodID", "ID"]).values for pf in period_features]
yy = [pf["EventType"].values for pf in period_features]
XX_eval = [pf.drop(columns=["MatchID", "PeriodID", "ID"]).values for pf in period_features_test]
XX_train = 3*[0]
XX_test = 3*[0]
yy_train = 3*[0]
yy_test = 3*[0]
for i in range(3):
    XX_train[i], XX_test[i], yy_train[i], yy_test[i] = train_test_split(XX[i], yy[i], test_size=0.3, random_state=42)

accuracy_results = {}
accuracy_results["Model"] = ["GloVe-50", "GloVe-200", "BERT"]

# Dummy Classifier
print("Beginning dummy classifier...")
beginning = time.time()
dummy_results = []
for i in range(3):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(XX_train[i], yy_train[i])
    dummy_results.append(accuracy_score(yy_test[i], dummy.predict(XX_test[i])))
    dummy.fit(XX[i], yy[i])
    y_pred = dummy.predict(XX_eval[i])
    pred_df = pd.concat([period_features_test[i][["ID"]].reset_index(drop=True), pd.DataFrame(y_pred, columns=["EventTypePredicted"])], axis=1)
    pred_df.to_csv(f"predictions/predictions_{accuracy_results['Model'][i]}_dummy.csv", index=False)
accuracy_results["Dummy"] = dummy_results
print(f"Dummy classifier took {time.time()-beginning} seconds.")

# logistic regression
print("Beginning logistic regression...")
beginning = time.time()
lr_results = []
for i in range(3):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(XX_train[i], yy_train[i])
    lr_results.append(accuracy_score(yy_test[i], lr.predict(XX_test[i])))
    lr.fit(XX[i], yy[i])
    y_pred = lr.predict(XX_eval[i])
    pred_df = pd.concat([period_features_test[i][["ID"]].reset_index(drop=True), pd.DataFrame(y_pred, columns=["EventTypePredicted"])], axis=1)
    pred_df.to_csv(f"predictions/predictions_{accuracy_results['Model'][i]}_lr.csv", index=False)
    
accuracy_results["Logistic Regression"] = lr_results
print(f"Logistic regression took {time.time()-beginning} seconds.")

# Random Forest
print("Beginning random forest...")
beginning = time.time()
rf_results = []
for i in range(3):
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(XX_train[i], yy_train[i])
    rf_results.append(accuracy_score(yy_test[i], rf.predict(XX_test[i])))
    rf.fit(XX[i], yy[i])
    y_pred = rf.predict(XX_eval[i])
    pred_df = pd.concat([period_features_test[i][["ID"]].reset_index(drop=True), pd.DataFrame(y_pred, columns=["EventTypePredicted"])], axis=1)
    pred_df.to_csv(f"predictions/predictions_{accuracy_results['Model'][i]}_rf.csv", index=False)
accuracy_results["Random Forest"] = rf_results
print(f"Random forest took {time.time()-beginning} seconds.")

# SVM
print("Beginning SVM...")
beginning = time.time()
svm_results = []
for i in range(3):
    svm = SVC(random_state=42, kernel='linear', probability=True)
    svm.fit(XX_train[i], yy_train[i])
    svm_results.append(accuracy_score(yy_test[i], svm.predict(XX_test[i])))
    svm.fit(XX[i], yy[i])
    y_pred = svm.predict(XX_eval[i])
    pred_df = pd.concat([period_features_test[i][["ID"]].reset_index(drop=True), pd.DataFrame(y_pred, columns=["EventTypePredicted"])], axis=1)
    pred_df.to_csv(f"predictions/predictions_{accuracy_results['Model'][i]}_svm.csv", index=False)
accuracy_results["SVM"] = svm_results
print(f"SVM took {time.time()-beginning} seconds.")

# XGBoost
print("Beginning XGBoost...")
beginning = time.time()
xgb_results = []
for i in range(3):
    xgb = XGBClassifier(
    random_state=42,
    learning_rate=0.05,  # Reduced
    n_estimators=200,    # Increased
    max_depth=3,         # Reduced to prevent overfitting
    min_child_weight=3,  # Helps with overfitting
    subsample=0.8,       # Use 80% of data per tree
    colsample_bytree=0.8 # Use 80% of features per tree
)
    xgb.fit(XX_train[i], yy_train[i])
    xgb_results.append(accuracy_score(yy_test[i], xgb.predict(XX_test[i])))
    xgb.fit(XX[i], yy[i])
    y_pred = xgb.predict(XX_eval[i])
    pred_df = pd.concat([period_features_test[i][["ID"]].reset_index(drop=True), pd.DataFrame(y_pred, columns=["EventTypePredicted"])], axis=1)
    pred_df.to_csv(f"predictions/predictions_{accuracy_results['Model'][i]}_xgb.csv", index=False)
accuracy_results["XGBoost"] = xgb_results
print(f"XGBoost took {time.time()-beginning} seconds.")

# Bagged SVM
print("Beginning bagged SVM...")
beginning = time.time()
bagged_svm_results = []
for i in range(3):
    base_svm = SVC(random_state=42, kernel='rbf', probability=True)
    bagged_svm = BaggingClassifier(
        estimator=base_svm,
        n_estimators=70,  # you can adjust this number
        max_samples=0.8,  # you can adjust this fraction
        random_state=42
    )
    bagged_svm.fit(XX_train[i], yy_train[i])
    bagged_svm_results.append(accuracy_score(yy_test[i], bagged_svm.predict(XX_test[i])))
    bagged_svm.fit(XX[i], yy[i])
    y_pred = bagged_svm.predict(XX_eval[i])
    pred_df = pd.concat([period_features_test[i][["ID"]].reset_index(drop=True), pd.DataFrame(y_pred, columns=["EventTypePredicted"])], axis=1)
    pred_df.to_csv(f"predictions/predictions_{accuracy_results['Model'][i]}_bagged_svm.csv", index=False)
accuracy_results["Bagged SVM"] = bagged_svm_results
print(f"Bagged SVM took {time.time()-beginning} seconds.")

results = pd.DataFrame(accuracy_results)
print("Accuracy results on test set:")
print(results)
results.to_csv("classifier_comparison.csv", index=False)