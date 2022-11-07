# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:23:20 2022

@author: Leonard
"""

from Data import Data
from Model import Model
from sklearn.metrics import mean_squared_error
data = Data()
print(df:=data.df_processed.head())
data.visualize_data()
model=Model(data.df_processed)
results=model.fit_logit(plot=True)
print(results.summary())
test_data_path = "data/test.csv"
prediction=model.predict_logit(save=False)
model.plot_SVC_features(3,7,gamma=3,poly=True)
# features_list=[i for i in range(1,9)]
features_list=[3,6,7]
score_train, score_test, selected_features = model.fit_SVC(features_list, kernel='rbf',gamma='auto')
print(f"Mean accuracy of Random Forest Predictions on the train data is: {round(score_train,4)} and the test data is: {round(score_test,4)}\n")

res_svc=model.predict_SVC(features_list, kernel='rbf',gamma='auto',save=True)
score_train_rf, score_test_rf, selected_features = model.fit_RF(features_list, n_estimators=100, train_test_split=0.9)
print(f"Mean accuracy of Random Forest Predictions on the train data is: {round(score_train_rf,4)} and the test data is: {round(score_test_rf,4)}\n")
res_rf=model.predict_RF(features_list, n_estimators=100, save=True)

difference=mean_squared_error(res_rf,res_svc)
print(f"The difference between both methods is {round(difference*100,1)}%\n")