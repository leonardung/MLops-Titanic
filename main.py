# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:23:20 2022

@author: Leonard
"""

from Data import Data
from Model import Model
data = Data()
print(df:=data.df_processed.head())
# data.visualize_data()
model=Model(data.df_processed)
results=model.fit_logit(plot=False)
print(results.summary())
test_data_path = "data/test.csv"
prediction=model.predict_logit(save=False)
model.plot_SVC_features(3,7,gamma=3,poly=True)
features_list=[3,6,7]
score_train, score_test, selected_features = model.fit_SVC(features_list, kernel='rbf',gamma='auto')
res_svc=model.predict_SVC(features_list, kernel='rbf',gamma='auto',save=True)
score_train_rf, score_test_rf, selected_features = model.fit_RF(features_list, n_estimators=100, train_test_split=0.9)
res_rf=model.predict_RF(features_list, n_estimators=100, save=True)

error=sum(abs(res_rf-res_svc))/len(res_rf)