# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:09:52 2022

@author: Leonard
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm
import KaggleAux.predict as ka
from visualization import data_visualization, logit_visualization, SVC_features_visualization
import sklearn.ensemble as ske

class Model():
    def __init__(self, data: DataFrame,test_data_path="data/test.csv"):
        """
        Constructor
        
        Parameters
        ----------
        :data: clean data
        :test_data_path: Path to the test data
        
        """
        self.df = data
        self.test_data = pd.read_csv(test_data_path)
        self.test_data['Survived'] = 1.23
        self.formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
        
    def fit_logit(self,plot=False):
        """
        Fit the data with the "logit" model
        
        Parameters
        ----------
        :plot: plot the prediction
        
        Return
        ---------- 
        Fitted model (Use .summary() to show description)
        
        """
        formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' 
        # create a results dictionary to hold our regression results for easy analysis later        
        self.results = {} 
        # create a regression friendly dataframe using patsy's dmatrices function
        y,x = dmatrices(formula, data=self.df, return_type='dataframe')
        # instantiate our model
        model = sm.Logit(y,x)
        # fit our model to the training data
        res = model.fit()
        # save the result for outputing predictions later
        self.results['Logit'] = [res, formula]
        if plot:
            logit_visualization(x,y,res)
        return res
        
    def predict_logit(self,save=False):
        """
        predict the test data with the "logit" model
        
        Parameters
        ----------
        :save: Save the list to "data/output/logitregres.csv"
        
        Return
        ---------- 
        Series of the predictions
        
        """
        compared_resuts = ka.predict(self.test_data, self.results, 'Logit')
        compared_resuts = Series(compared_resuts)  # convert our model to a series for easy output
        if save:
            compared_resuts.to_csv("data/output/logitregres.csv")
        return compared_resuts
    
    def fit_SVC(self,features_list, kernel='rbf',gamma=3, train_test_split=0.9):
        """
        Fit the data with a C-Support Vector Classification
        
        Parameters
        ----------
        :features_list: list of the features index
        :kernel: Specifies the kernel type to be used in the algorithm (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)
        :gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ (‘scale’, ‘auto’ or float)
        :train_test_split: Percentage of data provided to the train set
        
        Returns
        -------
        :score_train: Accuracy of the train set
        :score_test: Accuracy of the test set
        :selected_features: List of the names of the features
    
        """
        y, x = dmatrices(self.formula_ml, data=self.df, return_type='dataframe')
        selected_features=list(x.columns[features_list])
        X = np.asarray(x)
        X = X[:,features_list]
        y = np.asarray(y)
        # needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape. 
        y = y.flatten()
        n_sample = len(X)

        np.random.seed(0)
        order = np.random.permutation(n_sample)

        X = X[order]
        y = y[order].astype(np.float64)

        # do a cross validation
        precentage_of_sample = int(train_test_split * n_sample)
        X_train = X[:precentage_of_sample]
        y_train = y[:precentage_of_sample]
        X_test = X[precentage_of_sample:]
        y_test = y[precentage_of_sample:]
        self.clf = svm.SVC(kernel=kernel, gamma=gamma,verbose=False).fit(X_train, y_train)                                                            
        score_train=self.clf.score(X_train,y_train)
        score_test=self.clf.score(X_test,y_test)
        return score_train, score_test, selected_features
    
    def predict_SVC(self,features_list, kernel='rbf',gamma=3, save=False):
        """
        Predict the test data with a C-Support Vector Classification
        
        Parameters
        ----------
        :features_list: list of the features index
        :kernel: Specifies the kernel type to be used in the algorithm (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)
        :gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ (‘scale’, ‘auto’ or float)
        :save: Save the list to "data/output/svm_{kernel}_{features_indexes}_g_{gamma}.csv"
        
        Return
        ---------- 
        Series of the predictions
        
        """
        _,x = dmatrices(self.formula_ml, data=self.test_data, return_type='dataframe')
        res_svm = self.clf.predict(x.iloc[:,features_list].dropna()) 
        res_svm = Series(res_svm)#,columns=['Survived'])
        
        if save: 
            features_indexes='_'.join([str(i) for i in features_list])
            res_svm.to_csv(f"data/output/svm_{kernel}_{features_indexes}_g_{gamma}.csv")
        return res_svm
    
    def plot_SVC_features(self, feature_1=2,feature_2=3,gamma=3,poly=False):
        """
        Plot the analysis of the features with different kernel ("linear", "rfb", "poly")
        
        Parameters
        ----------        
        :feature_1: 1st feature to analyse
        :feature_2: 2nd feature to analyse
        :poly: Analyse also the polynomial kernel (can be very slow compared to "linear" and "rfb")
        
        """
        y, x = dmatrices(self.formula_ml, data=self.df, return_type='dataframe')
        SVC_features_visualization(x,y,feature_1,feature_2,gamma=gamma,poly=poly)
        
    
    def fit_RF(self,features_list, n_estimators=100, train_test_split=0.9):
        """
        Fit the data with a random forest classifier
        
        Parameters
        ----------
        :features_list: list of the features index
        :n_estimators: The number of trees in the forest.
        :train_test_split: Percentage of data provided to the train set
        
        Returns
        -------
        :score_train: Accuracy of the train set
        :score_test: Accuracy of the test set
        :selected_features: List of the names of the features
    
        """
        y, x = dmatrices(self.formula_ml, data=self.df, return_type='dataframe')
        selected_features=list(x.columns[features_list])
        # RandomForestClassifier expects a 1 demensional NumPy array, so we convert


        X = np.asarray(x)
        X = X[:,features_list]  

        y = np.asarray(y)
        # needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape. 
        y = y.flatten()      

        n_sample = len(X)

        np.random.seed(0)
        order = np.random.permutation(n_sample)

        X = X[order]
        y = y[order].astype(np.float64)

        # do a cross validation
        precentage_of_sample = int(train_test_split * n_sample)
        X_train = X[:precentage_of_sample]
        y_train = y[:precentage_of_sample]
        X_test = X[precentage_of_sample:]
        y_test = y[precentage_of_sample:]
        #instantiate and fit our model
        self.results_rf = ske.RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)

        # Score the results
        score_train = self.results_rf.score(X_train, y_train)
        score_test = self.results_rf.score(X_test, y_test)
        return score_train, score_test, selected_features
    
    def predict_RF(self,features_list, n_estimators=100, save=False):
        """
        Predict the test data with a C-Support Vector Classification
        
        Parameters
        ----------
        :features_list: list of the features index
        :kernel: Specifies the kernel type to be used in the algorithm (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)
        :gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ (‘scale’, ‘auto’ or float)
        :save: Save the list to "data/output/rf_{n_estimators}_{features_indexes}.csv"
        
        Return
        ---------- 
        Series of the predictions
        
        """
        _,x = dmatrices(self.formula_ml, data=self.test_data, return_type='dataframe')
        res_rf = self.results_rf.predict(x.iloc[:,features_list].dropna()) 
        res_rf = Series(res_rf)#,columns=['Survived'])
        
        if save: 
            features_indexes='_'.join([str(i) for i in features_list])
            res_rf.to_csv(f"data/output/rf_{n_estimators}_{features_indexes}.csv")
        return res_rf
        