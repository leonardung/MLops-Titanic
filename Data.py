# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:50:54 2022

@author: Leonard
"""



import pandas as pd
from Helper.visualization import data_visualization

class Data:
    def __init__(self, datafile = "data/train.csv"):
        """
        Constructor
        
        Parameters
        ----------
        :datafile: Path to the dataset
        
        """
        self.df = pd.read_csv(datafile)
        self.df_processed = self.df.drop(['Ticket','Cabin'], axis=1)
        self.df_processed = self.df_processed.dropna() 
        
    def visualize_data(self):
        """
        Plots : 
            - Distribution of survival 
            
            - Survival by age 
            
            - Class distribution
            
            - Age distribution within classes
            
            - Passengers per boading location
            
            - Survivor with respect to gender
            
            - Survivor with respect to gender and class
        
        """
        data_visualization(self.df_processed)
        

if __name__ == '__main__':
    data = Data()
    print(data.df_processed.head())
    data.visualize_data()
