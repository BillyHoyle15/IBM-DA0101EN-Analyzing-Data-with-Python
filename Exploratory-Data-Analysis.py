# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:08:02 2019

@author: Suat
"""

'''
Data Analysis with Python
Exploratory Data Analysis

Welcome!
In this file, we will explore several methods to see if certain 
characteristics or features can be used to predict car price.

This is the 3rd part of the analysis. Check out 
'''

'''
What are the main characteristics which have the most impact on the car price?
1. Import Data from Module 2
'''

import pandas as pd
import numpy as np

#load data and store in dataframe df:
#This dataset was hosted on IBM Cloud object

path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
print(df.head())

'''
2. Analyzing Individual Feature Patterns using Visualization
'''

import matplotlib.pyplot as plt
import seaborn as sns

'''
How to choose the right visualization method?
When visualizing individual variables, it is important to first understand 
what type of variable you are dealing with. This will help us find the 
right visualization method for that variable.
''' 
# list the data types for each column
print(df.dtypes)

#for example, we can calculate the correlation between variables of type 
#"int64" or "float64" using the method "corr":

print(df.corr())

#The diagonal elements are always one; we will study correlation more 
#precisely Pearson correlation in-depth at the end of this file.


#Find the correlation between the following columns: 
#bore, stroke,compression-ratio , and horsepower.

print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())  
