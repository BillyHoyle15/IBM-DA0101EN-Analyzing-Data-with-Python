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

'''
Continuous numerical variables:
Continuous numerical variables are variables that may contain any value within 
some range. Continuous numerical variables can have the type "int64" or 
"float64". A great way to visualize these variables is by using scatterplots
with fitted lines.

In order to start understanding the (linear) relationship between an individual 
variable and the price. We can do this by using "regplot", which plots the 
scatterplot plus the fitted regression line for the data.

Let's see several examples of different linear relationships:
    
Positive linear relationship
Let's find the scatterplot of "engine-size" and "price"    
'''
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

'''
As the engine-size goes up, the price goes up: this indicates a positive 
direct correlation between these two variables. Engine size seems like a 
pretty good predictor of price since the regression line is almost a perfect 
diagonal line.

We can examine the correlation between 'engine-size' and 'price' and see it's 
approximately 0.87
'''
print(df[["engine-size", "price"]].corr())

#Highway mpg is a potential predictor variable of price
sns.regplot(x="highway-mpg", y="price", data=df)
#NOTE! Comment out the previous plottings for plotting correctly.

'''

As the highway-mpg goes up, the price goes down: this indicates an 
inverse/negative relationship between these two variables. Highway mpg could 
potentially be a predictor of price.

We can examine the correlation between 'highway-mpg' and 'price' and see 
it's approximately -0.704
'''

print(df[['highway-mpg', 'price']].corr())

  
