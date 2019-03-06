# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:19:59 2019

@author: Suat
"""

'''
DATA ANALYSIS with PYTHON
DATA WRANGLING

What is the purpose of Data Wrangling?
    Data Wrangling is the process of converting data from the initial format 
    to a format that may be better for analysis.

Question: What is the fuel consumption (L/100k) rate for the diesel car?

IMPORT DATA
You can find the "Automobile Data Set" from the following link: 
https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data. 
We will be using this data set throughout this course.
'''

import pandas as pd
import matplotlib.pylab as plt

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

#Python list headers containing name of headers

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

#Use the Pandas method read_csv() to load the data from the web address. 
#Set the parameter "names" equal to the Python list "headers".

df = pd.read_csv(filename, names = headers)

# To see what the data set looks like, we'll use the head() method.
print(df.head())

'''
As we can see, several question marks appeared in the dataframe; 
those are missing values which may hinder our further analysis.

So, how do we identify all those missing values and deal with them?
How to work with missing data?

Steps for working with missing data:

1. identify missing data
2. deal with missing data
3. correct data format

IDENTIFY and HANDLE MISSING VALUES
Identify missing values

Convert "?" to NaN
In the car dataset, missing data comes with the question mark "?". 
We replace "?" with NaN (Not a Number), which is Python's default missing 
value marker, for reasons of computational speed and convenience. 
Here we use the function:
.replace(A, B, inplace = True) 
to replace A by B
'''

import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
print(df.head(5))

'''
EVALUATING FOR MISSING DATA
The missing values are converted to Python's default. We use Python's built-in 
functions to identify these missing values. There are two methods to detect 
missing data:

1-.isnull()
2-.notnull()
The output is a boolean value indicating whether the value that is passed 
into the argument is in fact missing data.
'''
missing_data = df.isnull()
print(missing_data.head(5))

#"True" stands for missing value, while "False" stands for not missing value.

'''
Count missing values in each column
Using a for loop in Python, we can quickly figure out the number of missing 
values in each column. As mentioned above, "True" represents a missing value, 
"False" means the value is present in the dataset. In the body of the for loop 
the method ".value_counts()" counts the number of "True" values.
'''

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
    
'''
Based on the summary above, each column has 205 rows of data, seven columns 
containing missing data:

1. "normalized-losses": 41 missing data
2. "num-of-doors": 2 missing data
3. "bore": 4 missing data
4. "stroke" : 4 missing data
5. "horsepower": 2 missing data
6. "peak-rpm": 2 missing data
7. "price": 4 missing data
'''

'''
Deal with missing data
How to deal with missing data?
1. drop data
    a. drop the whole row
    b. drop the whole column
2. replace data
    a. replace it by mean
    b. replace it by frequency
    c. replace it based on other functions
Whole columns should be dropped only if most entries in the column are empty. 
In our dataset, none of the columns are empty enough to drop entirely. 
We have some freedom in choosing which method to replace data; however, 
some methods may seem more reasonable than others. We will apply each method 
to many different columns:

Replace by mean:

* "normalized-losses": 41 missing data, replace them with mean
* "stroke": 4 missing data, replace them with mean
* "bore": 4 missing data, replace them with mean
* "horsepower": 2 missing data, replace them with mean
* "peak-rpm": 2 missing data, replace them with mean

Replace by frequency:

* "num-of-doors": 2 missing data, replace them with "four".
    Reason: 84% sedans is four doors. Since four doors is most frequent, 
    it is most likely to occur

Drop the whole row:

* "price": 4 missing data, simply delete the whole row
    Reason: price is what we want to predict. Any data entry without price 
    data cannot be used for prediction; therefore any row now without price 
    data is not useful to us
'''

#Calculate the average of the column 
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

#Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#Calculate the mean value for 'bore' column
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

#Replace NaN by mean value
df["bore"].replace(np.nan, avg_bore, inplace=True)

# calculate the mean vaule for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)

#Calculate the mean value for the 'horsepower' column:
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

#Replace "NaN" by mean value:
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

#Calculate the mean value for 'peak-rpm' column:
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

#Replace NaN by mean value:
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#To see which values are present in a particular column, we can use the 
#".value_counts()" method:
print(df['num-of-doors'].value_counts())

#We can see that four doors are the most common type. We can also use 
#the ".idxmax()" method to calculate for us the most common type automatically:
print(df['num-of-doors'].value_counts().idxmax())

#The replacement procedure is very similar to what we have seen previously
#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

#Finally, let's drop all rows that do not have price data:
# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

print(df.head())

#Good! Now, we obtain the dataset with no missing values.

'''TO bE CONTINUED'''


