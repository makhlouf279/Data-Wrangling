import pandas as pd
import matplotlib.pylab as plt
from IPython.display import display
import numpy as np
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data")
print(df.shape)
print(df.columns)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", names = headers)
print(df.head())
#pour afficher tout les colones
"""
pd.options.display.max_columns = None 
display(df)
"""
#Convert "?" to NaN

df.replace("?", np.nan, inplace = True)
print(df.head(5))

#Evaluating for Missing Data

missing_data = df.isnull()
print(missing_data.head(5))

#Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   
