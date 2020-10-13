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
print(df.head(50))

#Evaluating for Missing Data

missing_data = df.isnull()
print(missing_data.head(5))

#Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   

#Deal with missing data// Replace by mean:
    """
    Replace by mean:

"normalized-losses": 41 missing data, replace them with mean
"stroke": 4 missing data, replace them with mean
"bore": 4 missing data, replace them with mean
"horsepower": 2 missing data, replace them with mean
"peak-rpm": 2 missing data, replace them with mean

Replace by frequency:

"num-of-doors": 2 missing data, replace them with "four".
Reason: 84% sedans is four doors. Since four doors is most frequent,
 it is most likely to occur
Drop the whole row:

"price": 4 missing data, simply delete the whole row
Reason: price is what we want to predict. Any data entry 
without price data cannot be used for prediction; therefore 
any row now without price data is not useful to us
"""
#Calculate the average of the column
avg_norm_loss= df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

#Replace "NaN" by mean value in "normalized-losses" column

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

#Calculate the mean value for 'bore' column
avg_bore= df['bore'].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)


#Replace NaN by mean value
df["bore"].replace(np.nan, avg_bore, inplace=True)


























