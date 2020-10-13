import pandas as pd

import matplotlib as plt
from matplotlib import pyplot
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

#Calculate the mean value for 'stroke' column

avg_stroke= df['stroke'].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
#Replace NaN by mean value
df["stroke"].replace(np.nan, avg_stroke, inplace=True)


#Calculate the mean value for the 'horsepower' column:
avg_horsepower = df['horsepower'].astype("float").mean(axis=0)
print("Average horsepower:", avg_horsepower)

#Replace "NaN" by mean value:

df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)


#Calculate the mean value for 'peak-rpm' column:

avg_peak= df['peak-rpm'].astype("float").mean(axis=0)
print("Average peak-rpm :", avg_peak)

#Replace "NaN" by mean value:
df["peak-rpm"].replace(np.nan, avg_peak, inplace=True)

#To see which values are present in a particular column, we can use the ".value_counts()" method:

df['num-of-doors'].value_counts()
"""
We can see that four doors are the most common
 type. We can also use the ".idxmax()" method to 
 calculate for us the most common type automatically:

"""
df['num-of-doors'].value_counts().idxmax()

#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

#Finally, let's drop all rows that do not have price data:

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)    
print(df.head())

#Good! Now, we obtain the dataset with no missing values.
#The last step in data cleaning is checking and 
#making sure that all data is in the correct format 
#(int, float, text or other).

print(df.dtypes)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print(df.dtypes)

#Data Standardization
print(df.head())

df['city-L/100km']=235/df["city-mpg"]
print(df.head())

df['highway-L/100km']=235/df["highway-mpg"]
# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
print(df.head())


#Data Normalization

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

# Write your code below and press Shift+Enter to execute 
df['height'] = df['height']/df['height'].max() 
# show the scaled columns
print(df[["length","width","height"]].head())

#Binning
#Convert data to correct format
df["horsepower"]=df["horsepower"].astype(int, copy=True)


#Lets plot the histogram of horspower, to see what the distribution of horsepower looks like.
plt.pyplot.hist(df["horsepower"])
# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")



bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

#We set group names:
group_names = ['Low', 'Medium', 'High']

#We apply the function "cut" the determine what each value of "df['horsepower']" belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))

#Lets see the number of vehicles in each bin.

print(df["horsepower-binned"].value_counts())

#Lets plot the distribution of each bin.
pyplot.bar(group_names, df["horsepower-binned"].value_counts())
# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#Bins visualization

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


#Indicator variable (or dummy variable)
print(df.columns)

#get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable= pd.get_dummies(df["fuel-type"])
print(dummy_variable.head())


#change column names for clarity

dummy_variable.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
print(dummy_variable.head())

# merge data frame "df" and "dummy_variable" 
df = pd.concat([df, dummy_variable], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)
print(df.head())


# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
print(dummy_variable2.head())

# merge data frame "df" and "dummy_variable_2" 
df = pd.concat([df, dummy_variable2], axis=1)

# drop original column "aspiration" from "df"
df.drop("aspiration", axis = 1, inplace=True)
print(df.head())

#save the new csv 

df.to_csv('clean_df.csv')








