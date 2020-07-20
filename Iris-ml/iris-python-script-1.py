import pandas as pd
import numpy as np

# reading the dataset from the .csv file in the csv-files folder
# the dataset did not have any column headers therefore header is set to none
df = pd.read_csv('./csv-files/iris.csv', header=None)

# originally the dataset did not have column headers so i added them
df.columns = ['petal-width', 'petal-length',
              'sepal-width', 'sepal-length', 'classification']

# changing the dataframe to use 1-indexing
df.index = np.arange(1, len(df)+1)

# printing the first 10 rows of the data
print(df.head(10))

# checking the shape and data types of the dataframe
print(df.shape, df.dtypes)

# correcting the values at 35 and 38
df.at[35, 'sepal-length'] = 0.2
df.at[38, 'petal-length'] = 3.6
df.at[38, 'sepal-width'] = 1.4

# adding a column called petal.ratio which is the ratio of petal length to width
df['Petal.Ratio'] = df['petal-length']/df['petal-width']

# adding a column called sepal.ratio which is the ratio of sepal length to width
df['Sepal.Ratio'] = df['sepal-length']/df['sepal-width']

# re-ordering the columns for better understanding and readability
cols = df.columns.tolist()
cols = cols[-3:-2] + cols[0:2] + cols[-2:-1] + cols[2:4] + cols[-1:]
df = df[cols]


# saving the new corrected csv file
df.to_csv('iris_corrected.csv', index=False)
