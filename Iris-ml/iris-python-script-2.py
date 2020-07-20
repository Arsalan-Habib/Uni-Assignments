import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# reading the new updated iris csv file
df = pd.read_csv('./csv-files/iris_corrected.csv', header=0)


# grouping by classification and using aggregate operations to get different statistics
df_grouped = df.groupby('classification')[['Petal.Ratio', 'Sepal.Ratio']].agg(
    ['mean', 'median', 'min', 'max', 'std'])

print(df_grouped)


# Visualizing the Data

# Histogram
ax = df.hist(column=['petal-width', 'petal-length', 'sepal-width',
                     'sepal-length'], grid=False, figsize=(10, 7))

# saving the histogram as an image
# fig = ax[0][0].get_figure()
# fig.savefig('./output-images/histogram.png')

# scatter-plots between petal-width and all other features
p_width_plot = sns.pairplot(df, y_vars="petal-width",
                            x_vars=df.columns.values, hue='classification')

# saving the scatterplot as an image
# p_width_plot.savefig('./output-images/petal-width-scatterplot.png')

# scatter-plots between petal-length and all other features
p_length_plot = sns.pairplot(df, y_vars="petal-length",
                             x_vars=df.columns.values, hue='classification')

# saving the scatterplot as an image
# p_length_plot.savefig('./output-images/petal-length-scatterplot.png')


# scatter-matrix
sm = pd.plotting.scatter_matrix(df[['petal-width', 'petal-length',
                                    'sepal-width', 'sepal-length']], alpha=0.8, figsize=(8, 8))

# saving the scatter-matrix as an image
# sm_fig = sm[0][0].get_figure()
# sm_fig.savefig('./output-images/scatter-matrix.png')


# Building the Classifier

# separating the target varaiable
X = df.values[:, 1:7]
Y = df.values[:, 0]

# splitting the Data set into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)

# Function to perform training with Entropy
clf_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=1, max_depth=4, min_samples_leaf=3)

# Training the model
clf_entropy.fit(X_train, y_train)

# Testing the test dataset
y_pred_en = clf_entropy.predict(X_test)

# Checking the accuracy of the model
print('\nAccuracy of the classifier is', accuracy_score(y_test, y_pred_en)*100)

# Use the model to predict the class of a flower. (It predicts it correctly i.e the flower below does actually belong to the iris-setosa class)
print('\nThe given flower belongs to the class of', clf_entropy.predict(
    [[4.4, 2.9, 0.6590909090909091, 1.4, 0.2, 0.14285714285714288]]))
