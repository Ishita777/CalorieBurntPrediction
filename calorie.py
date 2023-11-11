#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#Data Collection and Preprocessing

calories = pd.read_csv('/content/calories.csv')
calories.head()
exercise = pd.read_csv('/content/exercise.csv')
exercise.head()

#combining the dataframes
exercise_data = pd.concat([exercise, calories['Calories']], axis=1)
exercise_data.head()
exercise_data.shape
exercise_data.info()
exercise_data.isnull().sum()

#Data Analysis
exercise_data.describe()

#Data Visualization

#Gives some theme or plot for our graph, like grid, etc.
sns.set()

#Plotting the gender column in the count plot
sns.countplot(exercise_data, x="Gender")

#Finding the distribution of age column
sns.histplot(exercise_data['Age'])

#Finding the distribution of height column
sns.histplot(exercise_data['Height'])

#Finding the distribution of weight columns
sns.histplot(exercise_data['Weight'])

#Correlation
corr = exercise_data.corr()

#Constructing a heat map to understand the correlation

#Giving the size of the plot
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

#Converting textual data to numerical values
exercise_data.replace({'Gender':{'male':0, 'female':1}}, inplace=True)

#Separating features and target
X = exercise_data.drop(columns=['Calories', 'User_ID'], axis=1)
Y = exercise_data['Calories']

#Splitting the data into testing and training data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Model Training(XGBoost Regressor)
#Loading the model
model = XGBRegressor()

#Training the model with x_train
model.fit(x_train, y_train)

#Prediction on x_test data
test_data_prediction = model.predict(x_test)
print(test_data_prediction)

#Evaluation
#Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)
