#Install necessary libraries
pip install pandas
pip install seaborn
pip install matplotlib
pip install -U scikit-learn

#Data Analysis and Vizualization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline

#Model Development and Evaluation
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

#Load dataset
df = pd.read_csv("kc_house_data.csv")

#Data Wrangling
print(df.dtypes)
print(df.describe())

#Exploratory Analysis
#How does the distribution of house prices look?
plt.hist(df['price'], bins=25, color='skyblue', edgecolor='black')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

#How does the price vary by the number of bedrooms and bathrooms?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bedrooms', y='bathrooms', hue='price', palette='coolwarm', size='price', sizes=(20, 200), alpha=0.6)
plt.title('Price Variation by Bedrooms and Bathrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Number of Bathrooms')
plt.legend(title='Price', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#Does the size of the house (sqft_living) have a stronger correlation with the price than the lot size (sqft_lot)?
correlation_matrix = df[['sqft_living', 'sqft_lot', 'price']].corr()
print(correlation_matrix)

#Is there a relationship between the number of floors and house price?
plt.figure(figsize=(8, 5))
sns.boxplot(x='floors', y='price', data=df, palette="Blues") 
plt.title('House Price Distribution by Number of Floors')
plt.xlabel('Number of Floors')
plt.ylabel('Price')
plt.show()

#Does the presence of a basement influence the house price?
df['has_basement'] = df['sqft_basement'].apply(lambda x: 'Yes' if x > 0 else 'No')

df.groupby('has_basement')['price'].mean().plot(kind='bar', figsize=(8,5), color=['skyblue'], edgecolor='black')
plt.title('Average House Price Based on Basement Presence')
plt.xlabel('Has Basement?')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.show()

#How does the price vary based on whether the house is near the waterfront?
sns.boxplot(x='waterfront', y='price', data=df, palette="Blues")
plt.title('Price Distribution by Waterfront Property', fontsize=14)
plt.xlabel('Waterfront', fontsize=12)
plt.ylabel('Price')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

#Model Development
#Building a model using some core features.
features =["bedrooms", "bathrooms", "waterfront", "sqft_living", "sqft_above", "sqft_basement", "floors"]

X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

#Adding 2 more feautures.
features =["bedrooms", "bathrooms", "waterfront", "sqft_living", "sqft_above", "sqft_basement", "floors", "sqft_above", "condition", "grade"]
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

#Adding locations features.
features =["bedrooms", "bathrooms","grade", "waterfront", "sqft_living", "sqft_basement", "floors", "sqft_above", "condition", "grade", "zipcode", "lat", "long"]
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

#Adding all the numeric features.
X = df.drop(['price', 'date', 'id'], axis=1)
X = pd.get_dummies(X, drop_first=True)
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)

#Model Evaluation and Refinement
#Spliting the data into training and testing sets.
X = X = df.drop(['price', 'date', 'id'], axis=1)
X = pd.get_dummies(X, drop_first=True)
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("Number of test samples:", x_test.shape[0])
print("Number of training samples:",x_train.shape[0])

#Creating and fitting a Ridge Regression model to the training data.
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat=RidgeModel.predict(x_test)
print(r2_score(y_test, yhat))

#Applying a second-order polynomial transform to both the training and testing data.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.transform(x_test)

RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr, y_train)
yhat=RidgeModel.predict(x_test_pr)
print(r2_score(y_test, yhat))
