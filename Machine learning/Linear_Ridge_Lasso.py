import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from matplotlib import pyplot as plt
#import warnings filter
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore')

#function to visualize the model with matplotlib
def plot_sklearn(model, x, y, test_x):
    plt.scatter(x, y)
    xx = model.predict(test_x)
    plt.plot(test_x, xx,"r.-", label="Linear", color = "r")
    plt.show()
"""
Linear regression model
"""
#read file from CSV and split data set 1
df = pd.read_csv("linear_regression_data.csv", encoding='utf-8')
print(df.head())
y = df.drop(columns=['Distance'])
x = df.drop(columns=['Delivery Time'])

from sklearn.model_selection import train_test_split
x_linear_train, x_linear_test, y_linear_train, y_linear_test = train_test_split(x, y, test_size=0.3, random_state=2)

#create Linear regression model
model_linear = LinearRegression(fit_intercept=True).fit(x_linear_train, y_linear_train)
print("<Linear Regression> coef:{0} intercept:{1}".format(model_linear.coef_, model_linear.intercept_))

#calculate the score of the Linear regression model.
from sklearn.model_selection import cross_val_score
MSE5 = cross_val_score(model_linear, x, y, scoring='neg_mean_squared_error', cv=5)
print("Linear Regression score: ", np.mean(MSE5))
print("\n")

#call Visualization functions for LinearRegression model
plot_sklearn(model_linear, x_linear_train, y_linear_train, x_linear_test)

"""
Ridge Regression model
"""
#read file from CSV and split data set 2
df = pd.read_csv("linear_regression_data.csv", encoding='utf-8')
y = df.drop(columns=['Distance'])
x = df.drop(columns=['Delivery Time'])

#create Ridge regression model
from sklearn.model_selection import GridSearchCV
ridge = Ridge()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 10, 20]}
ridge_regressor= GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(x, y)

#calculate the score of the Ridge regression model.
print("<Ridge> best parameter:{0} best score:{1}".format(ridge_regressor.best_params_, ridge_regressor.best_score_))
print("\n")

#call Visualization functions for Ridge Regression model
plot_sklearn(ridge_regressor, x, y, x)

"""
Lasso Regression model
"""
#create Lasso regression model
lasso = Lasso()
lasso_regressor= GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(x, y)

#calculate the score of the Lasso regression model.
print("<Lasso> best parameter:{0} best score:{1}".format(lasso_regressor.best_params_, lasso_regressor.best_score_))

#call Visualization functions for Lasso Regression model
plot_sklearn(lasso_regressor, x, y, x)






