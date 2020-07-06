import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
#import warnings filter
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore')

#read file from CSV
df = pd.read_csv("Advertising.csv", encoding='utf-8')
print(df.head())
x = df.drop(columns=['Unnamed: 0','sales'])
y = df.drop(columns=['Unnamed: 0', 'TV', 'radio', 'newspaper'])
"""
Elastic Net regression
"""
EN = ElasticNet()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 10, 20]}
EN_regressor= GridSearchCV(EN, parameters, scoring='neg_mean_squared_error', cv=5)
EN_regressor.fit(x, y)

print("<ElasticNet> best parameter:{0} best score:{1}".format(EN_regressor.best_params_, EN_regressor.best_score_))
print("\n")

"""
Ridge Regression model
"""
#create Ridge regression model
ridge = Ridge()
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 10, 20]}
ridge_regressor= GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(x, y)

#calculate the score of the Ridge regression model.
print("<Ridge> best parameter:{0} best score:{1}".format(ridge_regressor.best_params_, ridge_regressor.best_score_))
print("\n")

"""
Lasso Regression model
"""
#create Lasso regression model
lasso = Lasso()
lasso_regressor= GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(x, y)

#calculate the score of the Lasso regression model.
print("<Lasso> best parameter:{0} best score:{1}".format(lasso_regressor.best_params_, lasso_regressor.best_score_))
