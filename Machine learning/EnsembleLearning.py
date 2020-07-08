#import warnings filter
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

#read .mat data set
from scipy.io import loadmat
mnist = loadmat('./mnist-original.mat')
x = mnist["data"].T
y = mnist["label"][0].reshape(-1, 1)

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

#convert to a dataframe and then describe the data
x_df = pd.DataFrame(x)
y_df = pd.DataFrame(y)
print(x_df.describe())
print(y_df.describe())

#data Distribution
import seaborn as sns
import matplotlib.pylab as plt
sns.distplot(y_df, kde=True, rug=True)
plt.show()

#divide data set (garbage is dropped)
from sklearn.model_selection import train_test_split
dfx_garbage, dfx_train, dfy_garbage, dfy_train = train_test_split(x_df, y_df, test_size=0.1, random_state=1)

dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(dfx_train, dfy_train, test_size=0.33, random_state=1)
dfx_ensemble_test, dfx_train, dfy_ensemble_test, dfy_train = train_test_split(dfx_train, dfy_train, test_size=0.5, random_state=1)

#make Logistic Regression model
from sklearn.linear_model import LogisticRegression
max_ = 30
LR_model = LogisticRegression(solver='liblinear', max_iter=max_).fit(dfx_train, dfy_train.values.ravel())

#accuracy calculation
from sklearn.metrics import accuracy_score
y_pred_LR = LR_model.predict(dfx_test)
print("Logistic Regression model's best accuracy", accuracy_score(dfy_test, y_pred_LR))

#visualization with seaborn
matrix1 = pd.crosstab(dfy_test.values.ravel(), y_pred_LR, rownames=['Actual'], colnames=['Predicted'], margins=True)
sns.heatmap(matrix1, annot=True, fmt="d")
plt.title("Heatmap")
plt.show()

#make Support Vector Machine model (linear)
from sklearn.svm import SVC
svm_linear_model = SVC(C=1.0, kernel='linear', gamma=10, probability=True)
svm_linear_model.fit(dfx_train, dfy_train.values.ravel())

#accuracy calculation
y_pred_SL = svm_linear_model.predict(dfx_test)
print("Support Vector Machine model(linear)'s best accuracy", accuracy_score(dfy_test, y_pred_SL))

#visualization with seaborn
matrix2 = pd.crosstab(dfy_test.values.ravel(), y_pred_SL, rownames=['Actual'], colnames=['Predicted'], margins=True)
sns.heatmap(matrix2, annot=True, fmt="d")
plt.title("Heatmap")
plt.show()

#make Support Vector Machine model (linear)
from sklearn.svm import SVC
svm_rbf_model = SVC(C=1.0, kernel='rbf', gamma=10, probability=True)
svm_rbf_model.fit(dfx_train, dfy_train.values.ravel())

#accuracy calculation
y_pred_SR = svm_rbf_model.predict(dfx_test)
print("Support Vector Machine model(rbf)'s best accuracy", accuracy_score(dfy_test, y_pred_SR))

#visualization with seaborn
matrix3 = pd.crosstab(dfy_test.values.ravel(), y_pred_SR, rownames=['Actual'], colnames=['Predicted'], margins=True)
sns.heatmap(matrix3, annot=True, fmt="d")
plt.title("Heatmap")
plt.show()

#run ensemble running with Majority Voting.
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[('LR', LR_model), ('SVML', svm_linear_model), ('SVMR', svm_rbf_model)], voting='soft')
ensemble.fit(dfx_train, dfy_train.values.ravel())

for clf in (LR_model, svm_linear_model, svm_rbf_model, ensemble):
    clf.fit(dfx_train, dfy_train.values.ravel())
    y_pred = clf.predict(dfx_ensemble_test)
    print(clf.__class__.__name__, accuracy_score(dfy_ensemble_test, y_pred))
    if (clf.__class__.__name__ == 'VotingClassifier'):
        #visualization with seaborn
        matrix4 = pd.crosstab(dfy_ensemble_test.values.ravel(), clf, rownames=['Actual'], colnames=['Predicted'], margins=True)
        sns.heatmap(matrix4, annot=True, fmt="d")
        plt.title("Heatmap")
        plt.show()
        








