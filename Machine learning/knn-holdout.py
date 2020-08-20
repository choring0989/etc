#import pandas
import pandas as pd

#read data using pandas
data_frame = pd.read_excel('DS-minilab-2-dataset_original.xlsx')

#check data
print(data_frame.head())

#separate target data
data = data_frame.drop(columns=['Index', 'Gender'])#'Gender'column is not necessary.
target = data_frame['Index']

###  Holdout method  ###
#import train_test_split
from sklearn.model_selection import train_test_split

#split data
x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.2, random_state=1)

#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#create a KNeighborsClassifier -> k=5
knn_h = KNeighborsClassifier(n_neighbors = 5)

#train the KNeighborsClassifier
knn_h.fit(x_train,y_train)

#show predicted model
print(knn_h.predict(x_test))

#check accuracy of the model on the test data
print("Holdout method score: ", knn_h.score(x_test, y_test))


###  k-fold method  ###
#create a new KNeighborsClassifier
knn_f = KNeighborsClassifier(n_neighbors = 5)

#import cross_val_score
from sklearn.model_selection import cross_val_score

# train model with cv of 5 (5' fold)
cv_scores = cross_val_score(knn_f, data, target, cv=5)

# print each cv score (accuracy) and average them
import numpy as np
print("k-fold method score: ", cv_scores)
print('k-fold method mean score:', np.mean(cv_scores))
