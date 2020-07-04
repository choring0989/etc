import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#import K-means clustering module
from sklearn.cluster import KMeans

#reading data
train_url ="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url ="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#check data
print(train.head())
print(test.head())
print("\n")

#do preprocessing - Drop unneeded columns, Fill the missing data with the mean.
train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

print(train.isna().sum())
print(test.isna().sum())
print("\n")

#devide data to x, y
train_x = np.array(train.drop(['Survived'], 1))
train_y = np.array(train['Survived'])

#Clusters divide to survived and not survived
hyperP = 2
kmeans = KMeans(n_clusters=hyperP)

#Scale the data
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)

#fit
kmeans.fit(train_x)

KMeans(algorithm='auto', copy_x=True, init='k-means++',max_iter=100, n_clusters=hyperP, n_init=10, n_jobs=1,precompute_distances='auto', random_state=None,tol=0.0001, verbose=0)
#***other parameters***
#KMeans(algorithm='auto', copy_x=True, init='k-means++',max_iter=100, n_clusters=hyperP, n_init=10, n_jobs=1,precompute_distances='auto', random_state=None,tol=0.0001, verbose=0)
#KMeans(algorithm='auto', copy_x=True, init='k-means++',max_iter=300, n_clusters=hyperP, n_init=10, n_jobs=1,precompute_distances='auto', random_state=None,tol=0.0001, verbose=0)
#KMeans(algorithm='sparse', copy_x=True, init='k-means++',max_iter=300, n_clusters=hyperP, n_init=30, n_jobs=1,precompute_distances='auto', random_state=None,tol=0.0001, verbose=0)
#KMeans(algorithm='auto', copy_x=True, init='k-means++',max_iter=500, n_clusters=hyperP, n_init=10, n_jobs=1,precompute_distances='auto', random_state=None,tol=0.0001, verbose=0)
#KMeans(algorithm='sparse', copy_x=True, init='k-means++',max_iter=500, n_clusters=hyperP, n_init=30, n_jobs=1,precompute_distances='auto', random_state=None,tol=0.0001, verbose=0)
params = "[max_iter=500, n_init=30, algorithm='sparse']"

#run cluster
correct = 0
for i in range(len(train_x)):
    predict_me= np.array(train_x[i].astype(float))
    predict_me= predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == train_y[i]:
        correct+= 1
print("k=",hyperP, "   parameters:", params, "   result:", correct/len(train_x))







