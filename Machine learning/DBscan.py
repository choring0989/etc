#import models
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
#import warnings filter
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore')

# Do labelencoding
def LabelEncoding():
    global df
    df = df.apply(LabelEncoder().fit_transform)
    print("******************Label encoder******************")
    print(df.head())

# Scaling the data using StandardScaler
def SDscaler():
    global train_x
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    train_x = normalize(train_x)

    # Converting the numpy array into a pandas DataFrame
    train_x = pd.DataFrame(train_x)
    print("******************Standard Scaler******************")
    print(train_x.head())

# Scaling the data using StandardScaler
def MMscaler():
    global train_x
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    train_x = normalize(train_x)

    # Converting the numpy array into a pandas DataFrame
    train_x = pd.DataFrame(train_x)
    print("******************MinMax Scaler******************")
    print(train_x.head())

# Calculate purity using the cross tabulazation.
def purity(matrix):
    matrix = matrix.as_matrix()
    n = 0
    for i in range(len(matrix[0])):
        n = n + max(matrix[0][i], matrix[1][i]) # choose max
    n = n*1/matrix.sum()
    print("->purity: ", n, "*****\n")
    return n

# Function to draw a DBscan model, This function can represent up to six clusters,
# and the outliers are displayed in black.
def DBscanModel(data, y, eps, min_s, p):
    db = DBSCAN(eps=eps, min_samples=min_s, p=p).fit_predict(data)

    ct = pd.crosstab(y, db)
    print("******************crosstab******************")
    print(ct)
    print("eps={0} min_s={1} p={2}".format(eps, min_s, p))
    purity(ct)
    # 3D visualization
    from mpl_toolkits.mplot3d import Axes3D
    # scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(data['P1'], data['P2'], data['P3'], c=db, alpha=0.5)
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')
    plt.show()

'''
main
'''
# Read file from CSV
df = pd.read_csv("mushrooms.csv", encoding='utf-8')

# Find wrong data and replace it with NaN.
df = df.replace('?', np.nan)

# Find missing data and if it exist, fill median.
# If a missing value exists, it is replaced by the nearest (previous) valid value.
if (df.isnull().sum().any()):
    df = df.fillna(method='ffill') # fill previous
else: print("Missing data does not exist.")

LabelEncoding() # call LabelEncoding

# Separate target data.
train_y = df['class']
train_x = df.drop('class', axis=1)

# Do scaling
#SDscaler() # call StandardScaler
MMscaler() # call MinMaxScaler

# Reduce the dimensionality of the data to make it
# visualizable
pca= PCA(n_components=3)
X_principal= pca.fit_transform(train_x)
X_principal= pd.DataFrame(X_principal)
X_principal.columns= ['P1', 'P2', 'P3']

DBscanModel(X_principal, train_y ,0.02, 3, 3)# p = 3 & LabelEncoding & MinMaxScaler