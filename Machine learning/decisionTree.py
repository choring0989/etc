#import pandas and set options
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

#read data and describe
train = pd.read_csv("heart.csv")
print(pd.DataFrame(train).describe())

#split train data x and y
feature_names = ["age","sex", "cp", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
train_x = train[feature_names]
train_y = train["target"]

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
    
#make DecisionTree model
from sklearn.tree import DecisionTreeClassifier
seed = 37
criteria ='entropy'
#criteria ='gini'
model = DecisionTreeClassifier(criterion=criteria, max_depth=20, random_state=seed).fit(train_x,train_y)

#cross validation using k-fold
from sklearn.model_selection import cross_val_score
fold=10
scores = cross_val_score(model, train_x, train_y, cv=fold)
average_score = sum(scores)/fold

#average accuracy score visualization
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

xlabel = np.arange(10)
plt.title("DecisionTree Classifier Bar Chart")
plt.bar(np.arange(10), scores)
plt.xlabel("fold")
plt.ylabel("accuracy")
plt.show()

#check cross-validation average
print('criterion = {}, cross-validation average = {:.4}'.format(criteria, average_score))

#visualization with graphviz
from sklearn.tree import export_graphviz
import graphviz
export_graphviz(model,
                feature_names=feature_names,
                class_names=["1", "0"],
                out_file="decision-tree.dot")

with open("decision-tree.dot") as f:
    dot_graph = f.read()
    
graphviz.Source(dot_graph)

#visualization with seaborn
import seaborn as sns
matrix = pd.crosstab(train["target"], model.predict(train_x), rownames=['Actual'], colnames=['Predicted'], margins=True)

sns.heatmap(matrix, annot=True, fmt="d")
plt.title("Heatmap")
plt.show()




