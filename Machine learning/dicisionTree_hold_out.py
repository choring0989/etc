import pandas as pd
import numpy as np
from pprint import pprint
import random as rd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

"""
**read file**
"""
data = pd.read_csv('decision_tree_data.csv', encoding='utf-8')

print(data)
print("\n")
features = data[["level", "lang", "tweets", "phd"]]
#target feature
target = data["interview"]

"""
**Entropy**

-def entropy(target_col): Function to obtain entropy value.
-target_col: Data to be located at the leaf node.
-return: float
"""
def entropy(target_col):
    element, count = np.unique(target_col, return_counts = True)
    print("*unique element: ", element.astype(np.bool),":", count)
    entropy = -np.sum([(count[i]/np.sum(count))
                       *np.log2(count[i]/np.sum(count))
                       for i in range(len(element))])
    return entropy

"""
**Information gain**

-def InfoGain(data,split_attribute_name,target_name):
Method for obtaining information gain
using difference of entropy value.
-data: All data set
-split_attribute_name:
The attribute name of the node for
which to obtain the information gain.
-target_name:
The name of the attribute that you want to specify as the leaf node.
-return: float
"""
def InfoGain(data,split_attribute_name,target_name):
 
    # Total entropy calculation
    total_entropy = entropy(data[target_name])
    print('\nTotal Entropy = ', round(total_entropy, 5))
    
    # weighted entropy calculation
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*
                               entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('Entropy of "', split_attribute_name, '" =', round(Weighted_Entropy, 5))
    
    # Infomation Gain calculation
    Information_Gain = total_entropy - Weighted_Entropy
    print('InfoGain(', split_attribute_name ,') = ',round(Information_Gain,5), '\n')
    return Information_Gain

"""
**ID3 Tree Generation Algorithm**
An algorithm that actually creates a tree.
-return: tree(Dictionary)
"""
def ID3(data,originaldata,features,target_attribute_name,parent_node_class = None):

    # Stopping criterion of the algorithm
 
    # 1. If all the values of the attributes are equal
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
 
    # 2. If there is no data, return the maximum value of the attributes
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])\
               [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
 
    # 3. When there is no feature, return parent node
    elif len(features) ==0:
        return parent_node_class
 
    # If it keep creating the tree
    else:
        # Define the target attributes of the parent node
        parent_node_class = np.unique(data[target_attribute_name])\
                            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Select attributes to split the data
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        print("----->>>Select [", best_feature, "]<<<-----\n\n")
        
        # Create tree structure
        tree = {best_feature:{}}
        
        # Excluding technical attributes
        # that exhibit the maximum information gain
        features = [i for i in features if i != best_feature]
        

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree

        return(tree)

testDataSelect = []
"""
#preprocessing
data["level"] = data["level"].replace("senior", int(3))
data["level"] = data["level"].replace("mid", int(2))
data["level"] = data["level"].replace("junior", int(1))

data["lang"] = data["lang"].replace("java", int(3))
data["lang"] = data["lang"].replace("python", int(2))
data["lang"] = data["lang"].replace("R", int(1))

data["tweets"] = data["tweets"].replace("yes", int(1))
data["tweets"] = data["tweets"].replace("no", int(0))

data["phd"] = data["phd"].replace("yes", int(1))
data["phd"] = data["phd"].replace("no", int(0))

data["interview"] = data["interview"].replace("TRUE", int(1))
data["interview"] = data["interview"].replace("FALSE", int(0))
"""

"""
**Holdout method**
Divides the data into test data set and training data set.
Use the holdout method.
9/10 is the training data set, and 1/10 is the test data set.
"""
for i in range(int(len(data)/10)):
    selectNum = rd.randint(0, len(data))
    if selectNum in testDataSelect and (i != 0):
        i = i-1
    else:
        testDataSelect.append(selectNum)

testDataSet = pd.DataFrame(columns=("level", "lang", "tweets", "phd", "interview"))
trainingDataSet = pd.DataFrame(columns=("level", "lang", "tweets", "phd", "interview"))
k=0
z=0
for i in range(len(data)):
    if i in testDataSelect:
        testDataSet.loc[k] = data.loc[i]
        k=k+1
    else:
        trainingDataSet.loc[z] = data.loc[i]
        z=z+1

print("-->testDataSet<--")
print(testDataSet)
print("-->trainingDataSet<--")
print(trainingDataSet)

#run ID3 algorithm
tree = ID3(trainingDataSet, trainingDataSet, ["level", "lang", "tweets", "phd"], "interview")
print("**Outcom of Tree with trainingDataSet**\n") 
pprint(tree)
print("\n\n")

tree_test = ID3(testDataSet, testDataSet, ["level", "lang", "tweets", "phd"], "interview")
print("**Outcom of Tree with testDataSet**\n") 
pprint(tree_test)
print("\n\n")

X_train =trainingDataSet[["level", "lang", "tweets", "phd"]]
X_test =testDataSet[["level", "lang", "tweets", "phd"]]
y_train =trainingDataSet[["interview"]]
y_test = testDataSet[["interview"]]

ml = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)

ml.fit(X_train, y_train)
y_pred = ml.predict(X_test)
print('all test cases:%d, error_num:%d' %(len(y_test), (y_test !=y_pred).sum()))
print('accuracy: %.2f' %accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=X_combined, y=y_combined, classifier=ml,
                     test_idx=range(105, 150))


