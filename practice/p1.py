from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
data=datasets.load_breast_cancer()
print(data)
df=pd.DataFrame(data.data,columns=data.feature_names)

df["target"]=data.target
print(df.head())



