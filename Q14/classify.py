import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\Q14\\iris.csv")
x=df.values[:,:-1]
y=df.values[:,-1]
dTree=DecisionTreeClassifier(criterion="entropy")
dTree.fit(x,y)
print("For Sepal length=5.5,Sepal width=3.2,Petal length=1.0,Petal Width=0.5,\n species is:",dTree.predict([[5.5,3.2,1.0,0.5]]))
