import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,RepeatedKFold,KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

ds=pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\confusion\\diabetes.csv")
x=ds.values[:,:-1]
y=ds.values[:,-1]
print(x.shape)
print(y.shape)
#print(x)
#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3) #test size=20%

#KNN model

Knn_model=KNeighborsClassifier(n_neighbors=5)
Knn_model.fit(x_train,y_train)
predictions=Knn_model.predict(x_test)

acc=accuracy_score(y_test,predictions)
print("Acurracy score of KNN classification:",acc)
prec=precision_score(y_test,predictions)
print("Precision score of KNN classification:",prec)
rec=recall_score(y_test,predictions)
print("Recall score of KNN classification:",rec)
conmat=confusion_matrix(y_test,predictions)
print("Confusion matrix of KNN classification: \n",conmat)

#Decision Tree Model
DTclassifier=DecisionTreeClassifier()
DTclassifier.fit(x_train,y_train)
predictions=DTclassifier.predict(x_test)

acc=accuracy_score(y_test,predictions)
print("Acurracy for DT classification:",acc)
prec=precision_score(y_test,predictions)
print("Precision score of DT classification:",prec)
rec=recall_score(y_test,predictions)
print("Recall score of DT classification:",rec)
conmat=confusion_matrix(y_test,predictions)
print("Confusion matrix of DT classification: \n",conmat)