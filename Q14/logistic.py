import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\Q14\\diabetes.csv")
print("Dataset preview:\n",df.head())
x=df.values[:,:-1]
y=df.values[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
lModel=LogisticRegression(random_state=3)
lModel.fit(x_train,y_train)
predictions=lModel.predict(x_test)
print("Pregnancy:2,Glucose:147,Blood pressure:70, skin thickness:30, Insulin:0, BMI:27.8, Pedigree fn:0.325, Age:35")
print("Diabetes status:",lModel.predict([[2,147,70,30,0,27.8,0.325,35]]))
print("Confusion matrix:\n",confusion_matrix(y_test,predictions))
print("Accuracy:",accuracy_score(y_test,predictions))




