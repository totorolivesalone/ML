import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data=pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\Q12\\homeprices.csv")
print(data)

reg=linear_model.LinearRegression()
reg.fit(data[['area']],data.price)
print("prediction price for area of 2500 sq meter::",reg.predict([[2500]]))
print("m value:: ",reg.coef_)
print("intercept c value:: ",reg.intercept_)

plt.xlabel('AREA')
plt.ylabel('PRICE')
plt.scatter(data.area,data.price,color="red",marker="*")
plt.plot(data.area,reg.predict(data[['area']]),color="blue")
plt.show()