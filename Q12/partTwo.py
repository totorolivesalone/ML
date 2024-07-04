import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data=pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\Q12\\MBA.csv")
print(data)

reg=linear_model.LinearRegression()
reg.fit(data[['Percentage']],data.Salary)
print("m value:: ",reg.coef_)
print("intercept c value:: ",reg.intercept_)

plt.xlabel('Percentage')
plt.ylabel('Salary')
plt.scatter(data.Percentage,data.Salary,color="red",marker="+")
plt.plot(data.Percentage,reg.predict(data[['Percentage']]),color="blue")
plt.show()
