import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data=pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\Q12\\canada_per_capita_income.csv")
print(data)

reg=linear_model.LinearRegression()
reg.fit(data[['year']],data.percapita)
print("prediction percapita for 2022::",reg.predict([[2022]]))
print("prediction percapita for 2027::",reg.predict([[2027]]))
print("m value:: ",reg.coef_)
print("intercept c value:: ",reg.intercept_)

plt.xlabel('YEAR')
plt.ylabel('PERCAPITA')
plt.scatter(data.year,data.percapita,color="red",marker="*")
plt.plot(data.year,reg.predict(data[['year']]),color="blue")
plt.show()