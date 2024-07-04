import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# Reading csv file to dataframe
df = pd.read_csv('C:\\Users\\unkno\\Documents\\ML\\Q13\\homeprices.csv')
df
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median()) ###fill na with median, this step is important for multivariate
reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'), df.price)

m1, m2, m3 = reg.coef_
c = reg.intercept_
print('Coefficients, \
\n\tm1 = {}, \
\n\tm2 = {}, \
\n\tm3 = {}'.format(m1, m2, m3))
print('Intercept, c = ', c)
y1 = m1*3000 + m2*3 + m3*40 + c
print('\ty1 = m1*x1 + m2*x2 + m3*x3 + c =\n\t', y1)
ans1 = reg.predict([[3000, 3, 40]])
print('(1) Price of home with 3000 sqr ft area, 3 bedrooms, 40 year old: ', ans1)

