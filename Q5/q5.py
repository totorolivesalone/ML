import pandas as pd

df = pd.read_csv('C:\\Users\\unkno\\Documents\\ML\\Q5\\data.csv')
print(df)
print("Drop 5 rows from tail::\n")
df.drop(df.tail(5).index,inplace=True) 
print(df)
df.to_csv('output.txt', sep=' ') ###fsaving data to txt file
print("Feature names::\n")
print(df.columns.values)