from numpy import *
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("C:\\Users\\unkno\\Documents\\ML\\Q16\\iris.csv")
print(df.head())
print(df.describe().transpose())
# Define the target column and predictors
target_column = ['Species']
predictors = ['Sepal Length', 'Sepal width', 'Petal Length', 'Petal Width']

# Normalize the predictor columns
df[predictors] = df[predictors] / df[predictors].max()

# Creating the training and test datasets
X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

# Print the shapes of the training and test datasets
print(X_train.shape)
print(X_test.shape)

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 8), activation='relu', solver='sgd', max_iter=500)
mlp.fit(X_train, y_train)

# Make predictions on the training data
predict_train = mlp.predict(X_train)

# Make predictions on the test data
predict_test = mlp.predict(X_test)

# Evaluate the performance of the model on the training data
print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

# Evaluate the performance of the model on the test data
print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))
