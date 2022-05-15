import pandas as pd
import numpy as np
import sklearn
from numpy import ndarray
from sklearn import linear_model
from sklearn.utils import shuffle
import csv

data = pd.read_csv("MOCK_DATA.csv", sep=",")

data = data[["Temperature", "Predict Sucess"]]
predict = "Predict Sucess"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.7)

linear_data = linear_model.LinearRegression()

linear_data.fit(x_train, y_train)
acc = linear_data.score(x_test, y_test)
print(acc)

prediction = linear_data.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x])

df = pd.DataFrame(prediction, columns=['Predicts Temperature'])
df.to_csv('test.csv')
