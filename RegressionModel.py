from sklearn import linear_model, datasets
import pandas as pd
import numpy as np
#digit dataset from sklearn
digits = datasets.load_digits()
print len(digits)
print digits
#create the LinearRegression model
clf = linear_model.LinearRegression()

#set training set
x, y = digits.data[:-1], digits.target[:-1]
print len(digits)
#train model
clf.fit(x, y)

#predict
y_pred = clf.predict([digits.data[-1]])
y_true = digits.target[-1]

print(y_pred)
print(y_true)

# data = datasets.load_boston() ## loads Boston dataset from datasets library
#
# # define the data/predictors as the pre-set feature names
# df = pd.DataFrame(data.data, columns=data.feature_names)
#
# # Put the target (housing value -- MEDV) in another DataFrame
# target = pd.DataFrame(data.target, columns=['MEDV'])
# X = df
# y = target['MEDV']
# lm = linear_model.LinearRegression()
# model = lm.fit(X,y)
# predictions = lm.predict(X)
# print(predictions)[0:5]