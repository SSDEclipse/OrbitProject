import pandas
import matplotlib.pyplot as plt
import math
import pickle, os
import numpy as np
import csv
import random
from sklearn import linear_model, datasets



Distance = pickle.load(open(os.getcwd()+'/Distance', 'rb'))
firstcoefficient = pickle.load(open(os.getcwd()+'/firstcoefficient', 'rb'))
import pandas as pd
df3 = pd.DataFrame(list(Distance), columns=['d'])

clf2 = linear_model.LinearRegression()

x2 = df3[80:]
y2 = firstcoefficient[80:]
def Reverse(lst):
    return [ele for ele in reversed(lst)]
# Reverse(y2)
clf2.fit(x2, y2)
predict3 =[]
for i in range (80):
    predict3.append(Distance[i])
acts2 = firstcoefficient[0:80]
preds3 = []
for i in range(80):
    preds3.append(clf2.predict(Distance[i]))
def error (pred, act):
    return (pred-act)/act

errordiff2 = []
for i in range(80):
    errordiff2.append(error(preds3[i], acts2[i]))
print "average error is", sum(errordiff2)/len(errordiff2)





