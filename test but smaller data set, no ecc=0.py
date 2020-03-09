
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle, os
import numpy as np
import csv
import random
from sklearn import linear_model, datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

data = pandas.read_csv('stuffbutonly300.csv')
print(data.shape)
import pandas as pd
# df1 = pd.DataFrame(data, columns = columnNames)

eccentricity = list(data['pl_orbeccen'])
eccentricityerrorhigh = list(data['pl_orbeccenerr1'])
eccentricityerrorlow = list(data['pl_orbeccenerr2'])
del eccentricity[0]
del eccentricityerrorhigh[0]
del eccentricityerrorlow[0]

eccentricity = [float(c) for c in eccentricity]
eccentricityerrorhigh = [float(c) for c in eccentricityerrorhigh]
eccentricityerrorlow = [float(c) for c in eccentricityerrorlow]
eccentricityhigh = [eccentricity[i]+eccentricityerrorhigh[i] for i in range(len(eccentricity))]
eccentricitylow = [eccentricity[i]+eccentricityerrorlow[i] for i in range(len(eccentricity))]
# print (eccentricityerrorlow)
# print (eccentricityerrorhigh)
# print eccentricityhigh
# print (eccentricity)
# print eccentricitylow


PlanetRadius = list(data['pl_radj'])
StarMass = list(data['st_mass'])
Distance = list(data['pl_orbsmax'])
PlanetNumber = list(data['pl_pnum'])
EffTemp = list(data['st_teff'])
OrbPer = list(data['pl_orbper'])

del PlanetRadius[0]
del StarMass[0]
del Distance[0]
del PlanetNumber[0]
del EffTemp[0]
del OrbPer[0]
del OrbPer[-3]
PlanetRadius = [float(c) for c in PlanetRadius]
StarMass = [float(c) for c in StarMass]
Distance = [float(c) for c in Distance]
PlanetNumber = [float(c) for c in PlanetNumber]
EffTemp = [float(c) for c in EffTemp]
OrbPer = [float(c) for c in OrbPer]

PlanetRadius = [7.1492*10**7*i for i in PlanetRadius]
StarMass = [1.989*10**30*i for i in StarMass]
Distance = [149597870691.*i for i in Distance]
PlanetRadiusErrorHigh = list(data['pl_radjerr1'])
PlanetRadiusErrorLow = list(data['pl_radjerr2'])
DistanceErrorHigh = list(data['pl_orbsmaxerr1'])
DistanceErrorLow = list(data['pl_orbsmaxerr2'])
StarMassErrorHigh = list(data['st_masserr1'])
StarMassErrorLow = list(data['st_masserr2'])
del PlanetRadiusErrorHigh[0]
del PlanetRadiusErrorLow[0]
del DistanceErrorHigh[0]
del DistanceErrorLow[0]
del StarMassErrorHigh[0]
del StarMassErrorLow[0]
PlanetRadiusErrorHigh2 = [float (c) for c in PlanetRadiusErrorHigh]
PlanetRadiusErrorLow2 = [float (c) for c in PlanetRadiusErrorLow]
DistanceErrorHigh2 = [float (c) for c in DistanceErrorHigh]
DistanceErrorLow2 = [float (c) for c in DistanceErrorLow]
StarMassErrorHigh2 = [float (c) for c in StarMassErrorHigh]
StarMassErrorLow2 = [float (c) for c in StarMassErrorLow]


PlanetRadiusErrorHigh =[]
PlanetRadiusErrorLow =[]
StarMassErrorHigh =[]
StarMassErrorLow =[]
DistanceErrorHigh =[]
DistanceErrorLow =[]
for i in range(len(PlanetRadiusErrorHigh2)):
    PlanetRadiusErrorHigh.append(7.1492*10**7*PlanetRadiusErrorHigh2[i])
for i in range(len(PlanetRadiusErrorLow2)):
    PlanetRadiusErrorLow.append(7.1492*10**7*PlanetRadiusErrorLow2[i])
for i in range(len(StarMassErrorHigh2)):
    StarMassErrorHigh.append(1.989*10**30*StarMassErrorHigh2[i])
for i in range(len(StarMassErrorLow2)):
    StarMassErrorLow.append(1.989*10**30*StarMassErrorLow2[i])
for i in range(len(DistanceErrorHigh2)):
    DistanceErrorHigh.append(149597870691*DistanceErrorHigh2[i])
for i in range(len(DistanceErrorLow2)):
    DistanceErrorLow.append(149597870691*DistanceErrorLow2[i])
PlanetRadiusHigh = []
PlanetRadiusLow= []
for i in range(len(PlanetRadius)):
    PlanetRadiusHigh.append(PlanetRadius[i]+PlanetRadiusErrorHigh[i])
for i in range(len(PlanetRadius)):
    PlanetRadiusLow.append(PlanetRadius[i]+PlanetRadiusErrorLow[i])


StarMassHigh= []
StarMassLow= []
for i in range(len(StarMass)):
    StarMassHigh.append(StarMass[i]+StarMassErrorHigh[i])
for i in range(len(StarMass)):
    StarMassLow.append(StarMass[i]+StarMassErrorLow[i])
DistanceHigh= []
DistanceLow= []
for i in range(len(Distance)):
    DistanceHigh.append(Distance[i]+DistanceErrorHigh[i])
for i in range(len(Distance)):
    DistanceLow.append(Distance[i]+DistanceErrorLow[i])

def Energy (M, m, d):
    return (6.674*10**-11)*(-1.)*M*m*(1./(2.*d))
highenergy =[Energy(StarMassHigh[i], PlanetRadiusHigh[i], DistanceLow[i]) for i in range(len(PlanetRadius))]
lowenergy =[Energy(StarMassLow[i], PlanetRadiusLow[i], DistanceHigh[i]) for i in range(len(PlanetRadius))]
midenergy =[Energy(StarMass[i], PlanetRadius[i], Distance[i]) for i in range(len(PlanetRadius))]

def AngularMomentum (m,d,M):
    return (m*d*(6.674*10**-11*M*(1/d))**0.5)
highmomentum = [AngularMomentum(PlanetRadiusHigh[i], DistanceHigh[i], StarMassHigh[i]) for i in range(len(PlanetRadius))]
lowmomentum = [AngularMomentum(PlanetRadiusLow[i], DistanceLow[i], StarMassLow[i]) for i in range(len(PlanetRadius))]
midmomentum = [AngularMomentum(PlanetRadius[i], Distance[i], StarMass[i]) for i in range(len(PlanetRadius))]

def SGP (M):
    return -6.674*10**-11*M
StandardGPhigh = [SGP(StarMassHigh[i]) for i in range(len(StarMass))]
StandardGPlow = [SGP(StarMassLow[i]) for i in range(len(StarMass))]
StandardGPmid = [SGP(StarMass[i]) for i in range(len(StarMass))]

PlanetRadiusLow1 = []
for i in range(len(PlanetRadiusLow)):
    if PlanetRadiusLow[i]!=0:
        PlanetRadiusLow1.append(PlanetRadiusLow[i])
    else:
        print i
# PlanetRadiusLow1 = PlanetRadiusLow
# StandardGPlow1 =[]
# for i in range(len(StandardGPlow)):
#     if StandardGPlow[i]!=0:
#         StandardGPlow1.append(StandardGPlow[i])
# print len(StandardGPlow1)
# StandardGPlow1 = StandardGPlow

def eccentricitynoroot (Energy, l, m, u):
    return (1+(2*Energy*(l**2))/((m**3)*(u**2)))

higheccentricity = [eccentricitynoroot(lowenergy[i], lowmomentum[i], PlanetRadiusHigh[i], StandardGPhigh[i]) for i in range(len(PlanetRadiusHigh))]
mideccentricity = [eccentricitynoroot(midenergy[i], midmomentum[i], PlanetRadius[i], StandardGPmid[i]) for i in range(len(PlanetRadiusHigh))]
loweccentricity = [eccentricitynoroot(highenergy[i], highmomentum[i], PlanetRadiusLow1[i], StandardGPlow[i]) for i in range(len(PlanetRadiusLow1))]
higheccentricitynozeroes = []
for i in range(len(PlanetRadius)):
    if higheccentricity[i]>0:
        higheccentricitynozeroes.append((higheccentricity[i])**0.5)

mideccentricitynozeroes = []
for i in range(len(PlanetRadius)):
    if mideccentricity[i] > 0:
            mideccentricitynozeroes.append((mideccentricity[i]) ** 0.5)
# loweccentricitynozeroes = []
# for i in range(len(PlanetRadiusLow1)):
#     if loweccentricity[i]>0:
#         loweccentricitynozeroes.append((loweccentricity[i])**0.5)
# print len(loweccentricitynozeroes)

eccentricitysquared = []
for i in range(len(eccentricity)):
    eccentricitysquared.append(eccentricity[i]**2)




energyerrors = []
for i in range(len(midenergy)):
    energyerrors.append(math.fabs(highenergy[i]-midenergy[i]))
momentumerrors = []
for i in range(len(midmomentum)):
    momentumerrors.append(math.fabs(highmomentum[i]-midmomentum[i]))
StandardGPerrors = []
for i in range(len(StandardGPmid)):
    StandardGPerrors.append(math.fabs(StandardGPhigh[i]-StandardGPmid[i]))
PlanetRadiuserrors = []
for i in range(len(PlanetRadius)):
    PlanetRadiuserrors.append(math.fabs(PlanetRadiusHigh[i]-PlanetRadius[i]))
print energyerrors
testeccentricity = [eccentricitynoroot(midenergy[i]+0.0005*energyerrors[i], midmomentum[i]-0.00002*momentumerrors[i], PlanetRadius[i]+0.09*PlanetRadiuserrors[i], StandardGPmid[i]+0.10000*StandardGPerrors[i]) for i in range(len(PlanetRadiusHigh))]

eccentricitydifference = []
for i in range(len(eccentricitysquared)):
    eccentricitydifference.append(testeccentricity[i]-eccentricitysquared[i])

#digit dataset from sklearn
digits = datasets.load_digits()

#create the LinearRegression model
clf = linear_model.LinearRegression()
#set training set
x, y = digits.data[:-1], digits.target[:-1]
test = zip(eccentricity, PlanetRadius)

#train model
# clf.fit(x, y)

#predict
# y_pred = clf.predict([digits.data[-1]])
# y_true = digits.target[-1]

# print(y_pred)
# print(y_true)
x_1, y_1 = PlanetRadius, eccentricity
#
# print len(x_1)
# print len(y_1)
# clf.fit(x_1, y_1)
import pandas as pd
del PlanetRadius[-3]
del Distance[-3]
del PlanetNumber[-3]
del eccentricity[-3]

df = pd.DataFrame(list(zip(Distance, PlanetNumber, PlanetRadius,)), columns =['d', 'smass', 'rad'])
# # print df
# df = df1[['pl_orbsmax', 'pl_massj']]
# df1.drop(df1.index[0])

x=df[:-60]

y=eccentricity[:-60]
poly = PolynomialFeatures(degree=1)
X_ = poly.fit_transform(x)
clf = linear_model.LinearRegression()
clf.fit(X_, y)
predict = [PlanetRadius[-27], Distance[-27], PlanetRadius[-27],]
predict_ = poly.fit_transform(predict)

def error (pred, act):
    return math.fabs(pred-act)
predict2 =[]
for i in range (60):
    predict2.append([Distance[-i], PlanetNumber[-i], PlanetRadius[-i]])

predict2_ = poly.fit_transform(predict2)
preds2 = []
for i in range(60):
    preds2.append(clf.predict(predict2_[i]))
acts2 = eccentricity[-60:]
del preds2[3]
del acts2[3]
errordiff = []
for i in range(59):
    errordiff.append(error(preds2[i], acts2[i]))
df2 = pd.DataFrame(list(zip(preds2, acts2)),
               columns =['pred', 'act'])
# predictreal = [7.57*10**7, 778*10**9]
# predictreal_ = poly.fit_transform(predictreal)
# print clf.predict(predictreal_)
print 'the error is', (sum(errordiff))/60
