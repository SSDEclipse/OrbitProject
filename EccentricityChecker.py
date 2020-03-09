
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

data = pandas.read_csv("Planets640FewColumns2-EccentricityNoZeroes.csv")
print(data.shape)
import pandas as pd

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



PlanetMass = list(data['pl_masse'])
StarMass = list(data['st_mass'])
Distance = list(data['pl_orbsmax'])
del PlanetMass[0]
del StarMass[0]
del Distance[0]
PlanetMass = [float(c) for c in PlanetMass]
StarMass = [float(c) for c in StarMass]
Distance = [float(c) for c in Distance]
PlanetMass = [5.976*10**24*i for i in PlanetMass]
StarMass = [1.989*10**30*i for i in StarMass]
Distance = [149597870691.*i for i in Distance]
PlanetMassErrorHigh = list(data['pl_masseerr1'])
PlanetMassErrorLow = list(data['pl_masseerr2'])
DistanceErrorHigh = list(data['pl_orbsmaxerr1'])
DistanceErrorLow = list(data['pl_orbsmaxerr2'])
StarMassErrorHigh = list(data['st_masserr1'])
StarMassErrorLow = list(data['st_masserr2'])
del PlanetMassErrorHigh[0]
del PlanetMassErrorLow[0]
del DistanceErrorHigh[0]
del DistanceErrorLow[0]
del StarMassErrorHigh[0]
del StarMassErrorLow[0]
PlanetMassErrorHigh2 = [float (c) for c in PlanetMassErrorHigh]
PlanetMassErrorLow2 = [float (c) for c in PlanetMassErrorLow]
DistanceErrorHigh2 = [float (c) for c in DistanceErrorHigh]
DistanceErrorLow2 = [float (c) for c in DistanceErrorLow]
StarMassErrorHigh2 = [float (c) for c in StarMassErrorHigh]
StarMassErrorLow2 = [float (c) for c in StarMassErrorLow]


PlanetMassErrorHigh =[]
PlanetMassErrorLow =[]
StarMassErrorHigh =[]
StarMassErrorLow =[]
DistanceErrorHigh =[]
DistanceErrorLow =[]
for i in range(len(PlanetMassErrorHigh2)):
    PlanetMassErrorHigh.append(5.976*10**24*PlanetMassErrorHigh2[i])
for i in range(len(PlanetMassErrorLow2)):
    PlanetMassErrorLow.append(5.976*10**24*PlanetMassErrorLow2[i])
for i in range(len(StarMassErrorHigh2)):
    StarMassErrorHigh.append(1.989*10**30*StarMassErrorHigh2[i])
for i in range(len(StarMassErrorLow2)):
    StarMassErrorLow.append(1.989*10**30*StarMassErrorLow2[i])
for i in range(len(DistanceErrorHigh2)):
    DistanceErrorHigh.append(149597870691*DistanceErrorHigh2[i])
for i in range(len(DistanceErrorLow2)):
    DistanceErrorLow.append(149597870691*DistanceErrorLow2[i])
PlanetMassHigh = []
PlanetMassLow= []
for i in range(len(PlanetMass)):
    PlanetMassHigh.append(PlanetMass[i]+PlanetMassErrorHigh[i])
for i in range(len(PlanetMass)):
    PlanetMassLow.append(PlanetMass[i]+PlanetMassErrorLow[i])


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
highenergy =[Energy(StarMassHigh[i], PlanetMassHigh[i], DistanceLow[i]) for i in range(len(PlanetMass))]
lowenergy =[Energy(StarMassLow[i], PlanetMassLow[i], DistanceHigh[i]) for i in range(len(PlanetMass))]
midenergy =[Energy(StarMass[i], PlanetMass[i], Distance[i]) for i in range(len(PlanetMass))]

def AngularMomentum (m,d,M):
    return (m*d*(6.674*10**-11*M*(1/d))**0.5)
highmomentum = [AngularMomentum(PlanetMassHigh[i], DistanceHigh[i], StarMassHigh[i]) for i in range(len(PlanetMass))]
lowmomentum = [AngularMomentum(PlanetMassLow[i], DistanceLow[i], StarMassLow[i]) for i in range(len(PlanetMass))]
midmomentum = [AngularMomentum(PlanetMass[i], Distance[i], StarMass[i]) for i in range(len(PlanetMass))]

def SGP (M):
    return -6.674*10**-11*M
StandardGPhigh = [SGP(StarMassHigh[i]) for i in range(len(StarMass))]
StandardGPlow = [SGP(StarMassLow[i]) for i in range(len(StarMass))]
StandardGPmid = [SGP(StarMass[i]) for i in range(len(StarMass))]

PlanetMassLow1 = []
for i in range(len(PlanetMassLow)):
    if PlanetMassLow[i]!=0:
        PlanetMassLow1.append(PlanetMassLow[i])
    else:
        print i
# PlanetMassLow1 = PlanetMassLow
# StandardGPlow1 =[]
# for i in range(len(StandardGPlow)):
#     if StandardGPlow[i]!=0:
#         StandardGPlow1.append(StandardGPlow[i])
# print len(StandardGPlow1)
# StandardGPlow1 = StandardGPlow
def coefficient (l, m, SGP):
    return  (l**2)/(m**2*SGP)
firstcoefficient =[]
for i in range(len(Distance)):
    firstcoefficient.append(coefficient(midmomentum[i], PlanetMass[i], StandardGPmid[i]))

def eccentricitynoroot (Energy, l, m, u):
    return (1+(2*Energy*(l**2))/((m**3)*(u**2)))

higheccentricity = [eccentricitynoroot(lowenergy[i], lowmomentum[i], PlanetMassHigh[i], StandardGPhigh[i]) for i in range(len(PlanetMassHigh))]
mideccentricity = [eccentricitynoroot(midenergy[i], midmomentum[i], PlanetMass[i], StandardGPmid[i]) for i in range(len(PlanetMassHigh))]
loweccentricity = [eccentricitynoroot(highenergy[i], highmomentum[i], PlanetMassLow1[i], StandardGPlow[i]) for i in range(len(PlanetMassLow1))]
higheccentricitynozeroes = []
for i in range(len(PlanetMass)):
    if higheccentricity[i]>0:
        higheccentricitynozeroes.append((higheccentricity[i])**0.5)

mideccentricitynozeroes = []
for i in range(len(PlanetMass)):
    if mideccentricity[i] > 0:
            mideccentricitynozeroes.append((mideccentricity[i]) ** 0.5)


eccentricitysquared = []
for i in range(len(eccentricity)):
    eccentricitysquared.append(eccentricity[i]**2)

# for i in range(len(PlanetMassErrorHigh)):
#     PlanetMassErrorHigh[i]>i>PlanetMassErrorLow[i]



energyerrors = []
for i in range(len(midenergy)):
    energyerrors.append(math.fabs(highenergy[i]-midenergy[i]))
momentumerrors = []
for i in range(len(midmomentum)):
    momentumerrors.append(math.fabs(highmomentum[i]-midmomentum[i]))
StandardGPerrors = []
for i in range(len(StandardGPmid)):
    StandardGPerrors.append(math.fabs(StandardGPhigh[i]-StandardGPmid[i]))
planetmasserrors = []
for i in range(len(PlanetMass)):
    planetmasserrors.append(math.fabs(PlanetMassHigh[i]-PlanetMass[i]))
print energyerrors
testeccentricity = [eccentricitynoroot(midenergy[i]+0.0005*energyerrors[i], midmomentum[i]-0.00002*momentumerrors[i], PlanetMass[i]+0.09*planetmasserrors[i], StandardGPmid[i]+0.10000*StandardGPerrors[i]) for i in range(len(PlanetMassHigh))]
eccentricitydifference = []
for i in range(len(eccentricitysquared)):
    eccentricitydifference.append(testeccentricity[i]-eccentricitysquared[i])

#digit dataset from sklearn
digits = datasets.load_digits()

#create the LinearRegression model
clf = linear_model.LinearRegression()
#set training set
x, y = digits.data[:-1], digits.target[:-1]
test = zip(eccentricity, PlanetMass)


#train model
# clf.fit(x, y)

#predict
# y_pred = clf.predict([digits.data[-1]])
# y_true = digits.target[-1]

# print(y_pred)
# print(y_true)
x_1, y_1 = PlanetMass, eccentricity
#
# print len(x_1)
# print len(y_1)
# clf.fit(x_1, y_1)
import pandas as pd
df = pd.DataFrame(list(zip(Distance, StarMass)),
               columns =['d', 'smass'])
df3 = pd.DataFrame(list(Distance), columns=['d'])



x=df[:-80]
y=eccentricity[:-80]
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
clf = linear_model.LinearRegression()
clf.fit(X_, y)
predict = [Distance[-27], StarMass[-27]]
predict_ = poly.fit_transform(predict)


def error (pred, act):
    return (pred-act)/act
predict2 =[]
for i in range (80):
    predict2.append([Distance[-i], StarMass[-i]])
predict2_ = poly.fit_transform(predict2)

preds2 = []
for i in range(80):
    preds2.append(clf.predict(predict2_[i])+0.06)
acts2 = eccentricity[-80:]
errordiff = []
for i in range(80):
    errordiff.append(error(preds2[i], acts2[i]))

df2 = pd.DataFrame(list(zip(preds2, acts2)),
               columns =['pred', 'act'])
predictreal = [778.57*10**9, 10**30]
predictreal_ = poly.fit_transform(predictreal)



x2 = df3[:-80]
y2 = firstcoefficient[:-80]
clf2 = linear_model.LinearRegression()
clf2.fit(x2, y2)
predict3 =[]
for i in range (80):
    predict3.append(Distance[-i])



acts2 = firstcoefficient[-80:]
preds3 = []
for i in range(80):
    preds3.append(clf2.predict(predict3[i]))
errordiff2 = []
for i in range(80):
    errordiff2.append(error(preds3[i], acts2[i]))


pickle.dump(Distance, open(os.getcwd()+'/Distance', 'wb'))
pickle.dump(firstcoefficient, open(os.getcwd()+'/firstcoefficient', 'wb'))


