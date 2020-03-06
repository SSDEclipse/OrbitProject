
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
#
# filename = 'Planets640FewColumns2-EccentricityNoZeroes.csv'
# # columnNames = ['pl_hostname', 'pl_letter', 'pl_name', 'pl_discmethod', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'st_mass', 'st_masserr1', 'st_masserr2', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_masse', 'pl_masseerr1', 'pl_masseerr2']
# data = pandas.read_csv(filename, names = columnNames)
# print(data.shape)

# filename = 'Planets640FewColumns2-EccentricityNoZeroes.csv'
# columnNames = ['pl_hostname', 'pl_letter', 'pl_name', 'pl_discmethod', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'st_mass', 'st_masserr1', 'st_masserr2', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_masse', 'pl_masseerr1', 'pl_masseerr2']
data = pandas.read_csv("Planets640FewColumns2-EccentricityNoZeroes.csv")
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
print (eccentricityerrorlow)
print (eccentricityerrorhigh)
print eccentricityhigh
print (eccentricity)
print eccentricitylow


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
print PlanetMass
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

print(PlanetMassErrorHigh2)
print(PlanetMassErrorLow2)
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
print(PlanetMassHigh)
print(PlanetMass)
print(PlanetMassLow)

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
print ('we got em chief')

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
print('yeet')
for i in range(len(PlanetMassLow)):
    if PlanetMassLow[i]!=0:
        PlanetMassLow1.append(PlanetMassLow[i])
    else:
        print i
print len(PlanetMassLow1)
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
print highenergy
print highmomentum
print PlanetMassLow
print StandardGPlow
print 1./2.
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
# loweccentricitynozeroes = []
# for i in range(len(PlanetMassLow1)):
#     if loweccentricity[i]>0:
#         loweccentricitynozeroes.append((loweccentricity[i])**0.5)
# print len(loweccentricitynozeroes)
print len(loweccentricity)
print len(higheccentricitynozeroes)
print len(higheccentricity)
print (eccentricity)
print len(mideccentricity)
eccentricitysquared = []
for i in range(len(eccentricity)):
    eccentricitysquared.append(eccentricity[i]**2)
print eccentricitysquared
print higheccentricity
print mideccentricity
print eccentricityhigh
print eccentricity
print eccentricitylow
# for i in range(len(PlanetMassErrorHigh)):
#     PlanetMassErrorHigh[i]>i>PlanetMassErrorLow[i]
print('its time for some rarted code')
# hmm, it needs a int for the range, but i have floats
# maybe be a big brain and mutiply all of the elements in the list by 10^n * x so that everything becomes an int
# find x by looking at how many siggy figgies the error bars use
print PlanetMassErrorHigh

# newlist = []
# for i in range(len(PlanetMassErrorHigh)):
#     newlist.append(round(PlanetMassErrorHigh[i], 3))
# print round(PlanetMassErrorHigh[0], 0)
# print PlanetMassErrorHigh[0]
# x = round(1.567891234*10, 2)
# print x
# print newlist
# testeccentricity = []
# for i in range(len(lowmomentum)):
#     for e in ((lowenergy[i]), (highenergy[i])):
#         for l in ((lowmomentum[i]), (highmomentum[i])):
#             for m in ((PlanetMassLow[i]), (PlanetMassHigh[i])):
#                 for u in ((StandardGPlow[i]), (StandardGPhigh[i])):
#                     testeccentricity.append(eccentricitynoroot(i, l, m, u))
# for i in ((lowenergy[i]), (highenergy[i])):
#     for l in ((lowmomentum[l]), (highmomentum[l])):
#         for m in ((PlanetMassLow[m]), (PlanetMassHigh[m])):
#             for u in ((StandardGPlow[u]), (StandardGPhigh[u])):
#                 testeccentricity.append(eccentricitynoroot(i, l, m, u))
print ((highenergy[0]-midenergy[0]))
print ((lowenergy[0]-midenergy[0]))
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
print len(testeccentricity)
print len(eccentricity)
print 'its testin time'
print testeccentricity
print eccentricitysquared
print 'testin over'
eccentricitydifference = []
for i in range(len(eccentricitysquared)):
    eccentricitydifference.append(testeccentricity[i]-eccentricitysquared[i])
print eccentricitydifference
print 'C is an average grade'
print ((math.fabs(sum(eccentricitydifference))/len(eccentricitydifference)))**0.5
print 'holy  its so small'
#digit dataset from sklearn
digits = datasets.load_digits()
print len(digits)
print digits
#create the LinearRegression model
clf = linear_model.LinearRegression()
#set training set
x, y = digits.data[:-1], digits.target[:-1]
yeet = zip(eccentricity, PlanetMass)

print x
print y
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
print yeet
import pandas as pd
df = pd.DataFrame(list(zip(Distance, StarMass)),
               columns =['d', 'smass'])
df3 = pd.DataFrame(list(Distance), columns=['d'])

# # print df
# df = df1[['pl_orbsmax', 'pl_massj']]
# df1.drop(df1.index[0])
print 'ur bad'
print df
x=df[:-80]
y=eccentricity[:-80]
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
clf = linear_model.LinearRegression()
clf.fit(X_, y)
predict = [Distance[-27], StarMass[-27]]
predict_ = poly.fit_transform(predict)
print 'bot'
print clf.predict(predict_)
print eccentricity[-27]

def error (pred, act):
    return (pred-act)/act
predict2 =[]
for i in range (80):
    predict2.append([Distance[-i], StarMass[-i]])
print predict2
predict2_ = poly.fit_transform(predict2)
print 'reeeee'
print predict2
preds2 = []
for i in range(80):
    preds2.append(clf.predict(predict2_[i])+0.06)
acts2 = eccentricity[-80:]
errordiff = []
for i in range(80):
    errordiff.append(error(preds2[i], acts2[i]))
print sum(eccentricityerrorlow)/len(eccentricityerrorlow)
print sum(eccentricity)/len(eccentricity)
df2 = pd.DataFrame(list(zip(preds2, acts2)),
               columns =['pred', 'act'])
print df2
predictreal = [778.57*10**9, 10**30]
predictreal_ = poly.fit_transform(predictreal)
print clf.predict(predictreal_)
print (sum(errordiff))/80


x2 = df3[:-80]
y2 = firstcoefficient[:-80]
clf2 = linear_model.LinearRegression()
print x
print x2
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
print sum(errordiff2)/len(errordiff2)
print firstcoefficient
print Distance
# test = []
# for i in range(len(firstcoefficient)):
#     test.append(Distance[i]/firstcoefficient[i])
pickle.dump(Distance, open(os.getcwd()+'/Distance', 'wb'))
pickle.dump(firstcoefficient, open(os.getcwd()+'/firstcoefficient', 'wb'))


