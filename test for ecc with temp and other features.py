
import pandas
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plot
import math
import pickle, os
import numpy as np
import csv
import random
from sklearn import linear_model, datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import warnings
from matplotlib import cm
from collections import OrderedDict

warnings.filterwarnings("ignore")

data = pandas.read_csv('ecc-mjax-teff-num.csv')
data2 = pandas.read_csv('predictions.csv')
print(data.shape)
import pandas as pd

eccentricity = list(data['pl_orbeccen'])
eccentricityerrorhigh = list(data['pl_orbeccenerr1'])
eccentricityerrorlow = list(data['pl_orbeccenerr2'])
name = list(data2['pl_name'])

print eccentricity
del eccentricity[0]
del eccentricityerrorhigh[0]
del eccentricityerrorlow[0]

eccentricity = [float(c) for c in eccentricity]
eccentricityerrorhigh = [float(c) for c in eccentricityerrorhigh]
eccentricityerrorlow = [float(c) for c in eccentricityerrorlow]
eccentricityhigh = [eccentricity[i]+eccentricityerrorhigh[i] for i in range(len(eccentricity))]
eccentricitylow = [eccentricity[i]+eccentricityerrorlow[i] for i in range(len(eccentricity))]



PlanetRadius = list(data['pl_radj'])
StarMass = list(data['st_mass'])
Distance = list(data['pl_orbsmax'])
PlanetNumber = list(data['pl_pnum'])
EffTemp = list(data2['st_teff'])
StarDistance = list(data2['st_dist'])
Magnitude = list(data2['st_optmag'])
PlanetRadius2 = list(data2['pl_radj'])
PlanetNumber2 = list(data2['pl_pnum'])
Distance2 = list(data2['pl_orbsmax'])

del EffTemp[0]
del StarDistance[0]
del PlanetRadius[0]
del StarMass[0]
del Distance[0]
del PlanetNumber[0]
del Magnitude[0]
del Distance2[0]
del PlanetNumber2[0]
del PlanetRadius2[0]


PlanetRadius = [float(c) for c in PlanetRadius]
StarMass = [float(c) for c in StarMass]
Distance = [float(c) for c in Distance]
PlanetNumber = [float(c) for c in PlanetNumber]
EffTemp = [float(c) for c in EffTemp]
StarDistance = [float(c) for c in StarDistance]
Magnitude = [float(c) for c in Magnitude]
PlanetRadius2 = [float(c) for c in PlanetRadius2]
PlanetNumber2 = [float(c) for c in PlanetNumber2]
Distance2 = [float(c) for c in Distance2]


PlanetRadius = [7.1492*10**7*i for i in PlanetRadius]
StarMass = [1.989*10**30*i for i in StarMass]
Distance = [149597870691.*i for i in Distance]
PlanetRadius2 = [7.1492*10**7*i for i in PlanetRadius2]
Distance2 = [149597870691*i for i in Distance2]


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


clf = linear_model.LinearRegression()

yeet = zip(eccentricity, PlanetRadius)



x_1, y_1 = PlanetRadius, eccentricity

import pandas as pd
df = pd.DataFrame(list(zip(PlanetRadius, Distance, PlanetNumber)), columns =['rad', 'dist', 'pnum'])


x=df[:-80]

y=eccentricity[:-80]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(x)
clf = linear_model.LinearRegression()
clf.fit(X_, y)
predict = [PlanetRadius[-27], Distance[-27], PlanetNumber[-27]]
predict_ = poly.fit_transform(predict)

def error (pred, act):
    return (pred-act)
predict2 =[]
for i in range (80):
    predict2.append([PlanetRadius[-i], Distance[-i], PlanetNumber[-i]])

predict2_ = poly.fit_transform(predict2)

preds2 = []
for i in range(80):
    preds2.append(clf.predict(predict2_[i]))
del preds2[8]
acts2 = eccentricity[-80:]
del acts2[8]
errordiff = []
for i in range(79):
    errordiff.append(error(preds2[i], acts2[i]))
df2 = pd.DataFrame(list(zip(preds2, acts2)),
               columns =['pred', 'act'])
predictreal = [7.57*10**7, 0, 2]
predictreal_ = poly.fit_transform(predictreal)
print 'the error is', (sum(errordiff))/80








BolCorr = []
for i in range(len(EffTemp)):
    if EffTemp[i]>10000:
        BolCorr.append(-2.0)
    elif 10000>EffTemp[i]>7500:
        BolCorr.append(-0.3)
    elif 7500>EffTemp[i]>6000:
        BolCorr.append(-0.15)
    elif 6000>EffTemp[i]>5300:
        BolCorr.append(-0.4)
    elif 5300>EffTemp[i]>3500:
        BolCorr.append(-0.8)
    else:
        BolCorr.append(-2.0)

def AbsoluteMagnitude(app, d):
    return (app-5*math.log((d/10), 10))
def BolometricMagnitude(AbsMag, Correction):
    return AbsMag+Correction
def AbsoluteLuminosity(BolMag):
    return 10**((BolMag-4.72)/(-2.512))
def InnerRadius (AbsLum):
    return ((AbsLum/1.1)**0.5)*149597870691.*0.8
def OuterRadius (AbsLum):
    return ((AbsLum/0.53)**0.5)*149597870691.*1.2
StarAbMag = []
for i in range(len(Magnitude)):
    StarAbMag.append(AbsoluteMagnitude(Magnitude[i], StarDistance[i]))
StarBolMag = []
for i in range(len(Magnitude)):
    StarBolMag.append(BolometricMagnitude(StarAbMag[i], BolCorr[i]))
StarAbLum = []
for i in range(len(Magnitude)):
    StarAbLum.append(AbsoluteLuminosity(StarBolMag[i]))
HabitableInner = []
for i in range(len(Magnitude)):
    HabitableInner.append(InnerRadius(StarAbLum[i]))
HabitableOuter = []
for i in range(len(Magnitude)):
    HabitableOuter.append(OuterRadius(StarAbLum[i]))




predict3 =[]
for i in range(len(PlanetRadius2)):
    predict3.append([PlanetRadius2[i], Distance2[i], PlanetNumber2[i]])
predict3_ = poly.fit_transform(predict3)
preds3 = []
for i in range(len(PlanetRadius2)):
    preds3.append(clf.predict(predict3_[i]))
maxdist =[]
for i in range(len(preds3)):
    maxdist.append(Distance2[i]/(1-preds3[i]))
mindist =[]
for i in range(len(preds3)):
    mindist.append(Distance2[i]/(1+preds3[i]))
habitables = []
for i in range(len(mindist)):
    if mindist[i]>HabitableInner[i]:
         if maxdist[i]<HabitableOuter[i]:
            habitables.append(i+3)
print len(habitables)
names = []
for i in range(len(habitables)):
    names.append(name[habitables[i]-2])
print names


data2.to_csv(r'C:\Users\Krithi\Documents\export_dataframe.csv')



Distance = pickle.load(open(os.getcwd()+'/Distance', 'rb'))
firstcoefficient = pickle.load(open(os.getcwd()+'/firstcoefficient', 'rb'))
import pandas as pd
df3 = pd.DataFrame(list(Distance), columns=['d'])

clf2 = linear_model.LinearRegression()

x2 = df3[80:]
y2 = firstcoefficient[80:]

# Reverse(y2)
clf2.fit(x2, y2)
predict3 =[]
for i in range (80):
    predict3.append(Distance[i])
acts2 = firstcoefficient[0:80]
preds4 = []
for i in range(80):
    preds4.append(clf2.predict(Distance[i]))
def error (pred, act):
    return (pred-act)/act
errordiff2 = []
for i in range(80):
    errordiff2.append(error(preds4[i], acts2[i]))
print "average error is", sum(errordiff2)/len(errordiff2)
preds5 = []
for i in range(len(Distance2)):
    preds5.append(-1*clf2.predict(Distance2[i]))





plot.style.use('default')
fig = plot.figure()

fig.add_subplot(222, projection='polar')

# Set the title of the polar plot

plot.title('Graph of Exoplanet Orbit vs Habitable Zones')

# Radian values upto 2*pi
print len(Distance2)
print len(preds3)
rads = np.arange(0, (2 * np.pi), 0.01)

a = 1

b = 1

k = 1
#
# fig = plot.figure()
#
# for radian in rads:
#     radius =  ((Distance2[69]/149597870691)/(1+preds3[69]*np.cos(k * radian)))
#
#
#     plot.polar(radian, radius, 'o', label='o')
#
# for radian in rads:
#     plot.polar(radian,HabitableInner[69]/149597870691,'o', label='o')
#     plot.style.use('seaborn')
#
# for radian in rads:
#     plot.polar(radian,HabitableOuter[69]/149597870691,'o', label='o')
#     plot.style.use('seaborn')
#
# fig =plot.figure(figsize=(1, 1))
print len(Distance2)
print len(preds5)
print 'this is where the fun begins'
print habitables
for i in range(len(habitables)):
    print names[i]
    print preds5[habitables[i]-3]/149597870691
    print preds3[habitables[i]]
    print HabitableInner[habitables[i]-3]/149597870691
    print HabitableOuter[habitables[i]-3]/149597870691
# plot.show()

