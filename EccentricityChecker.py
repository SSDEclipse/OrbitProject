from __future__ import division

import pandas
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import math
import pickle, os
from sklearn.cluster import KMeans
import numpy as np
import csv
import random

filename = 'Planets640FewColumns2-EccentricityNoZeroes.csv'
columnNames = ['pl_hostname', 'pl_letter', 'pl_name', 'pl_discmethod', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'st_mass', 'st_masserr1', 'st_masserr2', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_masse', 'pl_masseerr1', 'pl_masseerr2']
data = pandas.read_csv(filename, names = columnNames)
print(data.shape)
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

def AngularMomentum (m,d,M):
    return (m*d*(6.674*10**-11*M*(1/d))**0.5)
highmomentum = [AngularMomentum(PlanetMassHigh[i], DistanceHigh[i], StarMassHigh[i]) for i in range(len(PlanetMass))]
lowmomentum = [AngularMomentum(PlanetMassLow[i], DistanceLow[i], StarMassLow[i]) for i in range(len(PlanetMass))]
def SGP (M):
    return -6.674*10**-11*M
StandardGPhigh = [SGP(StarMassHigh[i]) for i in range(len(StarMass))]
StandardGPlow = [SGP(StarMassLow[i]) for i in range(len(StarMass))]
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
print highenergy
print highmomentum
print PlanetMassLow
print StandardGPlow
print 1./2.
def eccentricitynoroot (Energy, l, m, u):
    return (1+(2*Energy*(l**2))/((m**3)*(u**2)))

higheccentricity = [eccentricitynoroot(lowenergy[i], lowmomentum[i], PlanetMassHigh[i], StandardGPhigh[i]) for i in range(len(PlanetMassHigh))]
loweccentricity = [eccentricitynoroot(highenergy[i], highmomentum[i], PlanetMassLow1[i], StandardGPlow[i]) for i in range(len(PlanetMassLow1))]
higheccentricitynozeroes = []
for i in range(len(PlanetMass)):
    if higheccentricity[i]>0:
        higheccentricitynozeroes.append((higheccentricity[i])**0.5)

# loweccentricitynozeroes = []
# for i in range(len(PlanetMassLow1)):
#     if loweccentricity[i]>0:
#         loweccentricitynozeroes.append((loweccentricity[i])**0.5)
# print len(loweccentricitynozeroes)
print len(loweccentricity)
print len(higheccentricitynozeroes)
print len(higheccentricity)

print eccentricity
# for i in range(len(PlanetMassErrorHigh)):
#     PlanetMassErrorHigh[i]>i>PlanetMassErrorLow[i]