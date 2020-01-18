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
# Loading files of 640 planets w/ mass of planet, star, and distance between them - These numbers are in kilograms
StarMass = pickle.load(open(os.getcwd()+'/starmassnew', 'rb'))
PlanetMass = pickle.load(open(os.getcwd()+'/planetmassnew', 'rb'))
Distance = pickle.load(open(os.getcwd()+'/distancenew', 'rb'))

# Loading in errors for above files - These numbers are in Earth masses - need to convert them to kilograms to get correct error bars
PlanetMassErrorHigh = pickle.load(open(os.getcwd()+'/PlanetMassErrorHigh', 'rb'))
PlanetMassErrorLow = pickle.load(open(os.getcwd()+'/PlanetMassErrorLow', 'rb'))
StarMassErrorHigh = pickle.load(open(os.getcwd()+'/StarMassErrorHigh', 'rb'))
StarMassErrorLow = pickle.load(open(os.getcwd()+'/StarMassErrorLow', 'rb'))
DistanceErrorHigh = pickle.load(open(os.getcwd()+'/DistanceErrorHigh', 'rb'))
DistanceErrorLow = pickle.load(open(os.getcwd()+'/DistanceErrorLow', 'rb'))
# del PlanetMassErrorHigh[0]
# del PlanetMassErrorLow[0]
# del DistanceErrorHigh[0]
# del DistanceErrorLow[0]
# del StarMassErrorHigh[0]
# del StarMassErrorLow[0]
# maipulating errors to floats
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
    PlanetMassErrorHigh.append(5.989*10**24*PlanetMassErrorHigh2[i])
for i in range(len(PlanetMassErrorLow2)):
    PlanetMassErrorLow.append(5.989*10**24*PlanetMassErrorLow2[i])
for i in range(len(StarMassErrorHigh2)):
    StarMassErrorHigh.append(1.989*10**30*StarMassErrorHigh2[i])
for i in range(len(StarMassErrorLow2)):
    StarMassErrorLow.append(1.989*10**30*StarMassErrorLow2[i])
for i in range(len(DistanceErrorHigh2)):
    DistanceErrorHigh.append(149597870691.*DistanceErrorHigh2[i])
for i in range(len(DistanceErrorLow2)):
    DistanceErrorLow.append(149597870691.*DistanceErrorLow2[i])




print(len(PlanetMass))
print(len(StarMass))
print(len(Distance))
print(len(PlanetMassErrorHigh))
print(len(PlanetMassErrorLow))
print(len(StarMassErrorLow))
print(len(StarMassErrorHigh))
print(len(DistanceErrorLow))
print(len(DistanceErrorHigh))
print((PlanetMassErrorHigh))
print((PlanetMassErrorLow))
print('yeet')
print((StarMassErrorLow))
print((StarMassErrorHigh))
print((DistanceErrorLow))
print((DistanceErrorHigh))


# trying to add errors to recorded values
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