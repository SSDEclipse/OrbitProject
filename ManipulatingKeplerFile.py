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
# Loading files of 640 planets w/ mass of planet, star, and distance between them
starmassnew = pickle.load(open(os.getcwd()+'/starmassnew', 'rb'))
planetmassnew = pickle.load(open(os.getcwd()+'/planetmassnew', 'rb'))
distancenew = pickle.load(open(os.getcwd()+'/distancenew', 'rb'))
# Loading in errors for above files

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
PlanetMassErrorHigh = [float (c) for c in PlanetMassErrorHigh]
PlanetMassErrorLow = [float (c) for c in PlanetMassErrorLow]
DistanceErrorHigh = [float (c) for c in DistanceErrorHigh]
DistanceErrorLow = [float (c) for c in DistanceErrorLow]
StarMassErrorHigh = [float (c) for c in StarMassErrorHigh]
StarMassErrorLow = [float (c) for c in StarMassErrorLow]

PlanetMassHigh = []
PlanetMassLow= []
StarMassHigh= []
StarMassLow= []
DistanceHigh= []
DistanceLow= []

print(len(planetmassnew))
print(len(starmassnew))
print(len(distancenew))
print(len(PlanetMassErrorHigh))
print(len(PlanetMassErrorLow))
print(len(StarMassErrorLow))
print(len(StarMassErrorHigh))
print(len(DistanceErrorLow))
print(len(DistanceErrorHigh))
print((PlanetMassErrorHigh))
print((PlanetMassErrorLow))
print((StarMassErrorLow))
print((StarMassErrorHigh))
print((DistanceErrorLow))
print((DistanceErrorHigh))


# trying to add errors to recorded values
for i in range(len(planetmassnew)):
    PlanetMassHigh.append(planetmassnew[i]+PlanetMassErrorHigh[i])
print(PlanetMassHigh)
print(planetmassnew)

