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
print('Hello there')
filename = 'Planets640FewColumns.csv'
columnNames = ['pl_hostname', 'pl_letter', 'pl_name', 'pl_discmethod', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'st_mass', 'st_masserr1', 'st_masserr2', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_masse', 'pl_masseerr1', 'pl_masseerr2']
data = pandas.read_csv(filename, names = columnNames)
print(data.shape)
planetmass = list(data['pl_massjerr2'])
starmass = list(data['pl_orbeccenerr2'])
distance = list(data['pl_name'])
del planetmass[0]
del starmass[0]
del distance[0]
planetmassnew2 = [float (c) for c in planetmass]
starmassnew2 = [float (c) for c in starmass]
distancenew2 = [float (c) for c in distance]
planetmassnew = []
starmassnew = []
distancenew = []
for i in range(len(planetmassnew2)):
    planetmassnew.append(5.989*10**24*planetmassnew2[i])
for i in range(len(starmassnew2)):
    starmassnew.append(1.989*10**30*starmassnew2[i])
for i in range(len(distancenew2)):
    distancenew.append(149597870691.*distancenew2[i])

print(len(distance))
if len(planetmassnew)==len(starmassnew)==len(distancenew):
    print('yeet')
StandardGravitationalParameter = []
for i in range(len(starmassnew)):
    StandardGravitationalParameter.append(-6.674*10**-11*starmassnew[i])

def Energy (M, m, d):
    return (6.674*10**-11)*(-1.)*M*m*(1./(2.*d))
PlanetsOrbitalEnergy = []
for i in range(len(starmassnew)):
    PlanetsOrbitalEnergy.append(Energy(starmassnew[i], planetmassnew[i], distancenew[i]))

def AngularMomentum (m,d,M):
    return (m*d*(6.674*10**-11*M*(1/d))**0.5)
PlanetsAngularMomentum = []
for i in range(len(planetmassnew)):
    PlanetsAngularMomentum.append(AngularMomentum(planetmassnew[i], distancenew[i], starmassnew[i]))
print(PlanetsAngularMomentum)

# print(Energy(1.989*10**30, 5.99*10**24, 1.49*10**11))
print(len(starmassnew))
print(len(PlanetsAngularMomentum))
print(len(PlanetsOrbitalEnergy))
print(len(StandardGravitationalParameter))
PlanetsOrbitalEccentricitynoroot = []

def eccentricitynoroot (e, l, m, u):
    return (1+(2*e*(l**2))/((m**3)*(u**2)))
for i in range(len(starmassnew)):
    PlanetsOrbitalEccentricitynoroot.append(eccentricitynoroot(PlanetsOrbitalEnergy[i], PlanetsAngularMomentum[i], planetmassnew[i], StandardGravitationalParameter[i]))


def eccentricity (Energy, l, m, u):
    return (1+(2*Energy*(l**2))/((m**3)*(u**2)))**0.5
PlanetsOrbitalEccentricity = []
for i in range(len(starmassnew)):
    PlanetsOrbitalEccentricity.append(eccentricity(PlanetsOrbitalEnergy[i], PlanetsAngularMomentum[i], planetmassnew[i], StandardGravitationalParameter[i]))