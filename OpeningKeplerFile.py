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
print 'this will give an error at the end, that is natural'
filename = 'Planets640FewColumns2.csv'
columnNames = ['pl_hostname', 'pl_letter', 'pl_name', 'pl_discmethod', 'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim', 'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'st_mass', 'st_masserr1', 'st_masserr2', 'pl_massj', 'pl_massjerr1', 'pl_massjerr2', 'pl_massjlim', 'pl_masse', 'pl_masseerr1', 'pl_masseerr2']
data = pandas.read_csv(filename, names = columnNames)
print(data.shape)
planetmass = list(data['pl_masse'])
starmass = list(data['st_mass'])
distance = list(data['pl_orbsmax'])
PlanetMassErrorHigh = list(data['pl_masseerr1'])
PlanetMassErrorLow = list(data['pl_masseerr2'])
DistanceErrorHigh = list(data['pl_orbsmaxerr1'])
DistanceErrorLow = list(data['pl_orbsmaxerr2'])
StarMassErrorHigh = list(data['st_masserr1'])
StarMassErrorLow = list(data['st_masserr2'])
# data['st_masserr2'] = data['st_masserr2'].fillna(0)
del PlanetMassErrorHigh[0]
del PlanetMassErrorLow[0]
del DistanceErrorHigh[0]
del DistanceErrorLow[0]
del StarMassErrorHigh[0]
del StarMassErrorLow[0]
pickle.dump(StarMassErrorHigh, open(os.getcwd()+'/StarMassErrorHigh', 'wb'))
pickle.dump(StarMassErrorLow, open(os.getcwd()+'/StarMassErrorLow', 'wb'))
pickle.dump(PlanetMassErrorLow, open(os.getcwd()+'/PlanetMassErrorLow', 'wb'))
pickle.dump(PlanetMassErrorHigh, open(os.getcwd()+'/PlanetMassErrorHigh', 'wb'))
pickle.dump(DistanceErrorHigh, open(os.getcwd()+'/DistanceErrorHigh', 'wb'))
pickle.dump(DistanceErrorLow, open(os.getcwd()+'/DistanceErrorLow', 'wb'))


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
    planetmassnew.append(5.976*10**24*planetmassnew2[i])
for i in range(len(starmassnew2)):
    starmassnew.append(1.989*10**30*starmassnew2[i])
for i in range(len(distancenew2)):
    distancenew.append(149597870691.*distancenew2[i])

if len(planetmassnew)==len(starmassnew)==len(distancenew):
    print('test')
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


pickle.dump(starmassnew, open(os.getcwd()+'/starmassnew', 'wb'))
pickle.dump(planetmassnew, open(os.getcwd()+'/planetmassnew', 'wb'))
pickle.dump(distancenew, open(os.getcwd()+'/distancenew', 'wb'))


PlanetsOrbitalEccentricitynoroot = []
def eccentricitynoroot (e, l, m, u):
    return (1+(2*e*(l**2))/((m**3)*(u**2)))
for i in range(len(starmassnew)):
    PlanetsOrbitalEccentricitynoroot.append(eccentricitynoroot(PlanetsOrbitalEnergy[i], PlanetsAngularMomentum[i], planetmassnew[i], StandardGravitationalParameter[i]))

PlanetsOrbitalEccentricityNoZeroes = []
for i in range(len(PlanetsOrbitalEccentricitynoroot)):
    if PlanetsOrbitalEccentricitynoroot[i]>0:
        PlanetsOrbitalEccentricityNoZeroes.append((PlanetsOrbitalEccentricitynoroot[i])**0.5)


def eccentricity (Energy, l, m, u):
    return (1+(2*Energy*(l**2))/((m**3)*(u**2)))**0.5
PlanetsOrbitalEccentricity = []
for i in range(len(starmassnew)):
    PlanetsOrbitalEccentricity.append(eccentricity(PlanetsOrbitalEnergy[i], PlanetsAngularMomentum[i], planetmassnew[i], StandardGravitationalParameter[i]))


