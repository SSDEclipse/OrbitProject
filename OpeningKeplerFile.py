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
filename = 'exoplanets4.csv'
columnNames = ['loc_rowid', 'mpl_orbper', 'mpl_orbsmax', 'mpl_orbeccen', 'mpl_orbincl', 'mpl_bmassj', 'mpl_radj', 'mpl_dens', 'mst_teff', 'mst_mass', 'mst_rad', 'mpl_massj', 'mpl_masse', 'mpl_rade', 'mpl_rads', 'mpl_ratror']
data = pandas.read_csv(filename, names = columnNames)
jupiterRadiiNan = list(data['mpl_radj'])
rowid = list(data['loc_rowid'])
del rowid[0:1]
del jupiterRadiiNan[0:1]
print(jupiterRadiiNan)
print(data.shape)