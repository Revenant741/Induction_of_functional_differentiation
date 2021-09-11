import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy

weight = []

with open('ga_data/5epoch_binde1.dat','rb') as f:
  d2 = pickle.load(f)
  #print(d2[0])
  weight = d2

numpy.set_printoptions(threshold=numpy.inf)
print(weight[-20])