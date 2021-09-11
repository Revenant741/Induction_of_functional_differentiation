  
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import random
import seaborn as sns
from sklearn.neighbors import KernelDensity
import pandas as pd
import matplotlib.cm as cm

h_in_x = []
h_in_y = []
h_out_x = []
h_out_y = []
in_x = []
in_y = []
out_x = []
out_y = []
ga_mode= False
read_name='hf_ga5_epoch200_bestmodel'
write_name= 'heatmap_bestmodel'
#write_name= 'ga_heatmap_hf_20_first'
generation = 3
survivor = 20 #生き残る個体
takepoint = survivor*(generation-1)
takelast = survivor*generation-1

with open('src/data/'+read_name+'_h_in_x.csv') as f:
  for row in csv.reader(f):
      h_in_x.append((row[0]))
with open('src/data/'+read_name+'_h_in_y.csv') as f:
  for row in csv.reader(f):
      h_in_y.append((row[0]))
with open('src/data/'+read_name+'_h_out_x.csv') as f:
  for row in csv.reader(f):
      h_out_x.append((row[0]))
with open('src/data/'+read_name+'_h_out_y.csv') as f:
  for row in csv.reader(f):
      h_out_y.append((row[0]))

def change_float_for_ga(str_list,float_list):
  for i in range(survivor):
    num = str_list[i].split(',')
    num[0]=num[0].replace('[','')
    num[-1]=num[-1].replace(']','')
    num = [float(n) for n in num]
    float_list.append(num)

def change_float(str_list,float_list):
  num = []
  for i in range(len(str_list)):
    num.append(float(str_list[i]))
  float_list.append(num)

if ga_mode == True:
  change_float_for_ga(h_in_x[takepoint:takelast],in_x)
  change_float_for_ga(h_in_y[takepoint:takelast],in_y)
  change_float_for_ga(h_out_x[takepoint:takelast],out_x)
  change_float_for_ga(h_out_y[takepoint:takelast],out_y)
  sp_all = []
  tp_all = []
  sp_all.append(in_x)
  sp_all.append(out_x)
  tp_all.append(in_y)
  tp_all.append(out_y)
  sp_pt = np.array(sp_all).reshape(640,-1)
  tp_pt = np.array(tp_all).reshape(640,-1)
else:
  change_float(h_in_x,in_x)
  change_float(h_in_y,in_y)
  change_float(h_out_x,out_x)
  change_float(h_out_y,out_y)
  sp_all = []
  tp_all = []
  sp_all.append(in_x)
  sp_all.append(out_x)
  tp_all.append(in_y)
  tp_all.append(out_y)
  sp_pt = np.array(sp_all).reshape(32,-1)
  tp_pt = np.array(tp_all).reshape(32,-1)

sp_pt = sp_pt.flatten()
tp_pt = tp_pt.flatten()
fig = plt.figure()
ax = fig.add_subplot(111)
bins=[np.linspace(0,0.7,41),np.linspace(0,0.7,41)]
H = ax.hist2d(sp_pt,tp_pt, bins=bins, cmap=cm.jet)
H[3].set_clim(0,3)
fig.colorbar(H[3],ax=ax)
plt.xlabel('I_{sp}',fontsize=15)
plt.ylabel('I_{tp}',fontsize=15)
plt.xlim(0,0.7)
plt.ylim(0,0.7)
print('-------------succes------------')

plt.savefig('src/img/'+write_name+'mutial_info.png')
plt.savefig('src/img/'+write_name+'mutial_info.pdf')