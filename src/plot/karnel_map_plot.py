  
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import random
import seaborn as sns
from sklearn.neighbors import KernelDensity
import pandas as pd

h_in_x = []
h_in_y = []
h_out_x = []
h_out_y = []
in_x = []
in_y = []
out_x = []
out_y = []
#等高線の形でプロットするプログラム
read_name='ga_hf_10_Normal'
write_name= 'ga_heatmap_hf_10_Normal'
generation = 4
survivor = 20 #生き残る個体
takepoint = -survivor

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

def change_float(str_list,float_list):
  for i in range(survivor):
    num = str_list[i].split(',')
    num[0]=num[0].replace('[','')
    num[-1]=num[-1].replace(']','')
    num = [float(n) for n in num]
    float_list.append(num)

change_float(h_in_x[takepoint:],in_x)
change_float(h_in_y[takepoint:],in_y)
change_float(h_out_x[takepoint:],out_x)
change_float(h_out_y[takepoint:],out_y)
calams = list(range(1,17))
index = list(range(1,640))
sp_all = []
tp_all = []
sp_all.append(in_x)
sp_all.append(out_x)
tp_all.append(in_y)
tp_all.append(out_y)
sp_pt = np.array(sp_all).reshape(640,-1)
tp_pt = np.array(tp_all).reshape(640,-1)
df_sp = pd.DataFrame(sp_pt)
df_tp = pd.DataFrame(tp_pt)
fig = plt.figure()
#sns.palplot(sns.color_palette("seismic", 24))
for i in range(1):
  sns.kdeplot(df_sp[i],df_tp[i], shade=True)
plt.xlim(0,0.7)
plt.ylim(0,0.7)
print('-------------succes------------')

plt.savefig('src/img/'+write_name+'mutial_info_K.svg')
plt.savefig('src/img/'+write_name+'mutial_info_K.pdf')