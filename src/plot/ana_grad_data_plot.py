import csv
from os import read
from traceback import print_tb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 

#純粋な個体精度の結果
input_grad = []
output_grad = []
#保存場所
point = 'src/data/'
name = 'grad/sp_grad'

with open(f''+point+name+'_Input_Neurons_grad_list.csv') as f:
    for row in csv.reader(f):
      input_grad.append(float(row[0]))

with open(f''+point+name+'_Output_Neurons_grad_list.csv', 'r') as f:
    for row in csv.reader(f):
      output_grad.append(float(row[0]))

#print(input_grad)
#print(output_grad)
#print(sum(input_grad)/1800)
#print(sum(output_grad)/1800)
input_grad1=np.maximum(sum(input_grad), 0)
input_grad2=input_grad / np.max(input_grad)

output_grad1=np.maximum(sum(output_grad), 0)
output_grad2=output_grad / np.max(output_grad)
print(input_grad1)
print(output_grad1)
