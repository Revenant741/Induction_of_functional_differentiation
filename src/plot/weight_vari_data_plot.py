from ast import arg
import csv
from statistics import mode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import torch
import random
import argparse
import torch.nn as nn
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_def import Analysis
import hessian_train
from my_def import hessianfree
from my_def.Use_Model import Use_Model
import model as Model
from input import inputdata
import train
import cloudpickle

#重みの変動値をInput NeuronsとOutput Neuronsで分けて描画
def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:1", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=180)
  #parser.add_argument('--name', type=str, default="hf_ga5_epoch200_firstmodel", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20_model.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--write_name', default='MI', help='savename')
  parser.add_argument('--neuron_start', type=int, default=-20, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--batch_num', type=int,default=1, help='use_optimizer')
  parser.add_argument('--name', type=str, default='adam_test', help='save_file_name')
  #python3 src/plot/weight_vari_data_plot.py --name 'neurons_sp'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #純粋な個体精度の結果
  accuracy = []
  loss = []
  accuracy2 = []
  loss2 = []
  #重み更新量のリスト
  weight = []
  weight2 = [] 
  #保存場所
  name = args.name
  point = 'src/img/'

  read_name = 'weight_diff_sp/neurons'

  with open('src/data/'+read_name+'_sp_acc.csv') as f:
      for row in csv.reader(f):
          accuracy.append(float(row[0])*100)

  with open('src/data/'+read_name+'_tp_acc.csv') as f:
      for row in csv.reader(f):
          accuracy2.append(float(row[0])*100)

  epoch =[i+1 for i in range(len(accuracy))]
  plt.figure()
  #plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
  #plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
  plt.xlabel('number of time',fontsize=15)
  plt.ylabel('Accuracy(%)',fontsize=15)
  #plt.yticks((20,30,40,50,60,60,70,80,90,100))
  plt.ylim(10, 105)
  #精度の描画
  plt.plot(epoch,accuracy,label="spatial information",color="g")
  plt.plot(epoch,accuracy2,label="temporal information",color="r")
  plt.legend(loc=4)
  plt.savefig(f''+point+name+'_acc.png')
  plt.savefig(f''+point+name+'_acc.pdf')
  
  with open('src/data/'+read_name+'_sp_loss.csv') as f:
    for row in csv.reader(f):
        loss.append(float(row[0]))

  with open('src/data/'+read_name+'_tp_loss.csv') as f:
    for row in csv.reader(f):
        loss2.append(float(row[0]))

  print(len(loss))
  #誤差の描画
  plt.figure()
  plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  plt.xlabel('number of time',fontsize=15)
  plt.ylabel('Loss',fontsize=15)
  plt.ylim(-0.1, 2.5)
  plt.plot(epoch,loss,label="spatial information",color="g")
  plt.plot(epoch,loss2,label="temporal information",color="r")
  plt.legend(loc=1)
  plt.savefig(f''+point+name+'_loss.png')
  plt.savefig(f''+point+name+'_loss.pdf')
  
  #重みの更新量の描画

  with open('src/data/'+read_name+'_Input_Neurons_var_list.csv') as f:
    for row in csv.reader(f):
        weight.append(float(row[0]))

  with open('src/data/'+read_name+'_Output_Neurons_var_list.csv') as f:
    for row in csv.reader(f):
        weight2.append(float(row[0]))

  #epoch =[i+1 for i in range(len(loss))]
  #重み更新量の描画
  plt.figure()
  plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  plt.xlabel('number of time',fontsize=15)
  plt.ylabel('Update',fontsize=15)
  plt.ylim(5, 180)
  plt.plot(epoch,weight,label="Input Neurons_update")
  plt.plot(epoch,weight2,label="Output Neurons_update")
  plt.legend(loc=0)
  plt.savefig(f''+point+name+'_update.png')
  plt.savefig(f''+point+name+'_update.pdf')
