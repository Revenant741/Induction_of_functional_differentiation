  
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import torch
import random
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_def.Use_Model import Use_Model
from input import inputdata
import train

#1世代分の優秀個体の全てのニューロンにおける相互情報量を算出
def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=5)
  parser.add_argument('--name', type=str, default="hf_ga5_epoch200_firstmodel", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_20_0_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_20_0_model.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--write_name', default='MI', help='savename')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--batch_num', type=int,default=1, help='use_optimizer')
#python3 src/plot/mutial_info_plot.py --model_point 20 --write_name 'ga4_hf_20_'

def import_data_and_clean():
  h_in_x = []
  h_in_y = []
  h_out_x = []
  h_out_y = []
  in_x = []
  in_y = []
  out_x = []
  out_y = []
  #ga_mode= False
  ga_mode= True
  
  read_name = 'ga_hf_5_Normal'
  write_name = 'ga_hf_5_ana'
  generation = 10
  survivor = 20 #生き残る個体
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
  if ga_mode == True:
    change_float_for_ga(h_in_x,in_x)
    change_float_for_ga(h_in_y,in_y)
    change_float_for_ga(h_out_x,out_x)
    change_float_for_ga(h_out_y,out_y)
  else:
    change_float(h_in_x,in_x)
    change_float(h_in_y,in_y)
    change_float(h_out_x,out_x)
    change_float(h_out_y,out_y)
  print(in_x)


def change_float_for_ga(str_list,float_list):
  for i in range(generation*survivor):
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)

  in_x = []
  in_y = []
  out_x = []
  out_y = []

  for i in range(args.model_point):
    args.model_point = i*-1
    setup = Use_Model(args)
    optimizer = torch.optim.Adam
    inputdata_test = inputdata.make_test(args)
    #探索後の重みと接続のデータの指定
    model, binde1, binde2, binde3, binde4 = setup.finded_ga_binde()
    print("finded_binde")
    #相互情報量の分析
    training= train.Adam_train(args,model,optimizer,inputdata_test)
    h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
    in_x.append(h_in_x)
    in_y.append(h_in_y)
    out_x.append(h_out_x)
    out_y.append(h_out_y)
  fig = plt.figure()
  #plt.scatter(in_x[-60:],in_y[-60:], c='blue',label="input neurons")
  #plt.scatter(out_x[-60:],out_y[-60:], c='red',label="output neurons")
  plt.scatter(in_x[-60:],in_y[-60:], c='blue')
  plt.scatter(out_x[-60:],out_y[-60:], c='red')
  plt.xlabel('I_{sp}',fontsize=15)
  plt.ylabel('I_{tp}',fontsize=15)
  plt.legend(loc='upper right')
  #plt.legend(fontsize=18)
  plt.xlim(0,0.7)
  plt.ylim(0,0.7)
  #plt.savefig('src/img/'+write_name+'mutial_info.png')
  print('-------------succes------------')
  plt.savefig('src/img/'+args.write_name+'mutial_info.svg')
  plt.savefig('src/img/'+args.write_name+'mutial_info.pdf')