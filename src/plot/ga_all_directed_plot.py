import numpy as np
import torch
import torch.nn as nn
import random
import copy
import matplotlib.pyplot as plt
import csv
import sys
import pickle
import os
import cloudpickle
import argparse
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from networkx.algorithms import bipartite
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as Model
import pygraphviz as pgv
import seaborn as sns
from my_def import Use_Model
from input import inputdata
sys.path.append("../src")
import train
import matplotlib.cm as cm
from plot import directed_plot as dp

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=195)
  parser.add_argument('--read_name', type=str, default='ga_hf_20/ga_hf_20_0', help='save_file_name')
  parser.add_argument('--write_name', type=str, default='directed_test', help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--model_path', type=str, default='ga_hf_20/ga_hf_20_0_', help='import_file_name_model')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  #python3 src/plot/ga_all_directed_plot.py --write_name '20epoch/ga_hf_dir/d_ga_hf_20'

def No_binde(size_middle=16):
  binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

def direct_plot(args,model,binde1,binde2,binde3,binde4,h_in_x,h_in_y,h_out_x,h_out_y,num):
  #重みの用意
  weight0_data,weight1_data,weight12_data,weight2_data,weight21_data,weight_fc_data = dp.weight_division(model)
  #print(weight12_data)
  #ノードの準備
  input_node = []
  output_node = []
  fc_node = []
  in_node = []
  color_patt = {"color" : "magenta"}
  color_patt1 = {"color" : "blue"}
  color_patt2 = {"color" : "red"}
  color_patt3 = {"color" : "green"}
  color_patt4 = {"color" : "orange"}
  color_patt5 = {"color" : "cyan"}

  #入力層
  for x in range(-16,0):
    in_node.append((x,color_patt1))

  #出力層
  #空間情報
  for j in range(32,35):
    #print(j)
    fc_node.append((j,color_patt4))
  #時間情報
  for j in range(35,38):
    #print(j)
    fc_node.append((j,color_patt5))

  for x in range(-16,0):
    in_node.append(x)
  all_binde = torch.ones(16,16)
  for j in range(32,38):
    fc_node.append(j)
  all_binde = torch.ones(16,16)
  
  #Input Neurons
  neuron_num = 0
  for n in range(0,16):
    #print(n)
    if h_in_x[neuron_num] > h_in_y[neuron_num]:
      input_node.append((n,color_patt2))
    elif h_in_x[neuron_num] < h_in_y[neuron_num]:
      input_node.append((n,color_patt3))
    else:
      input_node.append((n,color_patt))
    neuron_num += 1    
  #Output Neurons
  neuron_num = 0
  for m in range(16,32):
    #print(m)
    if h_out_x[neuron_num] > h_out_y[neuron_num]:
      output_node.append((m,color_patt2))
    elif h_out_x[neuron_num] < h_out_y[neuron_num]:
      output_node.append((m,color_patt3))
    else:
      output_node.append((m,color_patt))
    neuron_num += 1
  #print(in_node)
  #print(all_binde)
  #数式的二は　b1*w1+ b2*w21  b3*w12+b4*w2
  direct0 = dp.directed_in(args,in_node,input_node,all_binde,weight0_data,'_indata')
  direct1 = dp.directed_return(args,input_node,binde1,weight1_data,'_input_input')
  direct2 = dp.directed_return(args,output_node,binde4,weight2_data,'_output_output')
  direct3 = dp.directed_forward(args,input_node,output_node,binde3,weight12_data,'_input_output')
  direct4 = dp.directed_forward(args,input_node,output_node,binde2,weight21_data,'_output_input')
  direct5 = dp.directed_out(args,output_node,all_binde,fc_node,weight_fc_data,'fc',)
  dp.directed_all(args,in_node,input_node,output_node,fc_node,direct0,direct1,direct2,direct3,direct4,direct5,'all_NO_'+str(num))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  generation = 10
  print("finded_model")
  bindes, models = dp.import_data(args)
  #print(models)
  for i in range(generation):
    if i == 0:
      num = 0
    else:
      num += 20 
    if num == 80:
      num = 81
    model_num=num
    print(model_num)
    #モデルと接続構造をインポート
    #binde = bindes[-20]
    #binde = torch.from_numpy(binde).clone()
    #binde = binde.to(args.device)  
    model=models[model_num]
    #拘束条件がまともなら↓2行を逆に
    #binde1,binde2,binde3,binde4 = dp.binde_division(binde)
    binde1, binde2, binde3, binde4 = No_binde()
    optimizer = torch.optim.Adam
    inputdata_test = inputdata.make_test(args)
    #相互情報量の分析
    training= train.Adam_train(args,model,optimizer,inputdata_test)
    h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)   
    #ニューロンと重みの接続状態の描画
    direct_plot(args,model,binde1,binde2,binde3,binde4,h_in_x,h_in_y,h_out_x,h_out_y,i)
