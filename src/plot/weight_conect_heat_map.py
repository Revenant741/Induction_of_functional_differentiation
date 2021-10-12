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
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as Model
import pygraphviz as pgv
import seaborn as sns
import matplotlib.cm as cm

#各層の接続状態をヒートマップにしているプログラム
def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=195)
  parser.add_argument('--read_name', type=str, default='ga_hf_5_Normal', help='save_file_name')
  parser.add_argument('--write_name', type=str, default='directed_test', help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_5_Normal_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='hf_ga5_epoch200_bestmodel', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  #src/weight_conect.py --neuron_start 0 --neuron_num 16
  #python3 src/plot/weight_conect_heat_map.py  --read_name ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/weight_conect_heat_map.py --write_name '/conectome/conectome' --read_name func_diff_e20_p20_l10 --model_path func_diff_e20_p20_l10 --device 'cuda:0'
  #python3 src/plot/weight_conect_heat_map.py --write_name '/conectome/conectome_loss_eva_g100' --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'

def import_data(args):
  bindes = []
  with open('src/data/'+args.read_name+'_binde.dat','rb') as f:
    bindes = pickle.load(f)
    
    model = Model.esn_model.Binde_ESN_Execution_Model(args)
    with open('src/data/'+args.model_path+'_model.pkl', 'rb') as f:
        model = cloudpickle.load(f)
    np.set_printoptions(threshold=np.inf)
  return bindes, model

def weight_division(model):
  print(model.state_dict()['binde_esn.w_res1'].shape)
  print(model.state_dict()['binde_esn.w_res12'].shape)
  print(model.state_dict()['binde_esn.w_res2'].shape)
  print(model.state_dict()['binde_esn.w_res21'].shape)
  weight1 = model.state_dict()['binde_esn.w_res1'].tolist()
  weight2 = model.state_dict()['binde_esn.w_res12'].tolist()
  weight3 = model.state_dict()['binde_esn.w_res2'].tolist()
  weight4 = model.state_dict()['binde_esn.w_res21'].tolist()
  weight1_data = np.array(weight1)
  weight2_data = np.array(weight2)
  weight3_data = np.array(weight3)
  weight4_data = np.array(weight4)
  Input_Neurons = np.hstack([weight1_data,weight2_data])
  Output_Neurons = np.hstack([weight3_data,weight4_data])
  ALL_Neurons = np.vstack([Input_Neurons,Output_Neurons])
  return weight1_data,weight2_data,weight3_data,weight4_data,ALL_Neurons

def plus_binde(binde,ALL_Neurons):
  ALL_Neurons *= binde
  return ALL_Neurons

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #拘束条件とモデルをインポート
  bindes, model = import_data(args)
  binde = bindes[999]
  model = model[999]
  #重みの値の調整
  weight1_data,weight2_data,weight3_data,weight4_data,ALL_Neurons = weight_division(model)
  #重みの値に拘束条件を付与
  ALL_Neurons=plus_binde(binde,ALL_Neurons)
  print('-------------succes------------')
  #print(weight1_data.shape)
  print(ALL_Neurons)
  fig = plt.figure()
  sns.heatmap(weight1_data, cmap=cm.jet)
  plt.xlabel("InputNeurons")
  plt.ylabel("InputNeurons")
  plt.savefig('src/img/'+args.write_name+'_weight1_heat.png')
  fig = plt.figure()
  sns.heatmap(weight2_data, cmap=cm.jet)
  plt.xlabel("InputNeurons")
  plt.ylabel("OutputNeurons")
  plt.savefig('src/img/'+args.write_name+'_weight2_heat.png')
  fig = plt.figure()
  sns.heatmap(weight3_data, cmap=cm.jet)
  plt.xlabel("OutputNeurons")
  plt.ylabel("OutputNeurons")
  plt.savefig('src/img/'+args.write_name+'_weight3_heat.png')
  fig = plt.figure()
  sns.heatmap(weight4_data, cmap=cm.jet)
  plt.xlabel("OutputNeurons")
  plt.ylabel("InputNeurons")
  plt.savefig('src/img/'+args.write_name+'_weight4_heat.png')
  fig = plt.figure()
  sns.heatmap(ALL_Neurons, cmap=cm.jet,vmax=1.5, vmin=-1.5)
  plt.xlabel("Neuron_Number")
  plt.ylabel("Conect_Number")
  plt.savefig('src/img/'+args.write_name+'_weight_binde_ALL_heat.png')




