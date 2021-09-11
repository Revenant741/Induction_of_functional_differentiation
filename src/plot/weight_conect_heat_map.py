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

def import_data(args):
  bindes = []
  with open('src/data/'+args.read_name+'_binde.dat','rb') as f:
    bindes = pickle.load(f)
    
    model = Model.esn_model.Binde_ESN_Execution_Model(args)
    with open('src/data/'+args.model_path+'model.pkl', 'rb') as f:
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
  return weight1_data,weight2_data,weight3_data,weight4_data


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  bindes, model = import_data(args)
  weight1_data,weight2_data,weight3_data,weight4_data = weight_division(model)
  print('-------------succes------------')
  print(weight1_data.shape)
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




