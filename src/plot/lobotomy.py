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
import pygraphviz as pgv
import seaborn as sns
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as Model
from my_def import Use_Model
from input import inputdata
sys.path.append("../src")
import train
import matplotlib.cm as cm
import statistics
import math
import itertools

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=195)
  parser.add_argument('--read_name', type=str, default='ga_hf_20/ga_hf_20_0', help='save_file_name')
  parser.add_argument('--write_name', type=str, default='directed_test', help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  #parser.add_argument('--binde_path', type=str, default='ga_hf_20_best_train.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='ga_hf_20/ga_hf_20_0_', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--how', type=str,default='neuron_cut', help='layer_cut_or_neuron_cut')
  parser.add_argument('--what', type=str,default='loss', help='loss_or_func_diff')
  parser.add_argument('--mode', type=str,default='lobotomy', help='lobotomy_or_lobotomy_reverse')
  parser.add_argument('--mode_num', type=int,default=1, help='1or2or3or4')
  parser.add_argument('--weight_num', type=int,default=0, help='0or1or2or3or4')
  #neuroncut の時　mode 1がsp 2がtp
  #layercut の時　mode 1がsp input 2がsp output 3がtp input 4がtp output
  #python3 src/plot/lobotomy.py  --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1' --mode_num 1
  #python3 src/plot/lobotomy.py  --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/lobotomy.py --read_name func_diff_e20_p20_l10 --model_path func_diff_e20_p20_l10 --device 'cuda:0'

#モデル読み込み
def import_data(args):
  bindes = []
  with open('src/data/'+args.read_name+'_binde.dat','rb') as f:
    bindes = pickle.load(f)
    
    model = Model.esn_model.Binde_ESN_Execution_Model(args)
    with open('src/data/'+args.model_path+'_model.pkl', 'rb') as f:
        model = cloudpickle.load(f)
    np.set_printoptions(threshold=np.inf)
  return bindes, model

#拘束条件読み込み
def binde_division(binde):
  binde1=binde[:16,:16]
  binde2=binde[:16,16:32]
  binde3=binde[16:32,:16]
  binde4=binde[16:32,16:32]
  return binde1,binde2,binde3,binde4

def neuron_liq_layer(h_in_x,h_out_x,h_in_y,h_out_y,num,mode):
  sp_in_sort = np.argsort(h_in_x, axis=0)
  sp_out_sort =np.argsort(h_out_x, axis=0)
  tp_in_sort = np.argsort(h_in_y, axis=0)
  tp_out_sort = np.argsort(h_out_y, axis=0)
  if num == 1:
    cut_posison = sp_in_sort
    patt = 'sp_input_neurons_cut'
  elif num == 2:
    cut_posison = sp_out_sort+16
    patt = 'sp_output_neurons_cut'
  elif num == 3:
    cut_posison = tp_in_sort
    patt ='tp_input_neurons_cut'
  elif num == 4:
    cut_posison = tp_out_sort+16
    patt = 'tp_output_neurons_cut'
  #print(cut_posison)
  print(patt)
  if mode == 'lobotomy':
    cut_posison = cut_posison[::-1]
  elif mode == 'lobotomy_reverse':
    cut_posison = cut_posison
  return cut_posison, patt

def neuron_liq_neurons(h_in_x,h_out_x,h_in_y,h_out_y,num,mode):
  sp_mutial = h_in_x
  sp_mutial.extend(h_out_x)
  tp_mutial = h_in_y
  tp_mutial.extend(h_out_y)
  sp_sort = np.argsort(sp_mutial, axis=0)
  tp_sort =np.argsort(tp_mutial, axis=0)
  if num == 1:
    cut_posison = sp_sort
    patt = 'sp_neurons_cut'
  elif num == 2:
    cut_posison = tp_sort
    patt = 'tp_neurons_cut'
  #print(cut_posison)
  print(patt)
  if mode == 'lobotomy':
    cut_posison = cut_posison[::-1]
  elif mode == 'lobotomy_reverse':
    cut_posison = cut_posison
  return cut_posison, patt

def neuron_cut_total(h_in_x,h_out_x,h_in_y,h_out_y,num,mode):
  cut_posison_list = []
  sp_mutial = h_in_x
  sp_mutial.extend(h_out_x)
  tp_mutial = h_in_y
  tp_mutial.extend(h_out_y)
  sp_mutial.extend(tp_mutial)
  all_mutial = sp_mutial
  all_sort = np.argsort(all_mutial, axis=0)
  for i in range(len(all_sort)):
    if all_sort[i] >= 32:
      all_sort[i] -= 32
    cut_posison_list.append(all_sort[i])
  cut_posison = cut_posison_list
  patt = 'best_neurons_cut'
  #print(cut_posison)
  print(patt)
  if mode == 'lobotomy':
    cut_posison = cut_posison[::-1]
  elif mode == 'lobotomy_reverse':
    cut_posison = cut_posison
  return cut_posison, patt

def cut_layer_neurons(args,h_in_x,h_out_x,h_in_y,h_out_y,binde):
  what = args.what
  how = args.how
  mode = args.mode
  mode_num = args.mode_num
  sp_acc_list = []
  tp_acc_list = []
  cut_num = []
  rate = []
  print(how)
  #レイヤー毎の時間空間情報 1~4
  if how == 'layer_cut':
    cut_posison, patt = neuron_liq_layer(h_in_x,h_out_x,h_in_y,h_out_y,mode_num,mode)
  #ニューロン全体での時間空間情報 1~2
  elif how == 'neuron_cut':
    cut_posison, patt = neuron_liq_neurons(h_in_x,h_out_x,h_in_y,h_out_y,mode_num,mode)
  #print(cut_posison)
  #print(h_in_x)
  #ニューロンカット前の精度を算出
  binde1,binde2,binde3,binde4 = binde_division(binde)
  for i in range(len(cut_posison)):
    #結果における精度の評価
    testdata, sp_test, tp_test = inputdata_test
    loss_func = nn.BCEWithLogitsLoss()
    sp_acc,tp_acc,sp_loss,tp_loss = training.test(model,testdata,loss_func,optimizer,sp_test,tp_test,binde1,binde2,binde3,binde4)
    print_acc(i,sp_acc,tp_acc,sp_loss,tp_loss)
    sp_acc_list.append(sp_acc)
    tp_acc_list.append(tp_acc)
    cut_num.append(i)
    #ロボトミー割合の計算
    if how == 'layer_cut':
      rate.append((i)/16)
    elif how == 'neuron_cut':
      rate.append((i)/32)
    #特定のニューロンをカット
    print(f'cut!======={cut_posison[i]}=============')
    cut = torch.zeros(1,32).to(args.device)
    #binde[cut_posison[i]][:] = cut
    binde[:][cut_posison[i]] = cut
    #ニューロンカット後の拘束条件
    binde1,binde2,binde3,binde4 = binde_division(binde)
  #精度の推移のプロット
  plt.figure()
  plt.plot(cut_num,np.array(sp_acc_list)*100,label="spatial information",color="g")
  plt.plot(cut_num,np.array(tp_acc_list)*100,label="temporal information",color="r")
  #plt.plot(rate,sp_acc_list,label="spatial information",color="g")
  #plt.plot(rate,tp_acc_list,label="temporal information",color="r")
  plt.yticks((10,20,30,40,50,60,70,80,90,100))
  #plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
  #plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
  plt.ylim(0,101)
  #plt.xlim(0,1)
  if how == 'layer_cut':
    plt.xlim(0,16)
  elif how == 'neuron_cut':
    plt.xlim(0,32)
  #plt.xlabel('Rate',fontsize=15)
  plt.xlabel('Number of cut neutrons',fontsize=15)
  plt.ylabel('Accuracy(%)',fontsize=15)
  plt.legend(loc=3)
  if args.weight_num == 0:
    print('src/img/lobotomy_'+what+'_eva/'+what+'_eva_lobotomy_'+how+'_'+patt+'.png')
    if mode == 'lobotomy':
      #plt.title(what+'_eva_lobotomy_'+how+'_'+patt)
      plt.savefig('src/img/lobotomy_'+what+'_eva/'+what+'_eva_lobotomy_'+how+'_'+patt+'.png')
      plt.savefig('src/img/lobotomy_'+what+'_eva/'+what+'_eva_lobotomy_'+how+'_'+patt+'.pdf')
    elif mode == 'lobotomy_reverse':
      #plt.title(what+'_eva_lobotomy_'+how+'_'+patt)
      plt.savefig('src/img/lobotomy_'+what+'_eva_reverse/'+what+'_eva_lobotomy_reverse_'+how+'_'+patt+'.png')
    #plt.savefig('src/img/lobotomy/loss_eva_lobotomy_sp.png')
  else:
    print('src/img/lobotomy_'+what+'_eva_/'+what+'_eva_lobotomy_'+how+'_'+patt+'.png')
    print('no_save')

def print_acc(i,sp_acc,tp_acc,sp_loss,tp_loss):
  #結果のプロット
  cut_str = f'cut{i+1}'
  sp_loss_str = f'sp_loss{sp_loss.data:.2f}'
  tp_loss_str = f'tp_loss{tp_loss.data:.2f}'
  sp_accuracy_str = f'sp_accuracy{sp_acc:.2f}'
  tp_accuracy_str = f'tp_accuracy{tp_acc:.2f}'
  print(f'-----{cut_str}----')
  print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')

if __name__ == '__main__':
  #アーグパース
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #モデルと拘束条件をインポート
  bindes, models = import_data(args)
  binde = bindes[-9]
  #binde = bindes[227]
  binde = torch.from_numpy(binde).clone()
  binde = binde.to(args.device)
  binde1,binde2,binde3,binde4 = binde_division(binde)
  model = models[-9]
  #model = models[227]
  #print(binde)
  #相互情報量の分析
  optimizer = torch.optim.Adam
  inputdata_test = inputdata.make_test(args)
  #相互情報量の算出
  training= train.Adam_train(args,model,optimizer,inputdata_test)
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
  
  #相互情報量から拘束条件の再作成

  cut_layer_neurons(args,h_in_x,h_out_x,h_in_y,h_out_y,binde)

