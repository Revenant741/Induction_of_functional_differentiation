import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import pickle
import os
import cloudpickle
import argparse
import collections
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
  #python3 src/plot/pruning.py  --read_name ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/pruning.py  --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/pruning.py --read_name func_diff_e20_p20_l10 --model_path func_diff_e20_p20_l10 --device 'cuda:0'

#モデルの読み込み
def import_data(args):
  bindes = []
  with open('src/data/'+args.read_name+'_binde.dat','rb') as f:
    bindes = pickle.load(f)
    model = Model.esn_model.Binde_ESN_Execution_Model(args)
    with open('src/data/'+args.model_path+'_model.pkl', 'rb') as f:
        model = cloudpickle.load(f)
    np.set_printoptions(threshold=np.inf)
  return bindes, model

#重みの読み込み
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

#拘束条件読み込み
def binde_division(binde):
  binde1=binde[:16,:16]
  binde2=binde[:16,16:32]
  binde3=binde[16:32,:16]
  binde4=binde[16:32,16:32]
  return binde1,binde2,binde3,binde4

def print_acc(i,sp_acc,tp_acc,sp_loss,tp_loss):
  #結果のプロット
  cut_str = f'cut{i+1}'
  sp_loss_str = f'sp_loss{sp_loss.data:.2f}'
  tp_loss_str = f'tp_loss{tp_loss.data:.2f}'
  sp_accuracy_str = f'sp_accuracy{sp_acc:.2f}'
  tp_accuracy_str = f'tp_accuracy{tp_acc:.2f}'
  print(f'-----{cut_str}----')
  print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')

def best_sort(ALL_Neurons,binde):
  ALL_Neurons *= binde.tolist()
  ALL_Neurons=ALL_Neurons.flatten()
  ALL_Neurons_sort = np.argsort(ALL_Neurons)
  return ALL_Neurons_sort

def pruning_main(ALL_Neurons,what):
  #変数名の宣言
  sp_acc_list = []
  tp_acc_list = []
  rate = []
  neuron_num = len(ALL_Neurons)
  synapse_num = len(ALL_Neurons)*len(ALL_Neurons)
  #シナプスの値の絶対値化
  ALL_Neurons = np.abs(ALL_Neurons)
  #print(ALL_Neurons)
  #シナプス結合の強さ順のソート
  ALL_Neurons_sort = best_sort(ALL_Neurons,binde)
  #print(ALL_Neurons_sort)
  binde_first = binde.tolist()
  print(binde_first)
  binde_first=itertools.chain.from_iterable(binde_first)
  print(collections.Counter(binde_first))
  #print(f'初期のシナプスの接続数{synapse_num-binde_first.count(0)}')
  #print(f'初期のシナプスの接続率{binde_first.count(0)/synapse_num}')
  #print(f'初期のシナプスの接続非数{binde_first.count(0)}')
  #各ニューロンのシナプス結合を小さい順にそれぞれ一つずつカット
  for i in range(synapse_num):
    #print(ALL_Neurons_sort[i])
    first = int(ALL_Neurons_sort[i]/neuron_num)
    second = ALL_Neurons_sort[i]%neuron_num
    print(f'cut!=======[{first}][{second}]=============')
    binde[first][second] = 0
    #結果における精度の評価
    testdata, sp_test, tp_test = inputdata_test
    loss_func = nn.BCEWithLogitsLoss()
    sp_acc,tp_acc,sp_loss,tp_loss = training.test(model,testdata,loss_func,optimizer,sp_test,tp_test,binde1,binde2,binde3,binde4)
    print_acc(i,sp_acc,tp_acc,sp_loss,tp_loss)
    #精度の記録
    sp_acc_list.append(sp_acc)
    tp_acc_list.append(tp_acc)
    #プルーニング割合の計算
    rate.append((i)/synapse_num)
    print(rate[-1])
  binde_last = binde.tolist()
  binde_last=itertools.chain.from_iterable(binde_last)
  print(collections.Counter(binde_last))
  #print(f'プルーニング後のシナプスの接続数{synapse_num-binde_last.count(0)}')
  #print(f'プルーニング後のシナプスの接続率{binde_last.count(0)/synapse_num}')
  #print(f'プルーニング後のシナプスの接続非数{binde_last.count(0)}')
  #精度の推移のプロット
  plt.figure()
  plt.plot(rate,sp_acc_list,label="spatial information",color="g")
  plt.plot(rate,tp_acc_list,label="temporal information",color="r")
  plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
  plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
  plt.ylim(0,1)
  plt.xlim(0,1)
  plt.xlabel('Rate',fontsize=15)
  plt.ylabel('Accuracy(%)',fontsize=15)
  plt.legend(loc=3)
  plt.title('synapse_pruning')
  plt.savefig('src/img/'+what+'_eva_pruning/'+what+'_pruning.png')

if __name__ == '__main__':
  #アーグパース
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #モデルと拘束条件をインポート
  bindes, models = import_data(args)
  binde = bindes[999]
  binde = torch.from_numpy(binde).clone()
  binde = binde.to(args.device)
  binde1,binde2,binde3,binde4 = binde_division(binde)
  model = models[999]
  #モデルの用意
  optimizer = torch.optim.Adam
  inputdata_test = inputdata.make_test(args)
  training= train.Adam_train(args,model,optimizer,inputdata_test)
  #重みのデータの用意
  weight1_data,weight2_data,weight3_data,weight4_data,ALL_Neurons = weight_division(model)
  #重みデータとモデルを用いてプルーニング
  what = 'loss'
  #what = 'func_diff'
  pruning_main(ALL_Neurons,what)

