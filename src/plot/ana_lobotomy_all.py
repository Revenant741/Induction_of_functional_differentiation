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
from plot import weight_conect_heat_map as conectome
from plot import lobotomy
from plot import ga_mutual_info_all as ga_mt
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
  #python3 src/plot/lobotomy.py  --read_name ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/lobotomy.py --read_name func_diff_e20_p20_l10 --model_path func_diff_e20_p20_l10 --device 'cuda:0'
  #python3 src/plot/ana_lobotomy_all.py  --write_name 'lobotomy_ana/lobotomy' --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'

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

def cut_layer_neurons(h_in_x,h_out_x,h_in_y,h_out_y,mode_num,what,mode):
  sp_acc_list = []
  tp_acc_list = []
  cut_num = []
  rate = []
  cut_posison, patt = lobotomy.neuron_liq_neurons(h_in_x,h_out_x,h_in_y,h_out_y,mode_num,mode)
  #print(cut_posison)
  #print(h_in_x)
  for i in range(len(cut_posison)):
    #結果における精度の評価
    testdata, sp_test, tp_test = inputdata_test
    loss_func = nn.BCEWithLogitsLoss()
    sp_acc,tp_acc,sp_loss,tp_loss = training.test(model,testdata,loss_func,optimizer,sp_test,tp_test,binde1,binde2,binde3,binde4)
    lobotomy.print_acc(i,sp_acc,tp_acc,sp_loss,tp_loss)
    #相互情報量の分析
    h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
    args.write_name = 'lobotomy_ana/lobotomy'+str(i)
    ga_mt.plot_mutial_data(args,i,h_in_x,h_in_y,h_out_x,h_out_y)
    #コネクトームの分析と描画
    weight1_data,weight2_data,weight3_data,weight4_data,ALL_Neurons = conectome.weight_division(model)
    #重みの値に拘束条件を付与
    ALL_Neurons= conectome.plus_binde(binde.tolist(),ALL_Neurons)
    sns.heatmap(ALL_Neurons, cmap=cm.jet,vmax=1.5, vmin=-1.5)
    plt.xlabel("Neuron_Number")
    plt.ylabel("Conect_Number")
    plt.savefig('src/img/'+args.write_name+'_weight_binde_ALL_heat_NO'+str(i)+'.png')
    #特定のニューロンをカット
    print(f'cut!======={cut_posison[i]}=============')
    binde[:][cut_posison[i]] = 0
    #描画用のリスト
    sp_acc_list.append(sp_acc)
    tp_acc_list.append(tp_acc)
    print(sp_acc_list)
    cut_num.append(i)
    #ロボトミー割合の計算
    rate.append((i)/32)
    #精度の推移のプロット
    plt.figure()
    plt.plot(cut_num,sp_acc_list,label="spatial information",color="g")
    plt.plot(cut_num,tp_acc_list,label="temporal information",color="r")
    #plt.plot(rate,sp_acc_list,label="spatial information",color="g")
    #plt.plot(rate,tp_acc_list,label="temporal information",color="r")
    #plt.yticks((10,20,30,40,50,60,70,80,90,100))
    plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
    #plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
    #plt.xticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
    plt.ylim(0,1)
    #plt.xlim(0,1)
    plt.xlim(0,32)
    plt.xlabel('Rate',fontsize=15)
    plt.ylabel('Accuracy(%)',fontsize=15)
    plt.legend(loc=3)
    plt.title(what+'_eva_lobotomy_'+patt)
    plt.savefig('src/img/lobotomy_ana/'+what+'_eva_lobotomy_'+patt+'NO'+str(i)+'.png')

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
  training= train.Adam_train(args,model,optimizer,inputdata_test)
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
  mode_num = 2
  #what = 'loss'
  mode = 'lobotomy'
  what = 'func_diff'
  #相互情報量から拘束条件の再作成
  cut_layer_neurons(h_in_x,h_out_x,h_in_y,h_out_y,mode_num,what,mode)

