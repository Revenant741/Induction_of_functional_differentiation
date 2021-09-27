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
  #python3 src/plot/directed_plot.py --neuron_start 0 --neuron_num 16 --write_name 'd_ga_hf_20'

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

#重み読み込み
def weight_division(model):
  #print(model.state_dict()['binde_esn.w_in'].shape)
  #print(model.state_dict()['binde_esn.w_res1'].shape)
  #print(model.state_dict()['binde_esn.w_res12'].shape)
  #print(model.state_dict()['binde_esn.w_res2'].shape)
  #print(model.state_dict()['binde_esn.w_res21'].shape)
  #print(model.state_dict()['binde_esn.fc'].shape)
  weight0 = model.state_dict()['binde_esn.w_in'].tolist()
  weight1 = model.state_dict()['binde_esn.w_res1'].tolist()
  weight12 = model.state_dict()['binde_esn.w_res12'].tolist()
  weight2 = model.state_dict()['binde_esn.w_res2'].tolist()
  weight21 = model.state_dict()['binde_esn.w_res21'].tolist()
  weight_fc = model.state_dict()['binde_esn.fc'].tolist()
  weight0_data = np.array(weight0)
  weight1_data = np.array(weight1)
  weight12_data = np.array(weight12)
  weight2_data = np.array(weight2)
  weight21_data = np.array(weight21)
  weight_fc_data = np.array(weight_fc)
  return weight0_data,weight1_data,weight12_data,weight2_data,weight21_data,weight_fc_data

#拘束条件読み込み
def binde_division(binde):
  binde1=binde[:16,:16]
  binde2=binde[:16,16:32]
  binde3=binde[16:32,:16]
  binde4=binde[16:32,16:32]
  return binde1,binde2,binde3,binde4

#エッジ作成関数,重みをノードの色にする場合
def make_edge_color(args,binde,weight,direct,n,k):
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if binde[i][j] == 1:
        if 1 <= weight[i][j]:
          direct.append((i+n,j+k, {"color" : "red"}))
        elif weight[i][j] <= -1:
          direct.append((i+n,j+k, {"color" : "blue"}))
          pass
  return direct

#エッジ作成関数,重みを太さにしたい従来手法の区分け
def make_edge_before(args,binde,weight,direct,n,k):
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if binde[i][j] == 1:
        if 0 < weight[i][j] and weight[i][j] <= 0.4:
          direct.append((i+n,j+k,1))
        elif 0.4 < weight[i][j] and weight[i][j] <= 0.8:
          direct.append((i+n,j+k,2))
        elif 0.8 < weight[i][j]:
          direct.append((i+n,j+k,3))
  #print(len(direct))
  #print(direct)
  #print(edge_width)
  return direct

#エッジ作成関数,閾値を自分で決定した上で重みを太さにしたい提案手法の区分け
def make_edge_point(args,binde,weight,direct,n,k):
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if 0.75 <= weight[i][j]:
        direct.append((i+n,j+k,1))
      elif weight[i][j] <= -0.75:
        direct.append((i+n,j+k,2))
        pass
        
  #print(len(direct))
  #print(direct)
  #print(edge_width)
  return direct

#エッジ作成関数,閾値を平均値にした上で重みを太さにしたい提案手法の区分け
def make_edge_median(args,binde,weight,direct,n,k):
  weight_point = abs(weight)
  #print(weight_point)
  weight_point = itertools.chain.from_iterable(weight_point)
  median = statistics.median(weight_point)
  print(f'mdedian{median}')
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if binde[i][j] == 1:
        if median <= weight[i][j]:
          direct.append((i+n,j+k,1))
        elif weight[i][j] <= -1*median:
          direct.append((i+n,j+k,2))
          pass
  #print(len(direct))
  #print(direct)
  #print(edge_width)
  return direct

#ノードにおいて最も重要な重みのみ描画する
def make_edge(args,binde,weight,direct,n,k):
  best_weight = 0
  weight_abs = abs(weight)
  #print(weight_abs.shape)
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,args.neuron_num):
      if best_weight < weight_abs[i][j]:
        best_weight = weight_abs[i][j]
        if best_weight != 0:
          best_point_i=i
          best_point_j=j
    direct.append((best_point_i+n,best_point_j+k,1))
    #print(best_point_i,best_point_j)
    best_weight = 0
  #print(len(direct))
  #print(direct)
  #print(edge_width)
  return direct

#ノードにおいて最も重要な重みのみ描画する
def make_edge_output(args,binde,weight,direct,n,k):
  best_weight = 0
  weight_abs = abs(weight)
  #print(weight_abs.shape)
  for i in range(args.neuron_start,args.neuron_num):
    for j in range(args.neuron_start,6):
      if best_weight < weight_abs[i][j]:
        best_weight = weight_abs[i][j]
        if best_weight != 0:
          best_point_i=i
          best_point_j=j
    direct.append((best_point_i+n,best_point_j+k,1))
    print(best_point_i,best_point_j)
    best_weight = 0
  #print(len(direct))
  #print(direct)
  #print(edge_width)
  return direct

#Reservoir層の各層の内部結合
def directed_return(args,neuron_node,binde,weight,name):
  direct = []
  if name == '_output_output':
    n = 16
    k = 16
  else:
    n = 0
    k = 0
  direct = make_edge(args,binde,weight,direct,n,k)
  #print(direct)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(neuron_node)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------retrun_succes------------')
  agraph.draw('src/img/'+args.write_name+name+'.png', prog='dot')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  return direct

#Reservoir層の各層の順方向、逆方向の結合
def directed_forward(args,input_node,output_node,binde,weight,name):
  direct = []
  if name == '_input_output':
    n = 0
    k = 16
  else:
    n = 16
    k = 0
  direct = make_edge(args,binde,weight,direct,n,k)
  #print(direct)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(input_node, bipartite=0)
  G.add_nodes_from(output_node, bipartite=1)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------forward_succes------------')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  agraph.draw('src/img/'+args.write_name+name+'.png', prog='dot')
  return direct

def directed_in(args,in_node,input_node,binde,weight,name):
  direct = []
  #args.neuron_num = 16
  n = -16
  k = 0
  direct = make_edge(args,binde,weight,direct,n,k)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(in_node)
  G.add_nodes_from(input_node)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------in_succes------------')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  agraph.draw('src/img/'+args.write_name+name+'.png', prog='dot')
  return direct


#出力層とOutputNeuronの結合
def directed_out(args,output_node,all_binde,fc_node,weight,name):
  direct = []
  #args.neuron_num = 16
  n = 16
  k = 32
  direct = make_edge_output(args,all_binde,weight,direct,n,k)
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(output_node)
  G.add_nodes_from(fc_node)
  G.add_edges_from(direct)
  nx.write_graphml(G, "test.graphml")
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------out_succes------------')
  #ag.node_attr["shape"] = "circle" #  表示方法変更
  agraph.draw('src/img/'+args.write_name+name+'.png', prog='dot')
  #args.neuron_num = 16
  return direct

def directed_all(args,in_node,input_node,output_node,fc_node,direct0,direct1,direct2,direct3,direct4,direct5,name):
  # 有向グラフの作成
  G = nx.MultiDiGraph()
  G.add_nodes_from(in_node)
  G.add_nodes_from(input_node)
  G.add_nodes_from(output_node)
  G.add_nodes_from(fc_node)
  G.add_edges_from(direct0)
  G.add_edges_from(direct1)
  G.add_edges_from(direct2)
  G.add_edges_from(direct3)
  G.add_edges_from(direct4)
  G.add_edges_from(direct5)
  nx.write_graphml(G, "test.graphml") 
  agraph = nx.nx_agraph.to_agraph(G)
  print('-------------all_succes------------')
  #agraph.draw('src/img/'+args.write_name+name+'.png', prog='dot')
  agraph.draw('src/img/'+args.write_name+name+'.pdf', prog='dot')
  agraph.draw('src/img/'+args.write_name+name+'.svg', prog='dot')

def two_directed(args,weight,name):
  input_node = range(16)
  output_node = range(16,32)
  node_color = ["b"] * 16
  node_color.extend(["r"] * 16)

  g = nx.MultiDiGraph()
  g.add_nodes_from(input_node, bipartite=1)
  g.add_nodes_from(output_node, bipartite=0)
  n = 0
  k = 16
  val = 0.01
  for i in range(0,16):
    for j in range(0,16):
      g.add_edge(i+n,j+k,val)
      if 1 <= weight[i][j]:
        g.add_edge(i+n,j+k, color= 'red')
      elif weight[i][j] <= -1:
        g.add_edge(i+n,j+k, color= 'blue')
  A,B = bipartite.sets(g)
  for i in range(0,16):
    for j in range(0,16):
      g.remove_edge(i+n,j+k, val)
  pos = dict()
  pos.update((n,(1,i)) for i,n in enumerate(A))
  pos.update((n,(2,i)) for i,n in enumerate(B))
  nx.draw_networkx(g, pos, node_color=node_color)
  print('-------------two_graph_succes------------')
  plt.savefig('src/img/'+args.write_name+'_'+name+'.png')
  #agraph.draw('src/img/'+args.write_name+name+'.png', prog='neato') 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  bindes, model = import_data(args)
  binde = bindes[-20]
  binde = torch.from_numpy(binde).clone()
  binde = binde.to(args.device)
  weight0_data,weight1_data,weight12_data,weight2_data,weight21_data,weight_fc_data = weight_division(model[-20])
  binde1,binde2,binde3,binde4 = binde_division(binde)
  setup = Use_Model.Use_Model(args)
  optimizer = torch.optim.Adam
  inputdata_test = inputdata.make_test(args)
  #探索後の重みと接続のデータの指定
  print("finded_binde")
  #相互情報量の分析
  training= train.Adam_train(args,model,optimizer,inputdata_test)
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)

  #ノードの準備
  input_node = []
  output_node = []
  fc_node = []
  in_node = []
  color_patt = {"color" : "magenta"}
  color_patt1 = {"color" : "green"}
  color_patt2 = {"color" : "red"}
  color_patt3 = {"color" : "blue"}
  color_patt4 = {"color" : "orange"}
  color_patt5 = {"color" : "cyan"}

  #入力層
  for x in range(-16,0):
    in_node.append((x,color_patt1))

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

  #print(in_node)
  #print(all_binde)
  #数式的二は　b1*w1+ b2*w21  b3*w12+b4*w2
  direct0 = directed_in(args,in_node,input_node,all_binde,weight0_data,'_indata')
  direct1 = directed_return(args,input_node,binde1,weight1_data,'_input_input')
  direct2 = directed_return(args,output_node,binde4,weight2_data,'_output_output')
  direct3 = directed_forward(args,input_node,output_node,binde3,weight12_data,'_input_output')
  direct4 = directed_forward(args,input_node,output_node,binde2,weight21_data,'_output_input')
  direct5 = directed_out(args,output_node,all_binde,fc_node,weight_fc_data,'fc',)
  directed_all(args,in_node,input_node,output_node,fc_node,direct0,direct1,direct2,direct3,direct4,direct5,'all')
  
  #two_directed(args,weight2_data,'tow_input_output')
  #two_directed(args,weight4_data,'tow_output_input')


