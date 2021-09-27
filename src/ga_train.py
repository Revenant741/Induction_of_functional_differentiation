import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import copy
import matplotlib.pyplot as plt
import csv
import sys
import train
import hessian_train
from input import inputdata
from model import esn_model
from model import simple_model
from my_def import hessianfree
from my_def import Analysis
import pickle
import os
import time
#from torch.utils.tensorboard import SummaryWriter

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=5)
  parser.add_argument('--name', type=str, default='ga_hf_5_Normal', help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--pop', type=int, default=20, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=10, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=100, help='generation')
  parser.add_argument('--gene_length', default=16, help='pop_model_number')
  #test時
  #python3 src/ga_train.py --optimizer Adam --pop 10 --survivor 4 --name 'ga_test'
  #実行
  #python3 src/ga_train.py  --epoch 20 --device 'cuda:0' --name 'func_diff_e20_p20_l10'


#世代における個体の評価
def make_one_gene(args, g, bindes, ind_learn, optimizer, inputdata_test, ind):
  #世代の作成
  print('========')
  print(f'=学習=')
  ga_start_time = time.time()
  #生成した接続構造分学習
  for i in range(args.pop):
    acc = 0
    loss = 0
    if g == 0:
      #第一世代では接続構造を生成
      binde = torch.randint(0, 2, (args.gene_length*2, args.gene_length*2)).to(args.device)
    else:
      #新しい接続構造のみ学習
      binde = torch.from_numpy(bindes[i]).clone().to(args.device)
    #モデルの実行，重み,誤差，精度を返して来る．
    print(f'--個体{i+1}--')
    model = esn_model.Binde_ESN_Execution_Model(args).to(args.device)
    individual = ind_learn(args,model,optimizer,inputdata_test)
    learn_start = time.time()
    model, epochs, sp_accuracys, tp_accuracys, sp_loss_list, tp_loss_list = individual.main(binde[:16,:16],binde[:16,16:32],binde[16:32,:16],binde[16:32,16:32])
    #時間計測
    learn_finish_time = time.time() -learn_start
    print ("-----学習時間:{:.1f}".format(learn_finish_time) + "[sec]-----")
    h_in_x, h_in_y, h_out_x, h_out_y = individual.mutual_info(model,binde[:16,:16],binde[:16,16:32],binde[16:32,:16],binde[16:32,16:32])
    #総合的な精度と誤差を算出
    acc = (sp_accuracys[-1] + tp_accuracys[-1])/2
    loss = (sp_loss_list[-1] + tp_loss_list[-1])/2
    #個体毎に分散を算出
    sp_var =  (np.var(h_in_x)+np.var(h_out_x))
    tp_var =  (np.var(h_in_y)+np.var(h_out_y))
    eva= best_eva(sp_var,tp_var)
    binde = binde.to('cpu').detach().numpy().copy()
    ind.append((acc,loss,eva,binde,sp_accuracys[-1],tp_accuracys[-1],sp_loss_list[-1],tp_loss_list[-1],model,h_in_x, h_in_y, h_out_x, h_out_y))
    del individual
  #時間計測
  ga_finish_time = time.time()-ga_start_time
  print ("-----世代学習の経過時間:{:.1f}".format(ga_finish_time) + "[sec]-----")
  return ind

#分散の評価関数
def best_eva(sp_var,tp_var):
  raito = abs(sp_var-tp_var)
  eva = sp_var+tp_var-raito
  return eva

#選択，生き残る個体を決める関数
def evalution(ind):
  #0で精度、1で誤差，Falseで小さい順，Trueで大きい順
  #誤差
  #ind = sorted(ind, key=lambda x:x[1], reverse=False)
  #機能分化
  ind = sorted(ind, key=lambda x:x[2], reverse=True)
  return ind

#二点交叉
def tow_point_crossover(parent1, parent2,gene_length):
  child1 = copy.deepcopy(parent1)
  for i in range(4):
    r0 = random.randint(0,gene_length*2-1)
    r1 = random.randint(0,gene_length*2-1)
    r2 = random.randint(r1,gene_length*2)
    child1[r0,r1:r2]= parent2[r0,r1:r2]
  return child1

#突然変異
def mutate(parent,gene_length):
  child = copy.deepcopy(parent)
  for i in range(40):
    r1 = random.randint(0, gene_length*2-1)
    r2 = random.randint(0, gene_length*2-1)
    if child[r1][r2] == 0:
      child[r1][r2] = 1
    else:
      child[r1][r2] = 0
  return child

#交配関数
def crossbreed(args,binde,first_pop):
  #次世代の生成，生成個体の数は初代と同じ数
  while len(binde) < first_pop:
    m1 = random.randint(0,len(binde)-1)#親となる個体の決定
    m2 = random.randint(0,len(binde)-1)#親となる個体の決定
    child = tow_point_crossover(binde[m1],binde[m2],args.gene_length)#交叉処理
    #突然変異
    if random.random() < args.mutate_rate:
      m = random.randint(0,len(binde)-1)#突然変異する個体を選択
      child = mutate(binde[m],args.gene_length)
    binde.append(child)
  return binde

#保存用の処理
def for_save(args,SAVE,g,survival ,binde):
  print('========')
  print('=評価=')
  rank = 0
  #初期化(前世代の接続を不正に残さない為)
  binde = []
  SP_A ,TP_A ,SP_L ,TP_L ,VAR,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W = SAVE
  for acc,loss,var,binde1,sp_accuracy,tp_accuracy,sp_loss,tp_loss,models,h_in_x, h_in_y, h_out_x, h_out_y in survival:
    #精度の可視化
    rank += 1
    print(f'-----第{rank}位--精度={acc*100:.1f}%-----')
    #優秀な個体は次世代に持ち越し
    binde.append(binde1)
    #評価用の変数
    SP_A.append(sp_accuracy)
    TP_A.append(tp_accuracy)
    SP_L.append(sp_loss)
    TP_L.append(tp_loss)
    VAR.append(var)
    Models.append(models)
    X_IN.append(h_in_x)
    Y_IN.append(h_in_y)
    X_OUT.append(h_out_x)
    Y_OUT.append(h_out_y)
    G.append(g+1)
    W.append(binde1)
  analysis = Analysis.Analysis(args)
  #重み，結合，精度，誤差，世代を保存
  analysis.ga_save_to_data(Models,SP_A,TP_A,SP_L,TP_L,G,W)
  #1世代の相互情報量の記録
  analysis.save_to_mutual(X_IN, Y_IN, X_OUT, Y_OUT)
  #分散の保存
  analysis.save_to_var(VAR)
  #次の世代に持ち越し
  SAVE =SP_A ,TP_A ,SP_L ,TP_L ,VAR,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W
  return SAVE, binde

def ga_train(args,ind_learn,optimizer,inputdata):
  #時間計測
  total_time = time.time()
  #保存用の変数用意
  SP_A ,TP_A ,SP_L ,TP_L ,VAR,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W = [],[],[],[],[],[],[],[],[],[],[],[]
  SAVE =SP_A ,TP_A ,SP_L ,TP_L ,VAR,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W
  ind = []
  binde = []
  #学習データ
  inputdata_test = inputdata
  first_pop = args.pop
  #遺伝的アルゴリズムの開始
  for g in range(args.generation):
    #世代の作成
    print(f'\n世代{g+1}')
    #個体生成，学習，評価値を保存
    ind = make_one_gene(args, g, binde, ind_learn, optimizer, inputdata_test, ind)
    #評価値順にソート
    ind = evalution(ind)
    #優秀構造のみ残す
    survival = ind[0:args.survivor]
    #次世代の優秀個体として評価値を持ち越し
    ind = survival
    #優秀な構造のみ保持，保存
    SAVE, binde = for_save(args,SAVE,g,survival,binde)
    #次世代の設定
    #交配,新しい構造を生成
    binde = crossbreed(args,binde,first_pop)
    #学習する個体は新しく生成された個体のみ
    args.pop = first_pop-args.survivor
    #新しく評価が必要なもののみ追加
    binde = binde[-args.pop:]
    #合計経過時間
    total_finish = time.time()- total_time
    print ("=====TOTAL_TIME:{:.1f}".format(total_finish) + "[sec]=====")

if __name__ == '__main__': 
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  inputdata = inputdata.make_test(args)

  #最適化関数
  if args.optimizer == 'HessianFree':
    ind_learn = hessian_train.HessianFree_train
    optimizer = hessianfree.HessianFree
  elif args.optimizer == 'Adam':
    ind_learn = train.Adam_train
    optimizer = torch.optim.Adam
    
  #重みと構造の探索関数
  ga_train(args,ind_learn,optimizer,inputdata)

  #test
  # pop = 10 #初期個体 #次世代の個体数
  # ind_learn = train.Adam_train
  # epoch = 5
  # optimizer = torch.optim.Adam
  # generation = 2
  # survivor = 5 #生き残る個体
  # name = 'ga_test'
  # ga_train(pop,ind_learn,epoch,optimizer,generation,survivor,name)
  