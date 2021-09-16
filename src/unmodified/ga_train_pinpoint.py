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
  parser.add_argument('--pop', type=int, default=200, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=20, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=100, help='generation')
  parser.add_argument('--gene_length', default=16, help='pop_model_number')
  parser.add_argument('--serching', type=bool, default=True, help='Use_serching_parameter?')
  #test時
  #src/ga_train.py --optimizer Adam --pop 10 --survivor 4 --name 'ga_test'
  #実行中
  #python3 src/ga_train_pinpoint.py  --epoch 20 --device 'cuda:1' --name 'ga_pinpoint_output_only/ga_hf_output_only_20'

#世代における個体の評価
def make_one_gene(args, g, binde, ind_learn, optimizer, inputdata_test, ind):
  bindes = binde
  #世代の作成
  print('========')
  print(f'=学習=')
  ga_start_time = time.time()
  '''
  #初期世代における非探索対象のReservoir層の重みを全て固定する場合の処理
  if g == 0:
    A=torch.randint(1, 2, (16, 16))
    B=torch.randint(1, 2, (16, 16))
    C=torch.randint(1, 2, (16, 16))
  '''
  for i in range(args.pop):
    acc = 0
    loss = 0
    if g == 0:
      '''
      #初期世代における非探索対象のReservoir層の重みを全て固定する場合の処理
      D=torch.randint(0, 2, (16, 16))
      bind1=torch.cat((A,B),0)
      bind2=torch.cat((C,D),0)
      binde=torch.cat((bind1,bind2),1).to(args.device) 
      '''
      binde = torch.randint(0, 2, (args.gene_length*2, args.gene_length*2)).to(args.device) 
    else:
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
    acc = (sp_accuracys[-1] + tp_accuracys[-1])/2
    loss = (sp_loss_list[-1] + tp_loss_list[-1])/2
    binde = binde.to('cpu').detach().numpy().copy()
    ind.append((acc,loss,binde,sp_accuracys[-1],tp_accuracys[-1],sp_loss_list[-1],tp_loss_list[-1],model,h_in_x, h_in_y, h_out_x, h_out_y))
    del individual
  #時間計測
  ga_finish_time = time.time()-ga_start_time
  print ("-----世代学習の経過時間:{:.1f}".format(ga_finish_time) + "[sec]-----")
  return ind

#選択，生き残る個体を決める関数
def evalution(ind):
  #0で精度、1で誤差，Falseで小さい順，Trueで大きい順
  ind = sorted(ind, key=lambda x:x[1], reverse=False)
  return ind

#二点交叉
def tow_point_crossover(parent1, parent2,gene_length):
  child1 = copy.deepcopy(parent1)
  for i in range(4):
    r0 = random.randint(15,gene_length*2-1)
    r1 = random.randint(15,gene_length*2-1)
    r2 = random.randint(r1,gene_length*2)
    child1[r0,r1:r2]= parent2[r0,r1:r2]
  return child1

#突然変異
def mutate(parent,gene_length):
  child = copy.deepcopy(parent)
  for i in range(40):
    r1 = random.randint(15, gene_length*2-1)
    r2 = random.randint(15, gene_length*2-1)
    if child[r1][r2] == 0:
      child[r1][r2] = 1
    else:
      child[r1][r2] = 0
  return child

#保存用の処理
def for_save(args,SAVE,g,survival ,binde):
  print('========')
  print('=評価=')
  rank = 0
  SP_A ,TP_A ,SP_L ,TP_L ,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W = SAVE
  for acc,loss,binde1,sp_accuracy,tp_accuracy,sp_loss,tp_loss,models,h_in_x, h_in_y, h_out_x, h_out_y in survival:
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
  return SAVE, binde

def ga_train(args,ind_learn,optimizer,inputdata):
  total_time = time.time()
  SP_A ,TP_A ,SP_L ,TP_L ,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W = [],[],[],[],[],[],[],[],[],[],[]
  SAVE =SP_A ,TP_A ,SP_L ,TP_L ,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W
  ind = []
  binde = []
  inputdata_test = inputdata
  first_pop = args.pop
  for g in range(args.generation):
    #世代の作成
    print(f'\n世代{g+1}')
    ind = make_one_gene(args, g, binde, ind_learn, optimizer, inputdata_test, ind)
    #初期化(前世代の接続を不正に残さない為)
    binde = []
    #エリートを選択
    ind = evalution(ind)
    #選択,indの初期化
    survival = ind[0:args.survivor]
    ind = survival
    #記録用の処理
    SAVE, binde = for_save(args,SAVE,g,survival,binde)
    #交配
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
    #次世代の設定
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
    
  #生き残る個体数
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
  