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
from my_def import import_data
import pickle
import os
import time
import ga_train as ga
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
  parser.add_argument('--start_gene', default=1, help='restart_gene_number')
  parser.add_argument('--read_name', default='ga_hf_20_1', help='import_name')
  #実行
  #python3 src/ga_train_restart.py  --epoch 20 --device 'cuda:0' --name 'ga_hf_20_re1_' --read_name 'ga_hf_20epoch_1/ga_hf_20_1' 

def ga_train(args,ind_learn,optimizer,inputdata):
  total_time = time.time()
  SP_A ,TP_A ,SP_L ,TP_L ,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W = [],[],[],[],[],[],[],[],[],[],[]
  SP_A ,TP_A ,SP_L ,TP_L,Models,G,W = import_data.import_data(args,args.read_name,args.read_name,args.read_name)
  X_IN ,Y_IN ,X_OUT, Y_OUT = import_data.import_mutial_info(args.read_name)
  SAVE =SP_A ,TP_A ,SP_L ,TP_L ,X_IN ,Y_IN ,X_OUT, Y_OUT, Models, G, W
  ind = []
  binde = W[args.pop*-1:]
  #交配
  #次世代の生成，生成個体の数は初代と同じ数
  while len(binde) < args.pop:
    m1 = random.randint(0,len(binde)-1)#親となる個体の決定
    m2 = random.randint(0,len(binde)-1)#親となる個体の決定
    child = ga.tow_point_crossover(binde[m1],binde[m2],args.gene_length)#交叉処理
    #突然変異
    if random.random() < args.mutate_rate:
      m = random.randint(0,len(binde)-1)#突然変異する個体を選択
      child = ga.mutate(binde[m],args.gene_length)
    binde.append(child)
  inputdata_test = inputdata
  first_pop = args.pop
  print(SAVE)
  print(len(binde))
  for g in range(args.start_gene,args.generation):
    #世代の作成
    print(f'\n世代{g+1}')
    ind = ga.make_one_gene(args, g, binde, ind_learn, optimizer, inputdata_test, ind)
    #初期化(前世代の接続を不正に残さない為)
    binde = []
    #エリートを選択
    ind = ga.evalution(ind)
    #選択,indの初期化
    survival = ind[0:args.survivor]
    ind = survival
    #記録用の処理
    SAVE, binde = ga.for_save(args,SAVE,g,survival,binde)
    #交配
    #次世代の生成，生成個体の数は初代と同じ数
    while len(binde) < first_pop:
      m1 = random.randint(0,len(binde)-1)#親となる個体の決定
      m2 = random.randint(0,len(binde)-1)#親となる個体の決定
      child = ga.tow_point_crossover(binde[m1],binde[m2],args.gene_length)#交叉処理
      #突然変異
      if random.random() < args.mutate_rate:
        m = random.randint(0,len(binde)-1)#突然変異する個体を選択
        child = ga.mutate(binde[m],args.gene_length)
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
  