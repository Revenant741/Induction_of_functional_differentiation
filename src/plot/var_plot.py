import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from input import inputdata
from plot import directed_plot as dp
from plot import ga_all_directed_plot as ga_dp
from plot import ga_mutual_info_all as ga_mt
sys.path.append("../src")
import train

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
  #世代ごとの分散の
  #python3 src/plot/var_plot.py --write_name '20epoch/var_gene/var_change'
  #個体毎の分散の分析
  #python3 src/plot/var_plot.py --write_name '20epoch/var_pop/var_change'
  #python3 src/plot/var_plot.py --write_name '20epoch/var_pop2/var_move_change'

def No_binde(size_middle=16):
  binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

def ga_gene_var(args,models,inputdata_test,optimizer):
  ga_all_var = []
  pop = 20
  generation = int(len(models)/pop)
  for i in range(generation):
    #世代毎の相互情報量の初期化
    in_x = []
    in_y = []
    out_x = []
    out_y = []
    if i == 0:
      num = 0
    else:
      num += 20 
    for j in range(pop):
    #世代内の個体を一体づつ持ってくる
      model_num = num+j
      #print(model_num)
      model = models[model_num]
      binde1, binde2, binde3, binde4 = No_binde()
      #相互情報量の分析
      training= train.Adam_train(args,model,optimizer,inputdata_test)
      h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4) 
      #20個体毎に分散を算出する
      in_x.append(h_in_x)
      in_y.append(h_in_y)
      out_x.append(h_out_x)
      out_y.append(h_out_y)
    print(len(in_x))
    #世代における分散の算出
    sp_var =  (np.var(in_x)+np.var(out_x))
    tp_var =  (np.var(in_y)+np.var(out_y))
    #eva= nomal_eva(sp_var,tp_var)
    eva= best_eva(sp_var,tp_var)
    all_var = eva
    ga_all_var.append(all_var)
    print(f'sp_var{sp_var}----tp_var{tp_var}----all_var{all_var}')
  gene = [i+1 for i in range(generation)]
  fig = plt.figure()
  plt.plot(gene,ga_all_var)
  #plt.xlim(0,0.7)
  #plt.ylim(0,0.7)
  print('-------------succes------------')
  plt.savefig('src/img/'+args.write_name+'mutial_info_K.svg')
  plt.savefig('src/img/'+args.write_name+'mutial_info_K.png')
  plt.savefig('src/img/'+args.write_name+'mutial_info_K.pdf')

def all_pop_var(args,models,inputdata_test,optimizer):
  #描画要の変数
  ga_all_var = []
  pop = 20
  generation = int(len(models)/pop)
  for i in range(generation):
    #世代毎の相互情報量の初期化
    if i == 0:
      num = 0
    else:
      num += 20 
    for j in range(pop):
    #世代内の個体を一体づつ持ってくる
      model_num = num+j
      #print(model_num)
      model = models[model_num]
      binde1, binde2, binde3, binde4 = No_binde()
      #相互情報量の分析
      training= train.Adam_train(args,model,optimizer,inputdata_test)
      h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4) 
      #個体毎に分散を算出
      sp_var =  (np.var(h_in_x)+np.var(h_out_x))
      tp_var =  (np.var(h_in_y)+np.var(h_out_y))
      #eva= nomal_eva(sp_var,tp_var)
      eva= best_eva(sp_var,tp_var)
      all_var = eva
      ga_all_var.append(all_var)
      print(f'sp_var{sp_var}----tp_var{tp_var}----all_var{all_var}')
  gene = [i+1 for i in range(generation*pop)]
  fig = plt.figure()
  plt.plot(gene,ga_all_var,alpha= 0.5)
  #plt.scatter(gene,ga_all_var)
  #移動平均の個数
  num=5
  b=np.ones(num)/num
  #移動平均
  move_var=np.convolve(ga_all_var, b, mode='same')
  plt.plot(gene,move_var)
  plt.grid()
  plt.xlim(0,200)
  #plt.ylim(0,0.7)
  x_memori = [i for i in range(0,200,20)]
  plt.xticks(x_memori)
  print('-------------succes------------')
  print(ga_all_var)
  print('-------------sort------------')
  print(np.argsort(ga_all_var))
  plt.savefig('src/img/'+args.write_name+'mutial_info_K.svg')
  plt.savefig('src/img/'+args.write_name+'mutial_info_K.png')
  plt.savefig('src/img/'+args.write_name+'mutial_info_K.pdf')

def nomal_eva(sp_var,tp_var):
  eva = sp_var+tp_var
  return eva

def best_eva(sp_var,tp_var):
  raito = sp_var-tp_var
  eva = sp_var+tp_var-raito
  return eva

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  bindes, models = dp.import_data(args)
  print("finded_model")
  optimizer = torch.optim.Adam
  inputdata_test = inputdata.make_test(args)
  ga_gene_var(args,models,inputdata_test,optimizer)
  #all_pop_var(args,models,inputdata_test,optimizer)



  


    
      












