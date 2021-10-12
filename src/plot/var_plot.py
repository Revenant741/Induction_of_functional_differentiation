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
  #python3 src/plot/var_plot.py --write_name 'loss_eva_ga_var_change_g100' --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/var_plot.py --write_name 'func_diff_eva_ga_var_change' --read_name func_diff_e20_p20_l10 --model_path func_diff_e20_p20_l10
def No_binde(size_middle=16):
  binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

def ga_gene_var(args,models,inputdata_test,optimizer):
  ga_all_var = []
  pop = 10
  generation = int(len(models)/pop)
  print(len(models))
  print(generation)
  for i in range(generation):
    #世代毎の相互情報量の初期化
    in_x = []
    in_y = []
    out_x = []
    out_y = []
    if i == 0:
      num = 0
    else:
      num += pop
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
    #print(len(in_x))
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
  plt.xlim(0,100)
  plt.ylim(0,0.03)
  print('-------------succes------------')
  plt.xlabel('Generation',fontsize=15)
  plt.ylabel('Functional differentiation',fontsize=15)
  plt.savefig('src/img/'+args.write_name+'_func_diff_gene.svg')
  plt.savefig('src/img/'+args.write_name+'_func_diff_gene.png')
  plt.savefig('src/img/'+args.write_name+'_func_diff_gene.pdf')

def one_gene_var(args,models,inputdata_test,optimizer):
  fig = plt.figure()
  #描画要の変数
  #1世代用
  ga_one_var = []
  pop = 10
  generation = int(len(models)/pop)
  print(generation)
  pops = [x+1 for x in range(pop)]
  for i in range(0,generation,10):
    #世代毎の相互情報量の初期化
    if i == 0:
      num = 0
    else:
      num += pop
    for j in range(pop):
    #世代内の個体を一体づつ持ってくる
      model_num = num+j
      print(model_num)
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
      ga_one_var.append(all_var)
      #print(f'sp_var{sp_var}----tp_var{tp_var}----all_var{all_var}')
    plt.plot(pops,ga_one_var,label=str(i+10)+"gene")
    plt.legend(loc='upper right')
    #描画が終われば世代の数値がリセット
    ga_one_var = []
  plt.grid()
  plt.ylim(0,0.03)
  print('-------------succes------------')
  print('-------------sort------------')
  plt.savefig('src/img/'+args.write_name+'_func_diff_onegene.svg')
  plt.savefig('src/img/'+args.write_name+'_func_diff_onegene.png')
  plt.savefig('src/img/'+args.write_name+'_func_diff_onegene.pdf')

def all_pop_var(args,models,inputdata_test,optimizer):
  #描画要の変数
  #全て用
  ga_all_var = []
  pop = 10
  generation = int(len(models)/pop)
  gene = [i+1 for i in range(generation*pop)]
  for i in range(generation):
    #世代毎の相互情報量の初期化
    if i == 0:
      num = 0
    else:
      num += pop
    for j in range(pop):
    #世代内の個体を一体づつ持ってくる
      model_num = num+j
      print(model_num)
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
    #移動平均の個数
    ave=10
    b=np.ones(ave)/ave
    #移動平均の描画
    plt.grid()
    plt.legend(loc='upper right')
  #世代全体で保存する場合
  fig = plt.figure()
  plt.xlim(0,1000)
  plt.ylim(0,0.03)
  plt.plot(gene,ga_all_var,alpha= 0.5)
  move_var=np.convolve(ga_all_var, b, mode='same')
  plt.plot(gene,move_var)
  print('-------------succes------------')
  print(ga_all_var)
  print('-------------sort------------')
  print(np.argsort(ga_all_var))
  plt.savefig('src/img/'+args.write_name+'_func_diff_pop.svg')
  plt.savefig('src/img/'+args.write_name+'_func_diff_pop.png')
  plt.savefig('src/img/'+args.write_name+'_func_diff_pop.pdf')

def nomal_eva(sp_var,tp_var):
  eva = sp_var+tp_var
  return eva

#分散の評価関数
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
  #one_gene_var(args,models,inputdata_test,optimizer)



  


    
      












