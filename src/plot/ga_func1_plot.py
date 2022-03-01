import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
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
  #python3 src/plot/var_plot.py --write_name 'func_diff_eva_ga_var_change' --read_name func_diff_e20_p20_l10 --model_path func_diff_e20_p20_l10

  #最新
  #python3 src/plot/ga_func1_plot.py --write_name 'loss_eva_ga_func_diff1_change_g100' --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'

def No_binde(size_middle=16):
  binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

def all_pop_func_diff(args,models,inputdata_test,optimizer):
  #描画要の変数
  #全て用
  ga_all_func_diff = []
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
      h_in_x = (np.array(h_in_x))
      h_in_y = np.array(h_in_y)
      h_out_x = np.array(h_out_x)
      h_out_y = np.array(h_out_y)
      sp_func1 =  np.sum(np.abs((h_in_x)-(h_in_y))/math.sqrt(2))
      tp_func1 =  np.sum(np.abs((h_out_x)-(h_out_y))/math.sqrt(2))
      #eva= nomal_eva(sp_var,tp_var)
      all_func_diff= sp_func1 + tp_func1
      ga_all_func_diff.append(all_func_diff)
      print(f'sp_var{sp_func1}----tp_var{tp_func1}----all_var{all_func_diff}')
    #移動平均の個数
    ave=10
    b=np.ones(ave)/ave
    #移動平均の描画
    plt.grid()
    plt.legend(loc='upper right')
    #世代全体で保存する場合
    fig = plt.figure()
    plt.xlim(0,1000)
    plt.ylim(0,4)
    plt.xlabel('generated individual number',fontsize=15)
    plt.ylabel('$FD1$',fontsize=15)
    plt.plot(gene[:(i+1)*pop],ga_all_func_diff,alpha= 1,label="speciality Functional differentiation")
    move_var=np.convolve(ga_all_func_diff, b, mode='same')
    #plt.plot(gene[:(i+1)*pop],move_var,label="average")
    plt.legend()
    print('-------------succes------------')
    print(ga_all_func_diff)
    print('-------------sort------------')
    print(np.argsort(ga_all_func_diff))
    plt.savefig('src/img/'+args.write_name+'.svg')
    plt.savefig('src/img/'+args.write_name+'.png')
    plt.savefig('src/img/'+args.write_name+'.pdf')

def nomal_eva(sp_var,tp_var):
  eva = sp_var+tp_var
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
  all_pop_func_diff(args,models,inputdata_test,optimizer)




  


    
      












