import torch
import argparse
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
  #世代における最高精度個体の機能局在の分析
  #python3 src/plot/func_loca_ana.py --write_name '20epoch/ana_all/hf_20_func_loc'
  #相互情報の分散の最高値個体の機能局在の分析
  #python3 src/plot/func_loca_ana.py --write_name 'loss_eva_dire_and_mutual/loss_eva_dire_and_mutual' --read_name ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/func_loca_ana.py --write_name 'loss_eva_dire_and_mutual/loss_eva_dire_and_mutual_g100' --read_name ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20 --device 'cuda:1'
  #python3 src/plot/func_loca_ana.py --write_name 'func_diff_eva_dire_and_mutual/loss_eva_dire_and_mutual' --read_name ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --model_path ga_hf_loss_e20_p20_l10_c1_g50/ga_hf_pop_20 --device 'cuda:1'

def No_binde(size_middle=16):
  binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

def ga_best_acc_ana(args,optimizer,inputdata_test):
  generation = 10
  for i in range(generation):
    if i == 0:
      num = 0
    else:
      num += 20 
    if num == 80:
      num = 81
    model_num = num
    model = models[model_num]
    binde1, binde2, binde3, binde4 = No_binde()
    #相互情報量の分析
    training= train.Adam_train(args,model,optimizer,inputdata_test)
    h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4) 
    #接続構造の分析
    ga_dp.direct_plot(args,model,binde1,binde2,binde3,binde4,h_in_x, h_in_y, h_out_x, h_out_y,i)
    #相互情報量の図のプロット
    ga_mt.plot_mutial_data(args,i,h_in_x,h_in_y,h_out_x,h_out_y)

def best_func_loca_ana(args,optimizer,inputdata_test):
  num_list = [858,979,999,836,826]
  for num in num_list: 
    model_num = num
    model = models[model_num]
    binde1, binde2, binde3, binde4 = No_binde()
    #相互情報量の分析
    training= train.Adam_train(args,model,optimizer,inputdata_test)
    h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4) 
    #接続構造の分析
    ga_dp.direct_plot(args,model,binde1,binde2,binde3,binde4,h_in_x, h_in_y, h_out_x, h_out_y,num)
    #相互情報量の図のプロット
    ga_mt.plot_mutial_data(args,num,h_in_x,h_in_y,h_out_x,h_out_y)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  bindes, models = dp.import_data(args)
  print("finded_model")
  optimizer = torch.optim.Adam
  inputdata_test = inputdata.make_test(args)
  #ga_best_acc_ana(args,optimizer,inputdata_test)
  best_func_loca_ana(args,optimizer,inputdata_test)














