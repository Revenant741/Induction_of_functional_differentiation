  
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import torch
import random
import argparse
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_def.Use_Model import Use_Model
import model as Model
from input import inputdata
import train
import cloudpickle

#全世代分の優秀個体の全てのニューロンにおける相互情報量を算出
def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=5)
  #parser.add_argument('--name', type=str, default="hf_ga5_epoch200_firstmodel", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_20/ga_hf_20_0_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_20/ga_hf_20_0_model.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--write_name', default='MI', help='savename')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--batch_num', type=int,default=1, help='use_optimizer')
#python3 src/plot/ga_mutual_info_all.py --model_point 20 --write_name 'ga_hf_20_'

def No_binde(size_middle=16):
  binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

def finded_ga_model(args):
  model = Model.esn_model.Binde_ESN_Execution_Model(args)
  with open(args.model_path, 'rb') as f:
      model = cloudpickle.load(f)
  weight = []
  np.set_printoptions(threshold=np.inf)
  model = model[args.model_point]
  return model

def gene_all_pop_plot(args,pop,n):
  in_x = []
  in_y = []
  out_x = []
  out_y = []
  for i in range(pop):
    #世代内の個体を一体づつ持ってくる
    args.model_point = i+n
    print(args.model_point)
    optimizer = torch.optim.Adam
    setup = Use_Model(args)
    #探索後の重みと接続のデータの指定
    #model, binde1,binde2,binde3,binde4 = setup.finded_ga_binde()
    #モデルのみインポート
    model = finded_ga_model(args)
    binde1, binde2, binde3, binde4 = No_binde()
    print("finded_binde")
    #相互情報量の分析
    training= train.Adam_train(args,model,optimizer,inputdata_test)
    h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
    in_x.append(h_in_x)
    in_y.append(h_in_y)
    out_x.append(h_out_x)
    out_y.append(h_out_y)
  return in_x,in_y,out_x,out_y

def gene_best_pop_plot(args,pop,n):
  in_x = []
  in_y = []
  out_x = []
  out_y = []
  #世代内の個体を一体づつ持ってくる
  args.model_point = i+n
  print(args.model_point)
  optimizer = torch.optim.Adam
  setup = Use_Model(args)
  #探索後の重みと接続のデータの指定
  #model, binde1,binde2,binde3,binde4 = setup.finded_ga_binde()
  #モデルのみインポート
  model = finded_ga_model(args)
  binde1, binde2, binde3, binde4 = No_binde()
  print("finded_binde")
  #相互情報量の分析
  training= train.Adam_train(args,model,optimizer,inputdata_test)
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
  in_x.append(h_in_x)
  in_y.append(h_in_y)
  out_x.append(h_out_x)
  out_y.append(h_out_y)
  return in_x,in_y,out_x,out_y

def plot_mutial_data(args,gene,in_x,in_y,out_x,out_y):
  fig = plt.figure()
  #plt.scatter(in_x[-60:],in_y[-60:], c='blue',label="input neurons")
  #plt.scatter(out_x[-60:],out_y[-60:], c='red',label="output neurons")
  #plt.scatter(in_x[-60:],in_y[-60:], c='blue')
  #plt.scatter(out_x[-60:],out_y[-60:], c='red')
  #色なしの場合
  #plt.scatter(in_x[-60:],in_y[-60:], c='blue',label="Neurons")
  plt.scatter(in_x[-60:],in_y[-60:], c='blue')
  plt.scatter(out_x[-60:],out_y[-60:], c="orange")
  #plt.xlabel('I_{sp}',fontsize=15)
  #plt.ylabel('I_{tp}',fontsize=15)
  #plt.legend(loc='upper right')
  #plt.legend(fontsize=18)
  plt.xlim(0,0.7)
  plt.ylim(0,0.7)
  #plt.savefig('src/img/'+write_name+'mutial_info.png')
  print('-------------succes------------')
  plt.savefig('src/img/'+args.write_name+'NO'+str(gene)+'mutial_info.png')
  plt.savefig('src/img/'+args.write_name+'NO'+str(gene)+'MI.svg')

if __name__ == '__main__':
  generation = 4
  pop = 20

  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)

  inputdata_test = inputdata.make_test(args)
  n = 0
  for j in range(generation):
    if j == 0:
      n = 0
    else:
      n += 20
    print('generation'+str(j+1))
    #全ての個体の相互情報量の算出
    in_x,in_y,out_x,out_y = gene_all_pop_plot(args,pop,n)
    #世代で最も精度の高い個体の相互情報量の算出
    #in_x,in_y,out_x,out_y = gene_best_pop_plot(args,pop,n)
    #相互情報量の描画
    plot_mutial_data(args,j,in_x,in_y,out_x,out_y)


