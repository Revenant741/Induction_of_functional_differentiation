from ast import arg
import csv
from statistics import mode
from traceback import print_tb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 
import torch
import random
import argparse
import torch.nn as nn
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_def import Analysis
import hessian_train
from my_def import hessianfree
from my_def.Use_Model import Use_Model
import model as Model
from input import inputdata
import train
import cloudpickle

#重みの変動値をInput NeuronsとOutput Neuronsで分けて算出
def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:1", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=1)
  #parser.add_argument('--name', type=str, default="hf_ga5_epoch200_firstmodel", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20_model.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--write_name', default='MI', help='savename')
  parser.add_argument('--neuron_start', type=int, default=-20, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--batch_num', type=int,default=1, help='use_optimizer')
  parser.add_argument('--name', type=str, default='adam_test', help='save_file_name')
  #python3 src/plot/ana_grad.py --name 'grad/sp_grad'

def finded_ga_model(args):
  model = Model.esn_model.Binde_ESN_Execution_Model(args)
  with open(args.model_path, 'rb') as f:
      model = cloudpickle.load(f)
  np.set_printoptions(threshold=np.inf)
  model = model[args.model_point]
  return model

def grad_save_to_data(args,grad_input_list,grad_output_list):
  name = args.name
  point = 'src/data/'
  with open(f''+point+name+'_Input_Neurons_grad_list.csv', 'w') as f:
      writer = csv.writer(f)
      for input_neurons in grad_input_list:
          writer.writerow([input_neurons])
  with open(f''+point+name+'_Output_Neurons_grad_list.csv', 'w') as f:
      writer = csv.writer(f)
      for output_neurons in grad_output_list:
          writer.writerow([output_neurons])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #データ保存用のリスト
  Input_Neurons_var_list = []
  Output_Neurons_var_list = []
  epochs = []
  models = []
  #精度の保存パラメータ
  sp_loss_list = []
  tp_loss_list = []
  sp_accuracys = []
  tp_accuracys = []
  #微分値の保存
  sp_grad_input_list = []
  sp_grad_output_list = []
  tp_grad_input_list = []
  tp_grad_output_list = []
  #setup
  setup = Use_Model(args)
  #入力データ
  inputdata_test = inputdata.make_test(args)
  #評価データ
  testdata, sp_test, tp_test = inputdata_test
  #探索後の重みと接続のデータの指定
  model, binde1,binde2,binde3,binde4 = setup.finded_ga_binde()
  sp_right_ans = 0
  tp_right_ans = 0
  sp_total_loss = 0
  tp_total_loss = 0
  #誤差関数
  loss_func = nn.BCEWithLogitsLoss()
  #テストデータをスライス
  #spにおける実験
  for i in range(testdata.shape[2]):
    step_input = testdata[:10,:16,i]
    #精度の算出
    out,x_1,x_2 = model(step_input,binde1,binde2,binde3,binde4)
    sp_out = out[:,:3]
    tp_out = out[:,3:6]
    sp_ans = sp_test[:,i,:].type_as(sp_out)
    tp_ans = tp_test[:,i,:].type_as(tp_out)
    ans_point = torch.max(sp_ans,1)[1]
    #空間情報量におけるｙの微分処理
    for j in range(10):
      sp_out[j][ans_point[j]].backward(retain_graph=True)
      #print('point'+str(j))
      #print(sp_out[j][ans_point[j]])
      dy_dx3 = model.binde_esn.fc.grad
      dy_dx2= torch.tanh(dy_dx3)
      dy_dx1 = torch.mul(model.binde_esn.w_res12.grad,binde3)+model.binde_esn.b_res12.grad+torch.mul(model.binde_esn.w_res2.grad,binde4)+model.binde_esn.b_x2.grad
      print(f'sp_input',torch.sum(dy_dx1))
      print(f'sp_output',torch.sum(dy_dx2))
      #print(dy_dx2.shape)
      #print(dy_dx1.shape)
      sp_grad_input_list.append(int(torch.sum(dy_dx1)))
      sp_grad_output_list.append(int(torch.sum(dy_dx2)))
    sp_right = torch.max(sp_out,1)[1].eq(torch.max(sp_ans,1)[1]).sum().item()
    tp_right = torch.max(tp_out,1)[1].eq(torch.max(tp_ans,1)[1]).sum().item()
    sp_right_ans += sp_right
    tp_right_ans += tp_right
    #誤差の算出
    sp_ans = sp_ans.type_as(sp_out)
    tp_ans = tp_ans.type_as(tp_out)
    sp_loss = loss_func(sp_out,sp_ans)
    tp_loss = loss_func(tp_out,tp_ans)
    sp_total_loss += sp_loss
    tp_total_loss += tp_loss
  #誤差
  sp_loss = sp_total_loss.data/(list(sp_test.shape)[1])
  tp_loss = tp_total_loss.data/(list(tp_test.shape)[1])
  #精度
  sp_acc = sp_right_ans/(list(sp_test.shape)[1]*10)
  tp_acc = tp_right_ans/(list(tp_test.shape)[1]*10)
  model.initHidden()
  #表示用
  epoch_str = f'epoch{i+1}'
  sp_loss_str = f'sp_loss{sp_loss.data:.2f}'
  tp_loss_str = f'tp_loss{tp_loss.data:.2f}'
  sp_accuracy_str = f'sp_accuracy{sp_acc:.2f}'
  tp_accuracy_str = f'tp_accuracy{tp_acc:.2f}'
  print(f'-----{epoch_str}----')
  print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')
  epochs.append(i+1)
  sp_loss_list.append(sp_loss.data.item())
  tp_loss_list.append(tp_loss.data.item())
  sp_accuracys.append(sp_acc)
  tp_accuracys.append(tp_acc)
  print(sum(sp_grad_input_list))
  print(sum(sp_grad_output_list))
  print(sum(tp_grad_input_list))
  print(sum(tp_grad_output_list))
  grad_save_to_data(args,sp_grad_input_list,sp_grad_output_list)
  #grad_save_to_data(args,tp_grad_input_list,tp_grad_output_list)  


