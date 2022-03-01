from ast import arg
import csv
from statistics import mode
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
  parser.add_argument('--epoch', type=int, default=30)
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
  #python3 src/plot/weight_variation.py --name 'weight_change_tp/neurons'
  #python3 src/plot/weight_variation.py --name 'weight_change_sp/neurons'


def finded_ga_model(args):
  model = Model.esn_model.Binde_ESN_Execution_Model(args)
  with open(args.model_path, 'rb') as f:
      model = cloudpickle.load(f)
  np.set_printoptions(threshold=np.inf)
  model = model[args.model_point]
  return model

def weight_division(model):
  #print(model.state_dict()['binde_esn.w_res1'].shape)
  #print(model.state_dict()['binde_esn.w_res12'].shape)
  #print(model.state_dict()['binde_esn.w_res2'].shape)
  #print(model.state_dict()['binde_esn.w_res21'].shape)
  weight1 = model.state_dict()['binde_esn.w_res1'].tolist()
  weight2 = model.state_dict()['binde_esn.w_res12'].tolist()
  weight3 = model.state_dict()['binde_esn.w_res2'].tolist()
  weight4 = model.state_dict()['binde_esn.w_res21'].tolist()
  weight1_data = np.array(weight1)
  weight2_data = np.array(weight2)
  weight3_data = np.array(weight3)
  weight4_data = np.array(weight4)
  Input_Neurons = np.hstack([weight1_data,weight2_data])
  Output_Neurons = np.hstack([weight3_data,weight4_data])
  ALL_Neurons = np.vstack([Input_Neurons,Output_Neurons])
  return Input_Neurons,Output_Neurons

class HessianFree_sp_only_train(train.Adam_train):
  def __init__(self,args,model,optimizer,inputdata_test):
    super().__init__(args,model,optimizer,inputdata_test)

  def optimizer_set(self,model):
    optimizer = self.optimizer(model.parameters(), use_gnm=True, verbose=True)
    return optimizer

  def train(self,model,traindata,loss_func,optimizer,train_ans,binde1,binde2,binde3,binde4):
    optimizer.zero_grad()
    def closure():
      losses = 0
      #学習データをスライス
      for i in range(traindata.shape[2]):
        step_input = traindata[:10,:16,i]
        out, x_1, x_2 = model(step_input,binde1,binde2,binde3,binde4)
        ans = train_ans[:10,i,:3].type_as(out)
        #print(out[:,:3].shape)
        #print(out[:,:3])
        #print(ans.shape)
        loss = loss_func(out[:,:3],ans)
        losses += loss
      losses.backward(retain_graph=True)
      return losses, out
    optimizer.step(closure, M_inv=None)

class HessianFree_tp_only_train(train.Adam_train):
  def __init__(self,args,model,optimizer,inputdata_test):
    super().__init__(args,model,optimizer,inputdata_test)

  def optimizer_set(self,model):
    optimizer = self.optimizer(model.parameters(), use_gnm=True, verbose=True)
    return optimizer

  def train(self,model,traindata,loss_func,optimizer,train_ans,binde1,binde2,binde3,binde4):
    optimizer.zero_grad()
    def closure():
      losses = 0
      #学習データをスライス
      for i in range(traindata.shape[2]):
        step_input = traindata[:10,:16,i]
        out, x_1, x_2 = model(step_input,binde1,binde2,binde3,binde4)
        #print(out.shape)
        ans = train_ans[:10,i,3:6].type_as(out)
        #print(ans.shape)
        loss = loss_func(out[:,3:6],ans)
        losses += loss
      losses.backward(retain_graph=True)
      return losses, out
    optimizer.step(closure, M_inv=None)

def variation_make_image(args, epochs,Input_Neurons_var_list,Output_Neurons_var_list):
  name = args.name
  point = 'src/img/'
  #更新量のグラフ
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Update')
  plt.plot(epochs, Input_Neurons_var_list, label="Input Neurons update", color='orange')
  plt.plot(epochs, Output_Neurons_var_list, label="Output Neurons update", color='b')
  #plt.ylim(0,0.7)
  plt.legend(loc=0)
  plt.savefig(f''+point+name+'_update.png')

def variation_save_to_data(args, models, Input_Neurons_var_list,Output_Neurons_var_list):
  name = args.name
  point = 'src/data/'
  #torch.save(model.to('cpu').state_dict(), point+name+'_model.pth')
  with open(f''+point+name+'models.pkl', 'wb') as f:
    cloudpickle.dump(models, f)
  with open(f''+point+name+'_Output_Neurons_var_list.csv', 'w') as f:
      writer = csv.writer(f)
      for input_neurons in Input_Neurons_var_list:
          writer.writerow([input_neurons])
  with open(f''+point+name+'_Input_Neurons_var_list.csv', 'w') as f:
      writer = csv.writer(f)
      for output_neurons in Output_Neurons_var_list:
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
  setup = Use_Model(args)
  for i in range(args.epoch):
    #探索後の重みと接続のデータの指定
    model, binde1,binde2,binde3,binde4 = setup.finded_ga_binde()
    #重みを初期化
    #model = Model.esn_model.Binde_ESN_Execution_Model(args).to(args.device)
    #入力データ
    inputdata_test = inputdata.make_test(args)
    #最適化手法
    optimizer = hessianfree.HessianFree
    #モデル設定
    #train = hessian_train.HessianFree_train(args,model,optimizer,inputdata_test)
    #train = HessianFree_sp_only_train(args,model,optimizer,inputdata_test)
    train = HessianFree_tp_only_train(args,model,optimizer,inputdata_test)
    loss_func = nn.BCEWithLogitsLoss()
    #Input NeuronsとOutput Neuronsの重みを保存
    Input_Neurons,Output_Neurons = weight_division(model)
    optimizer = train.optimizer_set(model)
    #モデル保存
    models = np.append(models,model)
    #学習前の重みを保存
    Input_Neurons1,Output_Neurons1 = weight_division(model)
    #モデルの学習
    for iteration in range(8):
      #8回毎に変わる学習データ
      traindata, train_ans = inputdata.make_train(args)
      #学習（10入力される毎に誤差逆伝播）
      train.train(model,traindata,loss_func,optimizer,train_ans,binde1,binde2,binde3,binde4)
    model.initHidden()
    #評価
    testdata, sp_test, tp_test = inputdata_test
    sp_acc, tp_acc, sp_loss, tp_loss = train.test(model,testdata, loss_func,optimizer,sp_test, tp_test,binde1,binde2,binde3,binde4)
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
    #後の重みを保存
    Input_Neurons2,Output_Neurons2 = weight_division(model)
    #重みの増減値を計算
    Input_Neurons_variation = np.abs(Input_Neurons1-Input_Neurons2)
    Output_Neurons_variation = np.abs(Output_Neurons1-Output_Neurons2)
    print (np.sum(Input_Neurons_variation))
    print (np.sum(Output_Neurons_variation))
    #増減値保存
    Input_Neurons_var_list.append(np.sum(Input_Neurons_variation))
    Output_Neurons_var_list.append(np.sum(Output_Neurons_variation))
    #精度の図を描画
    analysis = Analysis.Analysis(args)
    analysis.make_image(epochs, sp_accuracys,sp_loss_list,tp_accuracys, tp_loss_list)
    #精度の結果を保存
    analysis.save_to_data(model, sp_accuracys, sp_loss_list, tp_accuracys, tp_loss_list)
    #増減値の描画
    variation_make_image(args,epochs,Input_Neurons_var_list,Output_Neurons_var_list)
    #増減値の保存
    variation_save_to_data(args, models, Input_Neurons_var_list,Output_Neurons_var_list)




  
  


