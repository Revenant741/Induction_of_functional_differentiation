import argparse
import torch
import torch.nn as nn
import numpy as np
from input import inputdata
import model as Model
import pickle
import os
import csv
import sklearn.metrics
from my_def import Analysis
from my_def import Use_Model
#from torch.utils.tensorboard import SummaryWriter

#Adamを用いたReservoir層の学習、HF法はこれを継承している、HF法を用いた物はhessian_train.pyを参照
def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=200)
  parser.add_argument('--name', type=str, default='adam_test', help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_5_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_5_model.pkl', help='import_file_name_model')
  parser.add_argument('--After_search', type=bool, default=True, help='Use_after_search_parameter?')
  parser.add_argument('--model_point', type=int, default=-20, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='Adam', help='use_optimizer')
#python3 src/train.py


class Adam_train:
  def __init__(self,args,model,optimizer,dataset):
    self.model = model
    self.epoch_num = args.epoch
    self.optimizer = optimizer
    self.inputdata_test = dataset
    self.args = args

  def optimizer_set(self,model):
    optimizer = self.optimizer(model.parameters())
    return optimizer

  def main(self,binde1,binde2,binde3,binde4):
    epochs = []
    sp_loss_list = []
    tp_loss_list = []
    sp_accuracys = []
    tp_accuracys = []
    model = self.model
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = self.optimizer_set(model)
    for epoch in range(self.epoch_num):
      for iteration in range(8):
        #8回毎に変わる学習データ
        traindata, train_ans = inputdata.make_train(self.args)
        #学習（10入力される毎に誤差逆伝播）
        self.train(model,traindata,loss_func,optimizer,train_ans,binde1,binde2,binde3,binde4)
      model.initHidden()
      #評価
      testdata, sp_test, tp_test = self.inputdata_test
      sp_acc, tp_acc, sp_loss, tp_loss = self.test(model,testdata, loss_func,optimizer,sp_test, tp_test,binde1,binde2,binde3,binde4)
      model.initHidden()
      #表示用
      epoch_str = f'epoch{epoch+1}'
      sp_loss_str = f'sp_loss{sp_loss.data:.2f}'
      tp_loss_str = f'tp_loss{tp_loss.data:.2f}'
      sp_accuracy_str = f'sp_accuracy{sp_acc:.2f}'
      tp_accuracy_str = f'tp_accuracy{tp_acc:.2f}'
      print(f'-----{epoch_str}----')
      print(f'----{sp_accuracy_str} | {sp_loss_str} | {tp_accuracy_str} | {tp_loss_str}----')
      epochs.append(epoch+1)
      sp_loss_list.append(sp_loss.data.item())
      tp_loss_list.append(tp_loss.data.item())
      sp_accuracys.append(sp_acc)
      tp_accuracys.append(tp_acc)
    return model, epochs, sp_accuracys, tp_accuracys, sp_loss_list, tp_loss_list

  def train(self, model,traindata,loss_func,optimizer,train_ans,binde1,binde2,binde3,binde4):
    optimizer.zero_grad()
    losses = 0
    #学習データをスライス
    for i in range(traindata.shape[2]):
      step_input = traindata[:10,:16,i]
      out, x_1, x_2 = model(step_input,binde1,binde2,binde3,binde4)
      ans = train_ans[:10,i,:6].type_as(out)
      loss = loss_func(out,ans)
      losses += loss
    losses.backward(retain_graph=True)
    optimizer.step()

  def test(self, model,testdata,loss_func,optimizer,sp_test,tp_test,binde1,binde2,binde3,binde4):
    sp_right_ans = 0
    tp_right_ans = 0
    sp_total_loss = 0
    tp_total_loss = 0
    #テストデータをスライス
    for i in range(testdata.shape[2]):
      step_input = testdata[:10,:16,i]
      #精度の算出
      out,x_1,x_2 = model(step_input,binde1,binde2,binde3,binde4)
      sp_out = out[:,:3]
      tp_out = out[:,3:6]
      sp_ans = sp_test[:,i,:].type_as(sp_out)
      tp_ans = tp_test[:,i,:].type_as(tp_out)
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
    return sp_acc,tp_acc,sp_loss,tp_loss

  def mutual_info(self,model,binde1,binde2,binde3,binde4):
    sp_ans = []
    tp_ans = []
    in_neurons = []
    out_neurons = []
    h_in_x = []
    h_in_y = []
    h_out_x = []
    h_out_y = []
    testdata, sp_test, tp_test = self.inputdata_test
    #print(testdata.shape)
    #ラベルの調整
    sp_test = sp_test.to('cpu').detach().numpy().copy()
    tp_test = tp_test.to('cpu').detach().numpy().copy()
    #テストデータのtime Step分繰り返す
    for i in range(testdata.shape[2]):
      step_input = testdata[:10,:16,i]
      #精度の算出
      out,x1,x2 = model(step_input,binde1,binde2,binde3,binde4)
      #出力の調整
      x1 = x1.to('cpu').detach().numpy().copy()
      x2 = x2.to('cpu').detach().numpy().copy()
      in_neurons.append(x1)
      out_neurons.append(x2)
      #空間ラベル
      sp_ans.append(np.argmax(sp_test[:,i,:],1))
      #時間ラベル
      tp_ans.append(np.argmax(tp_test[:,i,:],1))
    in_neurons = np.array(in_neurons)
    out_neurons = np.array(out_neurons)
    sp_ans = np.array(sp_ans)
    tp_ans = np.array(tp_ans)
    #print(in_neurons.shape)
    #print(in_neurons[:,:,0].shape)
    #print(sp_ans.shape)
    bins=8
    range_x1=(-1,1)
    for j in range(16):
      n_in = in_neurons[:,:,j]
      n_out = out_neurons[:,:,j]
      _,bins_x1  = np.histogram(n_in.flatten(), bins,range_x1)
      _,bins_x1  = np.histogram(n_out.flatten(), bins,range_x1)
      n_in_mutial = np.digitize(n_in.flatten(), bins_x1)
      n_out_mutial = np.digitize(n_out.flatten(), bins_x1)
      n_in_x_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,sp_ans.flatten())
      n_in_y_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,tp_ans.flatten())
      n_out_x_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,sp_ans.flatten())
      n_out_y_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,tp_ans.flatten())
      h_in_x.append(n_in_x_mutial)
      h_in_y.append(n_in_y_mutial)
      h_out_x.append(n_out_x_mutial)
      h_out_y.append(n_out_y_mutial)
    return h_in_x, h_in_y, h_out_x, h_out_y
  
  def __del__(self):
    pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  setup = Use_Model.Use_Model(args)
  inputdata_test = inputdata.make_test(args)
  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam
  if args.After_search == True:
    model, binde1, binde2, binde3, binde4 = setup.finded_ga_binde()
    print("finded_binde")
  else:
    model, binde1, binde2, binde3, binde4 = setup.random_binde()
    print("rondom")
  model = model.to(args.device)  
  training= Adam_train(args,model,optimizer,inputdata_test)
  model ,epochs, sp_accuracys, tp_accuracys, sp_loss_list, tp_loss_list = training.main(binde1,binde2,binde3,binde4)
  analysis = Analysis.Analysis(args)
  analysis.make_image(epochs, sp_accuracys, tp_accuracys, sp_loss_list, tp_loss_list)
  analysis.save_to_data(model, sp_accuracys, sp_loss_list, tp_accuracys, tp_loss_list)
  #相互情報量の分析
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
  analysis.save_to_mutual(h_in_x,h_in_y,h_out_x,h_out_y)
  analysis.mutual_plot(h_in_x,h_in_y,h_out_x,h_out_y)
