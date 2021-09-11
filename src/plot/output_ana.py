import numpy as np 
import torch
import torch.nn as nn
import argparse
from input import inputdata
from my_def import hessianfree
from my_def import Analysis
from my_def import Use_Model
import train
import hessian_train
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation
from io import BytesIO
from PIL import Image

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=5)
  parser.add_argument('--name', type=str, default="hf_ga5_epoch200_firstmodel", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_5_Normal_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/hf_ga5_epoch200_bestmodelmodel.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--write_name', default='output_ana', help='savename')
  parser.add_argument('--neuron_start', type=int, default=0, help='use_optimizer')
  parser.add_argument('--neuron_num', type=int,default=16, help='use_optimizer')
  parser.add_argument('--batch_num', type=int,default=1, help='use_optimizer')
  
  #テスト
  #python3 src/atracter.py --neuron_num 2 
def setup_model():
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  #print(args)
  setup = Use_Model.Use_Model(args)
  if args.After_serch == True:
    #探索後の重みと接続のデータの指定
    model, binde1, binde2, binde3, binde4 = setup.finded_binde()
    print("finded_binde")
  else:
    model, binde1, binde2, binde3, binde4 = setup.random_binde()
    print("rondom")
  model = model.to(args.device) 
  inputdata_test = inputdata.make_test(args,patt=9)
  return args,model, binde1, binde2, binde3, binde4, inputdata_test

def output_main(inputdata_test,model):
  loss_func = nn.BCEWithLogitsLoss()
  optimizer = hessianfree.HessianFree(model.parameters(), use_gnm=True, verbose=True)
  #評価
  output_run(args,model,inputdata_test, loss_func,optimizer,binde1,binde2,binde3,binde4)
  model.initHidden()

def output_run(args,model,inputdata_test, loss_func,optimizer,binde1,binde2,binde3,binde4):
  #評価
  testdata, sp_test, tp_test = inputdata_test
  #テストデータをスライス
  out_putdata = []
  x_1_data = []
  x_2_data = []
  #174時間分の出力
  for i in range(testdata.shape[2]):
    step_input = testdata[:10,:16,i]
    #精度の算出
    out,x_1,x_2 = model(step_input,binde1,binde2,binde3,binde4)
    x_1_data.append(x_1.tolist())
    x_2_data.append(x_2.tolist())
    out_putdata.append(out.tolist())
  plt.figure()
  plt.imshow(testdata[1].cpu().numpy())
  plt.savefig(f''+'src/img/'+args.write_name+'_input'+'.png')
  #相互情報量
  training= train.Adam_train(args,model,optimizer,inputdata_test)
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model,binde1,binde2,binde3,binde4)
  args.name = 'for_atracter'
  analysis = Analysis.Analysis(args)
  analysis.mutual_plot(h_in_x,h_in_y,h_out_x,h_out_y)
  s_in_x, lank_in_x = torch.sort(torch.tensor(h_in_x))
  s_in_y, lank_in_y = torch.sort(torch.tensor(h_in_y))
  s_out_x, lank_out_x = torch.sort(torch.tensor(h_out_x))
  s_out_y, lank_out_y = torch.sort(torch.tensor(h_out_y))
  #相互情報量のランキング
  print('input_sp'+str(lank_in_x))
  print('input_tp'+str(lank_in_y))
  print('output_sp'+str(lank_out_x+16))
  print('output_tp'+str(lank_out_y+16))
  #plot用の点群リストを作成
  x_1, t_1 = output_calculation(x_1_data)
  x_2,t_2 = output_calculation(x_2_data) 
  x_out,t_out = output_calculation(out_putdata)
  #全てラベル計算
  #plot_output(args,x_1,t_1,'x_1',sp_test,tp_test)
  #plot_output(args,x_2,t_2,'x_2',sp_test,tp_test)
  #リザバー層のニューロン出力
  plot_sp_tp_output(args,x_1,t_1,'x_1',sp_test,tp_test,h_in_x,h_in_y)
  plot_sp_tp_output(args,x_2,t_2,'x_2',sp_test,tp_test,h_out_x,h_out_y)
  if args.neuron_num >= 6:
    args.neuron_num = 6
  #plot_output(args,x_out,t_out,'x_out',sp_test,tp_test)
  #出力層の別分類
  plot_sp_tp_output(args,x_out,t_out,'out',sp_test,tp_test,h_in_x,h_in_y)
  
  #正解データのplot
  #正解データパターン化
  sp_patt = ans_patt(sp_test)
  tp_patt = ans_patt(tp_test)
  plot_ans_patt(args,sp_patt,t_out,'sp_ans')
  plot_ans_patt(args,tp_patt,t_out,'tp_ans')
  #入力データのパターン
  args.neuron_num = 16
  in_data = testdata.cpu().numpy()
  plot_in_patt(args,in_data,t_out,'x_in_')
  plot_in_patt(args,in_data,t_out,'x_in_')

def plot_in_patt(args,x,t,name):
  in_node = []
  for n in range(-16,0):
    in_node.append(n)
  for i in range(len(x[2])):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    x_patt = x[0,i,:]
    # 軸の設定
    #ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
    ax.set_xlabel('T', fontsize=18)
    ax.set_ylabel('Output', fontsize=18)
    label = str(i+1)
    x_plot = np.array(x_patt)
    x_plot = x_plot.flatten()
    t = np.array(t)
    ax.plot(t, x_plot,label=label)
    ax.legend()
    print('-------------in_plot------------')
    plt.savefig('src/img/'+args.write_name+'_'+name+str(in_node[i])+'.png')

def output_calculation(out_data):
  x = []
  t = []
  for i in range(len(out_data)):
    x.append(out_data[i])
    t.append([i])
  return x, t

def ans_patt(out_data):
  patt1 = []
  patt2 = []
  patt3 = []
  for i in range(len(out_data[1])):
    patt1.append((out_data[args.batch_num,i,0]).item())
    patt2.append((out_data[args.batch_num,i,1]).item())
    patt3.append((out_data[args.batch_num,i,2]).item())
  data_patt = patt1, patt2, patt3
  return data_patt

def plot_output(args,x,t,name,sp_test,tp_test):
  x = np.array(x)
  t = np.array(t)
  for n in range(args.neuron_num):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    time_point = 0
    batch_num = args.batch_num
    while int(x.shape[0]-15) > time_point:
      start_time = time_point
      stop_time = start_time+15
      time_point = stop_time
      label = 'sp'+str(torch.argmax(sp_test[batch_num,start_time,:]).item())+'tp'+str(torch.argmax(tp_test[batch_num,start_time,:]).item())
      if start_time == 0:
        start_time = 1
      x_plot = x[start_time-1:stop_time,batch_num,n]
      x_plot = x_plot.flatten()
      t_plot = t[start_time-1:stop_time,0]
      t_plot = t_plot.flatten()
      ax.plot(t_plot, x_plot,color=color_point,label = label)
    print(f'-------------plot{name}_{n}------------')
    # 軸の設定
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0, fontsize=18)
    ax.set_xlabel('T', fontsize=18)
    ax.set_ylabel('Output', fontsize=18)
    plt.savefig('src/img/'+args.write_name+'_'+name+'n'+str(n)+'.png')

def plot_sp_tp_output(args,x,t,name,sp_test,tp_test,h_x,h_y):
  plot_sp_output(args,x,t,name,sp_test,tp_test,h_x)
  plot_tp_output(args,x,t,name,sp_test,tp_test,h_y)

def plot_sp_output(args,x,t,name,sp_test,tp_test,h_x):
  x = np.array(x)
  t = np.array(t)
  for n in range(args.neuron_num):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    time_point = 1
    batch_num = args.batch_num
    point1 = 0
    point2 = 0
    point3 = 0
    point = point1,point2,point3
    while int(x.shape[0]-15) > time_point:
      start_time = time_point
      stop_time = start_time+15
      time_point = stop_time
      color_point, label = color_check(sp_test,batch_num,start_time)
      if start_time == 0:
        start_time = 1
      x_plot = x[start_time-1:stop_time,batch_num,n]
      x_plot = x_plot.flatten()
      t_plot = t[start_time-1:stop_time,0]
      t_plot = t_plot.flatten()
      point = plot_label_best(ax,point, t_plot, x_plot,label,color_point)
    # 軸の設定
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0, fontsize=18)
    ax.set_xlabel('T', fontsize=18)
    ax.set_ylabel('Output', fontsize=18)
    if name == 'out':
      ax.set_title('sp_plot_'+name+'_'+str(n+32))
      n +=32
    elif name == 'x_2':
      ax.set_title('sp_plot_'+name+'_'+str(n+16)+'_'+str('{:.2f}'.format(h_x[n])))
      n +=16
    else:
      ax.set_title('sp_plot_'+name+'_'+str(n)+'_'+str('{:.2f}'.format(h_x[n])))
    print(f'-------------sp_plot{name}_{n}------------')
    plt.savefig('src/img/'+args.write_name+'_'+name+'n'+str(n)+'_sp'+'.png')

def plot_tp_output(args,x,t,name,sp_test,tp_test,h_y):
  x = np.array(x)
  t = np.array(t)
  for n in range(args.neuron_num):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    time_point = 1
    batch_num = args.batch_num
    point1 = 0
    point2 = 0
    point3 = 0
    point = point1,point2,point3
    while int(x.shape[0]-31) > time_point:
      start_time = time_point
      stop_time = start_time+15
      time_point = stop_time
      color_point,label = color_check(tp_test,batch_num,start_time)
      if start_time == 0:
        start_time = 1
      x_plot = x[start_time-1:stop_time,batch_num,n]
      x_plot = x_plot.flatten()
      t_plot = t[start_time-1:stop_time,0]
      t_plot = t_plot.flatten()
      point = plot_label_best(ax,point, t_plot, x_plot,label,color_point)
    # 軸の設定
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0, fontsize=18)
    ax.set_xlabel('T', fontsize=18)
    ax.set_ylabel('Output', fontsize=18)
    if name == 'out':
      ax.set_title('tp_plot'+str(name)+str(n+32))
      n +=32
    elif name == 'x_2':
      ax.set_title('sp_plot_'+name+'_'+str(n+16)+'_'+str('{:.2f}'.format(h_y[n])))
      n +=16
    else:
      ax.set_title('tp_plot_'+str(name)+'_'+str(n)+'_'+str('{:.2f}'.format(h_y[n])))
    print(f'-------------tp_plot{name}_{n}------------')
    plt.savefig('src/img/'+args.write_name+'_'+name+'n'+str(n)+'_tp'+'.png')

def plot_ans_patt(args,x,t,name):
  fig = plt.figure(figsize=(15,5))
  ax = fig.add_subplot(111)
  print('-------------ans_plot------------')
  print(len(x))
  for i in range(len(x)):
    x_patt = x[i]
    # 軸の設定
    #ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
    ax.set_xlabel('T', fontsize=18)
    ax.set_ylabel('Output', fontsize=18)
    label = str(i+1)
    x_plot = np.array(x_patt)

    x_plot = x_plot.flatten()
    t = np.array(t)
    ax.plot(t, x_plot,label=label)
    ax.legend()
  plt.savefig('src/img/'+args.write_name+'_'+name+'n'+'.png')

def color_check(test,batch_num,time_point):
  if torch.argmax(test[batch_num,time_point,:]).item() == 0:
    colorpoint = 'b'
    label = '1'
  elif torch.argmax(test[batch_num,time_point,:]).item() == 1:
    colorpoint = 'orange'
    label = '2'
  else:
    colorpoint = 'g'
    label = '3'
  return colorpoint, label

def plot_label_best(ax,point,t_plot,x_plot,label,color_point):
  point1, point2, point3 = point
  if label == '1':
    if point1 == 0:
      ax.plot(t_plot, x_plot,label=label,color=color_point)
      point1 +=1
    else:
      ax.plot(t_plot, x_plot,color=color_point)
  elif label == '2':
    if point2 == 0:
      ax.plot(t_plot, x_plot,label=label,color=color_point)
      point2 +=1
    else:
      ax.plot(t_plot, x_plot,color=color_point)
  elif label == '3':
    if point3 == 0:
      ax.plot(t_plot, x_plot,label=label,color=color_point)
      point3 +=1
    else:
      ax.plot(t_plot, x_plot,color=color_point)

  point = point1,point2,point3
  return point

if __name__ == '__main__':  
  args,model, binde1, binde2, binde3, binde4, inputdata = setup_model()
  output_main(inputdata,model)
  #plot_atracter(args,x,y,z)