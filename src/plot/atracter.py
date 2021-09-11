import numpy as np 
import torch
import torch.nn as nn
import argparse
#上位ディレクトリのインポートの為のシステム
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from input.inputdata import make_test
from my_def import hessianfree
from my_def import Analysis
from my_def import Use_Model
sys.path.append("../src")
import train
import hessian_train
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation
from io import BytesIO
from PIL import Image
import train
from my_def import Analysis

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=5)
  parser.add_argument('--name', type=str, default="hf_ga5_epoch200_firstmodel", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_5_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_5_best_trainmodel.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=True, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=0, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
  parser.add_argument('--write_name', default='atracter_clor', help='savename')
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
    model, binde1, binde2, binde3, binde4 = setup.finded_one_binde()
    print("finded_binde")
  else:
    model, binde1, binde2, binde3, binde4 = setup.random_binde()
    print("rondom")
  model = model.to(args.device) 
  inputdata_test = make_test(args,patt=9)
  return args,model, binde1, binde2, binde3, binde4, inputdata_test

def atracter_main(inputdata_test):
  loss_func = nn.BCEWithLogitsLoss()
  optimizer = hessianfree.HessianFree(model.parameters(), use_gnm=True, verbose=True)
  #アトラクタ再構成
  atracter_run(args,model,inputdata_test, loss_func,optimizer,binde1,binde2,binde3,binde4)
  model.initHidden()

def atracter_run(args,model,inputdata_test, loss_func,optimizer,binde1,binde2,binde3,binde4):
  #評価
  testdata, sp_test, tp_test = inputdata_test
  #テストデータをスライス
  out_putdata = []
  x_1_data = []
  x_2_data = []
  #174時間分の出力
  #print(testdata.shape[2])
  for i in range(testdata.shape[2]):
    step_input = testdata[:10,:16,i]
    #精度の算出
    out,x_1,x_2 = model(step_input,binde1,binde2,binde3,binde4)
    #print(out.shape)
    x_1_data.append(x_1.tolist())
    x_2_data.append(x_2.tolist())
    out_putdata.append(out.tolist())
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
  print('output_sp'+str(lank_out_x))
  print('output_tp'+str(lank_out_y))
  #アトラクターの算出
  #print(np.array(x_1_data).shape)
  x_1_x,x_1_y,x_1_z = atracter_calculation(x_1_data) 
  x_2_x,x_2_y,x_2_z = atracter_calculation(x_2_data) 
  x_out_x,x_out_y,x_out_z = atracter_calculation(out_putdata)
  title = True
  #input_neurons
  plot_atracter(args,x_1_x,x_1_y,x_1_z,'x_1',sp_test,tp_test,h_in_x,h_in_y,title)
  #output_neurons
  plot_atracter(args,x_2_x,x_2_y,x_2_z,'x_2',sp_test,tp_test,h_out_x,h_out_y,title)
  if args.neuron_num >= 6:
    args.neuron_num = 6
  title = False
  #output_layer
  plot_atracter(args,x_out_x,x_out_y,x_out_z,'out',sp_test,tp_test,h_in_x,h_out_y,title)
  args.neuron_num = 1
  #input_data
  title = False
  for j in range(16):
    x_in_x,x_in_y,x_in_z = atracter_calculation(testdata[args.batch_num,j,:]) 
    input_patt = x_in_x,x_in_y,x_in_z
    plot_ans(args,input_patt,'input'+str(j),sp_test)
  #ans_data
  sp_ans1_patt, sp_ans2_patt, sp_ans3_patt = ans_patt(sp_test)
  tp_ans1_patt, tp_ans2_patt, tp_ans3_patt = ans_patt(tp_test)
  plot_ans(args,sp_ans1_patt,'sp_ans1',sp_test)
  plot_ans(args,sp_ans2_patt,'sp_ans2',sp_test)
  plot_ans(args,sp_ans3_patt,'sp_ans3',sp_test)
  plot_ans(args,tp_ans1_patt,'tp_ans1',tp_test)
  plot_ans(args,tp_ans2_patt,'tp_ans2',tp_test)
  plot_ans(args,tp_ans3_patt,'tp_ans3',tp_test)
  #相互情報量のランキング
  print('input_sp'+str(lank_in_x))
  print('input_tp'+str(lank_in_y))
  print('output_sp'+str(lank_out_x))
  print('output_tp'+str(lank_out_y))
  
def ans_patt(out_data):
  patt1 = []
  patt2 = []
  patt3 = []
  for i in range(len(out_data[1])):
    patt1.append((out_data[args.batch_num,i,0]).item())
    patt2.append((out_data[args.batch_num,i,1]).item())
    patt3.append((out_data[args.batch_num,i,2]).item())
  x1_ans,y1_ans,z1_ans = atracter_calculation(patt1)
  x2_ans,y2_ans,z2_ans = atracter_calculation(patt2)
  x3_ans,y3_ans,z3_ans = atracter_calculation(patt3)
  ans1_patt = x1_ans,y1_ans,z1_ans
  ans2_patt = x2_ans,y2_ans,z2_ans
  ans3_patt = x3_ans,y3_ans,z3_ans
  return ans1_patt, ans2_patt, ans3_patt

def atracter_calculation(out_data):
  x = []
  y = []
  z = []
  tau = 3
  for i in range(len(out_data)-2*tau):
    x.append(out_data[i])
    y.append(out_data[i+tau])
    z.append(out_data[i+2*tau])
  return x, y, z
  
def plot_ans(args,ans_patt,name,test_patt):
  x, y, z = ans_patt
  x = np.array(x)
  y = np.array(y)
  z = np.array(z)
  fig = plt.figure(figsize=(15,15))
  ax = fig.add_subplot(111, projection='3d')
  time_point = 0
  batch_num = args.batch_num
  while int(x.shape[0]) >= time_point:
    #label = 'neuron'+str(i+1)
    if torch.argmax(test_patt[batch_num,time_point,:]).item() == torch.argmax(test_patt[batch_num,time_point+29,:]).item():
      start_time = time_point
      stop_time = start_time+30
      time_point = stop_time
      label = 'patt'+str(torch.argmax(test_patt[batch_num,start_time,:]).item())
    else:
      start_time = time_point
      stop_time = start_time+15
      time_point = stop_time
      label = 'patt'+str(torch.argmax(test_patt[batch_num,start_time,:]).item())
    x_plot = x[start_time:stop_time]
    y_plot = y[start_time:stop_time]
    z_plot = z[start_time:stop_time]
    ax.plot(x_plot, y_plot, z_plot,label=label)
    #print(time_point)
  #ax.plot(x, y, z,label="ans")
  print('ans'+name)
  print('-------------plot------------')
  # 軸の設定
  ax.view_init(elev=15, azim=90)
  ax.set_title("attractor reconfiguration", fontsize=18)
  ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
  ax.set_xlabel('X', fontsize=18)
  ax.set_ylabel('Y', fontsize=18)
  ax.set_zlabel('Z', fontsize=18)
  plt.savefig('src/img/'+args.write_name+'_'+name+'.png')
  print('-------------gif------------')
  # imagemagickで作成したアニメーションをGIFで書き出す
  images = [render_frame(ax,angle,fig,name) for angle in range(72)]
  images[0].save('src/img/'+args.write_name+'_'+name+'n'+'.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

def plot_atracter(args,x,y,z,name,sp_test,tp_test,h_sp,h_tp,title):
  x = np.array(x)
  y = np.array(y)
  z = np.array(z)
  for n in range(args.neuron_num):
    fig, ax = rendering(args,x,y,z,sp_test,tp_test,n)
    print(name+'_neuron'+str(n))
    print('-------------plot------------')
    # 軸の設定
    ax.view_init(elev=15, azim=90)
    if title == True:
      ax.set_title("n_"+str(n)+"_sp_mutual={:.2f}".format(h_sp[n])+"_tp_mutual={:.2f}".format(h_tp[n]), fontsize=18)
    else:
      ax.set_title("n"+str(n)+"output", fontsize=18)
    #ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
    ax.legend(fontsize=15)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_zlabel('Z', fontsize=18)
    plt.savefig('src/img/'+args.write_name+'_'+name+'n'+str(n)+'.png')
    print('-------------gif------------')
    # imagemagickで作成したアニメーションをGIFで書き出す
    images = [render_frame(ax,angle,fig,name) for angle in range(72)]
    images[0].save('src/img/'+args.write_name+'_'+name+'n'+str(n)+'.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

def rendering(args,x,y,z,sp_test,tp_test,n):
  fig = plt.figure(figsize=(15,15))
  ax = fig.add_subplot(111, projection='3d')
  sp_test = sp_test.to('cpu')
  tp_test = tp_test.to('cpu')
  time_point = 0
  batch_num = args.batch_num
  #print(x.shape[0])
  while int(x.shape[0]) >= time_point:
    #label = 'neuron'+str(i+1)
    if torch.argmax(sp_test[batch_num,time_point,:]).item() == torch.argmax(sp_test[batch_num,time_point+29,:]).item() and torch.argmax(sp_test[batch_num,time_point,:]).item() == torch.argmax(sp_test[batch_num,time_point+29,:]).item():
      start_time = time_point
      stop_time = start_time+30
      time_point = stop_time
      label = 'sp'+str(torch.argmax(sp_test[batch_num,start_time,:]).item())+'tp'+str(torch.argmax(tp_test[batch_num,start_time,:]).item())
    else:
      start_time = time_point
      stop_time = start_time+15
      time_point = stop_time
      label = 'sp'+str(torch.argmax(sp_test[batch_num,start_time,:]).item())+'tp'+str(torch.argmax(tp_test[batch_num,start_time,:]).item())
    x_plot = x[start_time:stop_time,batch_num,n]
    y_plot = y[start_time:stop_time,batch_num,n]
    z_plot = z[start_time:stop_time,batch_num,n]
    #x_plot = x_plot.flatten()
    #y_plot = y_plot.flatten()
    #z_plot = z_plot.flatten()
    # axesに散布図を設定する
    ax.plot(x_plot, y_plot, z_plot,label=label)
    #ax.scatter(x_plot, y_plot, z_plot,label=label)
  return fig, ax

def render_frame(ax,angle,fig,name):
    """data の 3D 散布図を PIL Image に変換して返す"""
    ax.view_init(30, angle*5)
    plt.close()
    # PIL Image に変換
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)

if __name__ == '__main__':  
  args,model, binde1, binde2, binde3, binde4, inputdata = setup_model()
  atracter_main(inputdata)
  #plot_atracter(args,x,y,z)
  
  
  
  