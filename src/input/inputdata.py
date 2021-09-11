import torch
import matplotlib.pyplot as plt
import math
import random
import os

#データ作成用
def make_pattern(t_long,h_long,patterns):
  sp_patt = []
  tp_patt = []
  ts = torch.arange(0.0, t_long , 1.0)
  cols = [torch.Tensor() for _ in range(h_long)]
  for p in patterns:
    tp, sp = p
    a = -1
    for i in range(h_long):
      if i % sp == 0:
        a *= -1
      data = a * torch.cos(2.0*math.pi * tp *ts)
      cols[i] = torch.cat([cols[i], data])
    for j in range(list(ts.shape)[0]):
      #正解データの作成
      if sp == 2:
        spot = 0
        sp_patt.append(spot)
      elif sp == 4:
        spot = 1
        sp_patt.append(spot)
      elif sp == 8:
        spot = 2
        sp_patt.append(spot)
      #正解データの作成
      if tp == 1/4:
        spot = 0
        tp_patt.append(spot)
      elif tp ==1/8:
        spot = 1
        tp_patt.append(spot)
      elif tp ==1/16:
        spot = 2
        tp_patt.append(spot)
  sp_one_hot = torch.nn.functional.one_hot(torch.tensor(sp_patt), num_classes=3)
  tp_one_hot = torch.nn.functional.one_hot(torch.tensor(tp_patt), num_classes=3)
  patt = torch.cat((sp_one_hot,tp_one_hot),1)
  cols = torch.stack(cols).view(h_long, -1)
  return cols, patt, sp_one_hot, tp_one_hot

def make_seed_patt():
  spatial_patterns = [2, 4, 8]
  temporal_patterns = [1/4, 1/8, 1/16]
  patterns = []
  for sp in spatial_patterns:
    for tp in temporal_patterns:
      patterns.append((tp, sp))
  random.shuffle(patterns)
  return patterns

def make_train(args,t_long=15,h_long=16):
  batch_size = args.batch
  traindata = [torch.Tensor() for _ in range(batch_size)]
  ans = []
  for i in range(batch_size):
    patterns = make_seed_patt()
    #学習データの作成
    change_switch = random.randint(0,1)
    if change_switch == 0:
      trainpatt, train, _, _ = make_pattern(t_long*2,h_long,patterns[0:1])
      traindata[i] = torch.cat([traindata[i], trainpatt])
      ans.append(train)
    else:
      trainpatt, train, _, _ = make_pattern(t_long,h_long,patterns[0:2])
      traindata[i] = torch.cat([traindata[i], trainpatt])
      ans.append(train)
  traindata = torch.stack(traindata).view(batch_size, h_long, -1)
  train_ans = torch.stack(ans).view(batch_size,-1, 6)
  #学習データをgpuに
  traindata = traindata.to(args.device) 
  train_ans = train_ans.to(args.device) 
  return traindata, train_ans

def make_test(args,t_long=30, h_long=16, patt=6):
  batch_size = args.batch
  testdata = [torch.Tensor() for _ in range(batch_size)]
  sp_ans = []
  tp_ans = []
  for i in range(batch_size):
    patterns = make_seed_patt()
    batch_size = args.batch
    #シャッフルの後，評価データの作成
    testpatt,_ , sp_test, tp_test = make_pattern(t_long, h_long, patterns[0:patt])
    testdata[i] = torch.cat([testdata[i], testpatt])
    sp_ans.append(sp_test)
    tp_ans.append(tp_test)
  
  testdata = torch.stack(testdata).view(batch_size, h_long, -1)
  sp_test = torch.stack(sp_ans).view(batch_size,-1, 3)
  tp_test = torch.stack(tp_ans).view(batch_size,-1, 3)
  #評価データをgpuに
  testdata = testdata.to(args.device) 
  sp_test = sp_test.to(args.device) 
  tp_test = tp_test.to(args.device) 
  return testdata, sp_test, tp_test

if __name__ == '__main__':
  train, train_ans = make_train(args='cuda:0')
  testdata, sp_test, tp_test = make_test(args='cuda:0')
  #print(train.shape)
  #print(train_ans.shape)
  #print(testdata.shape)
  #print(sp_test.shape)
  train_name = str(os.path.dirname(os.path.abspath(__file__)))+'/img/train_input'
  test_name = str(os.path.dirname(os.path.abspath(__file__)))+'/img/test_input'
  plt.figure()
  plt.imshow(train[1].cpu().numpy())
  plt.savefig(f''+train_name+'.pdf')
  plt.savefig(f''+train_name+'.svg')
  plt.figure()
  plt.imshow(testdata[1].cpu().numpy())
  plt.savefig(f''+test_name+'.pdf')
  plt.savefig(f''+test_name+'.svg')