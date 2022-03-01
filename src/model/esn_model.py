import torch
import torch.nn as nn
import math
import numpy as np
import random

#普通の拘束条件付きESN
class Normal_Binde_ESN_Model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_in = 16
    self.size_middle = 16
    self.size_out = 6
    self.batch_size = args.batch
    self.device = args.device
    self.binde_esn = Normal_Binde_Reservior(args)

  def forward(self,x,binde1,binde2,binde3,binde4):
    output = self.binde_esn(x,binde1,binde2,binde3,binde4)
    return output

  def initHidden(self):
    self.binde_esn.x_1 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.binde_esn.x_2 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.binde_esn.x_out = torch.zeros(self.batch_size, self.size_middle).to(self.device) 

class Normal_Binde_Reservior(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_in = 16
    self.size_middle = 16
    self.size_out = 6
    self.batch_size = args.batch
    self.device = args.device
    self.x_1 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.x_2 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.x_out = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.w_in = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res1 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res12 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res2 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res21 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.fc = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.size_middle,self.size_out), a=math.sqrt(5)))
    self.b_in = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_x1 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_res12 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_x2 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_res21 = nn.Parameter(torch.Tensor(self.size_middle))
    self.reset_parameters(self.w_in,self.b_in)
    self.reset_parameters(self.w_res1,self.b_x1)
    self.reset_parameters(self.w_res12,self.b_res12)
    self.reset_parameters(self.w_res2,self.b_x2)
    self.reset_parameters(self.w_res21,self.b_res21)

  def forward(self, x, binde1, binde2, binde3, binde4):
    self.x_1 = torch.matmul(x,self.w_in)+self.b_in+torch.matmul(self.x_1,torch.mul(self.w_res1,binde1))+self.b_x1+torch.matmul(self.x_2,torch.mul(self.w_res21,binde2))+self.b_res21
    self.x_2 = torch.matmul(self.x_1,torch.mul(self.w_res12,binde3))+self.b_res12+torch.matmul(self.x_2,torch.mul(self.w_res2,binde4))+self.b_x2
    self.x_1 = torch.tanh(self.x_1)
    self.x_2 = torch.tanh(self.x_2)
    out = torch.matmul(self.x_2,self.fc)
    return out

  def reset_parameters(self, weight,bias):
    #重みの初期値
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    #バイアスの初期値
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)

#内部状態を観測出来るようにした拘束条件付きESN
class Binde_ESN_Execution_Model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_in = 16
    self.size_middle = 16
    self.size_out = 6
    self.batch_size = args.batch
    self.device = args.device
    self.binde_esn = Mutial_Info_Reservior(args)

  def forward(self,x,binde1,binde2,binde3,binde4):
    output, x1, x2 = self.binde_esn(x,binde1,binde2,binde3,binde4)
    return output, x1, x2

  def initHidden(self):
    self.binde_esn.x_1 = torch.zeros(self.batch_size, self.size_middle, requires_grad=True).to(self.device) 
    self.binde_esn.x_2 = torch.zeros(self.batch_size, self.size_middle, requires_grad=True).to(self.device) 
    self.binde_esn.x_out = torch.zeros(self.batch_size, self.size_middle, requires_grad=True).to(self.device) 
    #self.binde_esn.x_1 = torch.zeros(self.batch_size, self.size_middle).to("cuda:0") 
    #self.binde_esn.x_2 = torch.zeros(self.batch_size, self.size_middle).to("cuda:0") 
    #self.binde_esn.x_out = torch.zeros(self.batch_size, self.size_middle).to("cuda:0") 

class Mutial_Info_Reservior(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_in = 16
    self.size_middle = 16
    self.size_out = 6
    self.batch_size = args.batch
    self.device = args.device
    self.x_1 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.x_2 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.x_out = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.w_in = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res1 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res12 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res2 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res21 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.fc = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.size_middle,self.size_out), a=math.sqrt(5)))
    self.b_in = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_x1 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_res12 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_x2 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_res21 = nn.Parameter(torch.Tensor(self.size_middle))
    self.reset_parameters(self.w_in,self.b_in)
    self.reset_parameters(self.w_res1,self.b_x1)
    self.reset_parameters(self.w_res12,self.b_res12)
    self.reset_parameters(self.w_res2,self.b_x2)
    self.reset_parameters(self.w_res21,self.b_res21)

  def forward(self, x, binde1, binde2, binde3, binde4):
    self.x_1 = torch.matmul(x,self.w_in)+self.b_in+torch.matmul(self.x_1,torch.mul(self.w_res1,binde1))+self.b_x1+torch.matmul(self.x_2,torch.mul(self.w_res21,binde2))+self.b_res21
    self.x_2 = torch.matmul(self.x_1,torch.mul(self.w_res12,binde3))+self.b_res12+torch.matmul(self.x_2,torch.mul(self.w_res2,binde4))+self.b_x2
    self.x_1 = torch.tanh(self.x_1)
    self.x_2 = torch.tanh(self.x_2)
    out = torch.matmul(self.x_2,self.fc)
    return out, self.x_1, self.x_2

  def reset_parameters(self, weight,bias):
    #重みの初期値
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    #バイアスの初期値
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


