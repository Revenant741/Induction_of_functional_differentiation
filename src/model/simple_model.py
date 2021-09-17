import torch
import torch.nn as nn
import math

class RNN_Model(nn.Module):
  def __init__(self,size_in=16, size_middle=32, size_out=6, batch_size=10):
    super().__init__()
    self.size_middle = size_middle
    self.batch_size = batch_size
    self.hx = torch.zeros(batch_size, size_middle).cuda()
    self.res_conect = nn.Dropout(p=0.1)
    self.reservoir = nn.RNNCell(size_in,size_middle)
    self.fc = nn.Linear(size_middle,size_out)

  def forward(self, x):
    self.hx = self.reservoir(x,self.hx)
    out = self.fc(self.hx)
    torch.manual_seed(0)
    self.hx = self.res_conect(self.hx)
    return out
  
  def initHidden(self):
    self.hx = torch.zeros(self.batch_size, self.size_middle).to(self.device) 

class RNN_Execution_Model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_in = 16
    self.size_middle = 32
    self.size_out = 6
    self.batch_size = args.batch
    self.device = args.device
    self.hx = torch.zeros(self.batch_size, self.size_middle).to(self.device)
    #self.res_conect = nn.Dropout(p=0.5)
    self.rnn = nn.RNNCell(self.size_in,self.size_middle).to(self.device) 
    self.fc = nn.Linear(self.size_middle,self.size_out).to(self.device) 

  def forward(self, x, binde1, binde2, binde3, binde4):
    self.hx = self.rnn(x,self.hx)
    #self.hx = self.res_conect(self.hx)
    out = self.fc(self.hx)
    return out,self.hx[:,:16],self.hx[:,16:]
  
  def initHidden(self):
    self.hx = torch.zeros(self.batch_size, self.size_middle).to(self.device) 



