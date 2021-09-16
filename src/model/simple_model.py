import torch
import torch.nn as nn
import math

class ESN_Model(nn.Module):
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
    self.hx = torch.zeros(self.batch_size, self.size_middle).cuda()

class RNN_Model(nn.Module):
  def __init__(self,size_in=16, size_middle=32, size_out=6, batch_size=10):
    super().__init__()
    self.size_middle = size_middle
    self.batch_size = batch_size
    self.hx = torch.zeros(batch_size, size_middle).cuda()
    self.rnn = nn.RNNCell(size_in,size_middle)
    self.fc = nn.Linear(size_middle,size_out)

  def forward(self, x):
    self.hx = self.rnn(x,self.hx)
    out = self.fc(self.hx)
    return out
  
  def initHidden(self):
    self.hx = torch.zeros(self.batch_size, self.size_middle).cuda()



