import argparse
import torch
import numpy as np
import model as Model
import pickle
import cloudpickle

class Use_Model:
  def __init__(self,args):
    self.args = args
    self.device = args.device

  def random_binde(self,size_middle=16):
    model = Model.esn_model.Binde_ESN_Execution_Model(self.args)
    binde1 = torch.randint(0, 2, (size_middle, size_middle)).to(self.device)  
    binde2 = torch.randint(0, 2, (size_middle, size_middle)).to(self.device)  
    binde3 = torch.randint(0, 2, (size_middle, size_middle)).to(self.device)  
    binde4 = torch.randint(0, 2, (size_middle, size_middle)).to(self.device)  
    return model, binde1, binde2, binde3, binde4

  #一つの個体の情報をインポート
  def finded_one_binde(self):
    model = Model.esn_model.Binde_ESN_Execution_Model(self.args)
    with open(self.args.model_path, 'rb') as f:
        model = cloudpickle.load(f)
    weight = []
    with open(self.args.binde_path,'rb') as f:
      weight = pickle.load(f)
    #print(len(model))
    #print(len(weight))
    np.set_printoptions(threshold=np.inf)
    binde = torch.from_numpy(weight).clone()
    binde = binde.to(self.device)  
    return model,binde[:16,:16],binde[:16,16:32],binde[16:32,:16],binde[16:32,16:32]

  #遺伝的アルゴリズムの結果の中から一つの個体の情報をインポート
  def finded_ga_binde(self):
    model = Model.esn_model.Binde_ESN_Execution_Model(self.args)
    with open(self.args.model_path, 'rb') as f:
        model = cloudpickle.load(f)
    weight = []
    with open(self.args.binde_path,'rb') as f:
      weight = pickle.load(f)
    #print(len(model))
    #print(len(weight))
    np.set_printoptions(threshold=np.inf)
    model = model[self.args.model_point]
    bindes = weight[self.args.model_point]
    binde = torch.from_numpy(bindes).clone()
    binde = binde.to(self.device)  
    return model,binde[:16,:16],binde[:16,16:32],binde[16:32,:16],binde[16:32,16:32]

  def RNN_binde(self,size_middle=16):
    model = Model.simple_model.RNN_Execution_Model(self.args)
    binde1 = torch.randint(1, 2, (size_middle, size_middle)).to(self.device)  
    binde2 = torch.randint(1, 2, (size_middle, size_middle)).to(self.device)  
    binde3 = torch.randint(1, 2, (size_middle, size_middle)).to(self.device)  
    binde4 = torch.randint(1, 2, (size_middle, size_middle)).to(self.device)  
    return model, binde1, binde2, binde3, binde4