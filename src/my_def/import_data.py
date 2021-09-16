import numpy as np
import torch
import torch.nn as nn
import random
import copy
import matplotlib.pyplot as plt
import csv
import sys
import train
import hessian_train
from input import inputdata
from model import esn_model
from model import simple_model
import pickle
import os
import cloudpickle
import model as Model
import argparse

def import_data(args,read_name,binde_path,model_path):
  sp_accuracy = []
  tp_accuracy = []
  sp_loss = []
  tp_loss = []
  gene = []
  model = esn_model.Binde_ESN_Execution_Model(args)
  with open('src/data/'+model_path+'_model.pkl', 'rb') as f:
      model = cloudpickle.load(f)
  #cpuに移動
  #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
  binde = []
  with open('src/data/'+binde_path+'_binde.dat','rb') as f:
    binde = pickle.load(f)
  np.set_printoptions(threshold=np.inf)

  with open('src/data/'+read_name+'_sp_acc.csv') as f:
      for row in csv.reader(f):
          sp_accuracy.append(float(row[0]))

  with open('src/data/'+read_name+'_tp_acc.csv') as f:
      for row in csv.reader(f):
          tp_accuracy.append(float(row[0]))
  
  with open('src/data/'+read_name+'_sp_loss.csv') as f:
      for row in csv.reader(f):
          sp_loss.append(float(row[0]))

  with open('src/data/'+read_name+'_tp_loss.csv') as f:
      for row in csv.reader(f):
          tp_loss.append(float(row[0]))
  for g in range(args.start_gene):
    for i in range(args.survivor):
      gene.append(g)
  return sp_accuracy, tp_accuracy, sp_loss, tp_loss, model, gene, binde

def import_mutial_info(read_name):
  h_in_x = []
  h_in_y = []
  h_out_x = []
  h_out_y = []
  with open('src/data/'+read_name+'_h_in_x.csv') as f:
    for row in csv.reader(f):
        h_in_x.append((row[0]))
  with open('src/data/'+read_name+'_h_in_y.csv') as f:
    for row in csv.reader(f):
        h_in_y.append((row[0]))
  with open('src/data/'+read_name+'_h_out_x.csv') as f:
    for row in csv.reader(f):
        h_out_x.append((row[0]))
  with open('src/data/'+read_name+'_h_out_y.csv') as f:
    for row in csv.reader(f):
        h_out_y.append((row[0]))
  return h_in_x, h_in_y, h_out_x, h_out_y

if __name__ == '__main__':
  read_name = 'ga_test'
  binde_path = 'ga_test'
  model_path = 'ga_test'
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  args = parser.parse_args()
  SP_A ,TP_A ,SP_L ,TP_L , Models, G, binde = import_data(args,read_name,binde_path,model_path)
  X_IN ,Y_IN ,X_OUT, Y_OUT = import_mutial_info(read_name)
  print(len(SP_A) ,len(TP_A) ,len(SP_L) ,len(TP_L) , len(Models), len(G), len(binde))
  print(len(X_IN) ,len(Y_IN) ,len(X_OUT), len(Y_OUT))
  binde = np.array(binde)
  for j in range(len(binde)):
    print(str(j+1)+'個体目との比較')
    for i in range(len(binde)):
      print(np.sum(binde[j] == binde[i]))
  #print(SP_A ,TP_A ,SP_L ,TP_L , Models, G, W)
  #print(X_IN ,Y_IN ,X_OUT, Y_OUT)