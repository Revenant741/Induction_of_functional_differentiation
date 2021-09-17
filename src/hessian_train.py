import argparse
from input import inputdata
from my_def import hessianfree
from my_def import Analysis
from my_def import Use_Model
import train

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--epoch', type=int, default=180)
  parser.add_argument('--name', type=str, default="hf_ga_best", help='save_file_name')
  parser.add_argument('--batch', type=int,default=10, help='batch_size')
  parser.add_argument('--binde_path', type=str, default='src/data/ga_hf_5_binde.dat', help='import_file_name_of_binde')
  parser.add_argument('--model_path', type=str, default='src/data/ga_hf_5_model.pkl', help='import_file_name_model')
  parser.add_argument('--After_serch', type=bool, default=False, help='Use_after_serch_parameter?')
  parser.add_argument('--model_point', type=int, default=-10, help='Use_after_serch_parameter_point')
  parser.add_argument('--optimizer', default='HessianFree', help='use_optimizer')
#python3 src/hessian_train.py --name 'ga_hf_5_best_train' --binde_path 'src/data/ga_hf_5_binde.dat' --model_path 'src/data/ga_hf_5_model.pkl'
#python3 src/hessian_train.py --name 'loss_eva_bestloss' --binde_path 'src/data/ga_hf_loss_e20_p20_l10/ga_hf_pop_20_binde.dat' --model_path 'src/data/ga_hf_loss_e20_p20_l10/ga_hf_pop_20_model.pkl'
#python3 src/hessian_train.py --name 'RNN_ana_c0' --device "cuda:0"
class HessianFree_train(train.Adam_train):
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
        ans = train_ans[:10,i,:6].type_as(out)
        loss = loss_func(out,ans)
        losses += loss
      losses.backward(retain_graph=True)
      return losses, out
    optimizer.step(closure, M_inv=None)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  epoch = 1
  setup = Use_Model.Use_Model(args)
  inputdata_test = inputdata.make_test(args)
  if args.optimizer == 'HessianFree':
    optimizer = hessianfree.HessianFree
  #探索後の構造分析を行うか否か
  After_serch = args.After_serch
  #After_serch = False
  if After_serch == True:
    #探索後の重みと接続のデータの指定
    model, binde1, binde2, binde3, binde4 = setup.finded_ga_binde()
    print("finded_binde")
  else:
    #通常の拘束条件付きESNモデルでの学習
    #model, binde1, binde2, binde3, binde4 = setup.random_binde()
    #RNNモデルでの学習
    model, binde1, binde2, binde3, binde4 = setup.RNN_binde()
    print("rondom")
  model = model.to(args.device) 
  training= HessianFree_train(args,model,optimizer,inputdata_test)
  model ,epochs, sp_accuracys, tp_accuracys, sp_loss_list, tp_loss_list = training.main(binde1,binde2,binde3,binde4)
  analysis = Analysis.Analysis(args)
  analysis.make_image(epochs,sp_accuracys, sp_loss_list, tp_accuracys, tp_loss_list)
  analysis.save_to_data(model, sp_accuracys, sp_loss_list, tp_accuracys, tp_loss_list)
  #相互情報量の分析
  h_in_x, h_in_y, h_out_x, h_out_y = training.mutual_info(model.to(args.device),binde1,binde2,binde3,binde4)
  analysis.save_to_mutual(h_in_x,h_in_y,h_out_x,h_out_y)
  analysis.mutual_plot(h_in_x,h_in_y,h_out_x,h_out_y)
