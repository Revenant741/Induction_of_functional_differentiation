import csv
from os import read
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#純粋な個体精度の結果
accuracy = []
loss = []
accuracy2 = []
loss2 = []
accuracy3 = []
loss3 = []
accuracy4 = []
loss4 = []
read_name = 'ga_hf_20_best_train'

with open('src/data/'+read_name+'_sp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy.append(float(row[0])*100)

with open('src/data/'+read_name+'_tp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy2.append(float(row[0])*100)

'''
with open('data/adam_sp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy3.append(float(row[0]))

with open('data/adam_tp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy4.append(float(row[0]))
'''

epoch =[i+1 for i in range(len(accuracy))]
plt.figure()
#plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
#plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)
plt.yticks((10,20,30,40,50,60,70,80,90,100))
plt.ylim(10, 105)
plt.plot(epoch,accuracy,label="spatial information",color="g")
plt.plot(epoch,accuracy2,label="temporal information",color="r")
plt.legend(loc=4)
plt.savefig('src/img/20epoch_ga_hf_nest_acc.png')
plt.savefig('src/img/20epoch_ga_hf_nest_acc.svg')
'''
with open('data/hessian_sp_loss.csv') as x:
    for row in csv.reader(x):
        loss.append(float(row[0]))

with open('data/hessian_tp_loss.csv') as x:
    for row in csv.reader(x):
        loss2.append(float(row[0]))

with open('data/adam_sp_loss.csv') as x:
    for row in csv.reader(x):
        loss3.append(float(row[0]))

with open('data/adam_tp_loss.csv') as x:
    for row in csv.reader(x):
        loss4.append(float(row[0]))

epoch =[i+1 for i in range(len(loss))]

plt.figure()
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.plot(epoch,loss,label="hessian_sp")
plt.plot(epoch,loss2,label="hessian_tp")
plt.plot(epoch,loss3,label="adam_sp")
plt.plot(epoch,loss4,label="adam_tp")
plt.legend(loc=0)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('img/Comparison_binde_loss.png')
'''