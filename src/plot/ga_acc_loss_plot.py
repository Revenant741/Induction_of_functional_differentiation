import csv
import matplotlib.pyplot as plt
import numpy as np

#python3 src/plot/ga_acc_loss_plot.py 
read_name='ga_hf_loss_e20_p20_l10_c1_g100/ga_hf_pop_20'
#read_name='func_diff_e20_p20_l10'
#write_name = 'func_diff_eva'
write_name = 'loss_eva_g100'
accuracy = []
accuracy2 = []
gene = []
generation = []
survivor = 10 #生き残る個体
generation = 100

with open('src/data/'+read_name+'_sp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy.append(float(row[0])*100)

with open('src/data/'+read_name+'_tp_acc.csv') as f:
    for row in csv.reader(f):
        accuracy2.append(float(row[0])*100)

# with open('src/data/'+read_name+'_generation1.csv') as f:
#     for row in csv.reader(f):
#         generation.append(float(row[0])-1)

#現在の世代の値
for i in range(generation):
  gene.append(i)

acc1 = np.array(accuracy).reshape(-1,survivor)
ave1 = np.mean(acc1, axis=1)
y_err1 = np.std(acc1, axis=1)
#print(acc1.shape)
#print(ave1.shape)
acc2 = np.array(accuracy2).reshape(-1,survivor)
ave2 = np.mean(acc2, axis=1)
y_err2 = np.std(acc2, axis=1)

plt.figure()
fig, ax = plt.subplots()
ax.errorbar(gene,ave1[:len(gene)], yerr=y_err1[:len(gene)],linestyle="None",capsize=5,label="spatial standard deviation",color="lightgreen")
ax.errorbar(gene,ave2[:len(gene)], yerr=y_err2[:len(gene)],linestyle="None",capsize=5,label="temporal standard deviation",color="lightcoral")
plt.plot(gene,ave1[:len(gene)],label="moving average spatial information",color="g")
plt.plot(gene,ave2[:len(gene)],label="moving average temporal information",color="r")
plt.xlim(0,len(gene)-1)
#plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
plt.yticks((10,20,30,40,50,60,70,80,90,100))
#plt.xticks((0,1,2,3,4))
plt.ylim(0,105)
plt.xlabel('Generation',fontsize=15)
plt.ylabel('Accuracy(%)',fontsize=15)
plt.legend(loc=4)
plt.savefig('src/img/'+write_name+'_acc.svg')
plt.savefig('src/img/'+write_name+'_acc.png')
plt.savefig('src/img/'+write_name+'_acc.pdf')


loss = []
loss2 = []
with open('src/data/'+read_name+'_sp_loss.csv') as f:
    for row in csv.reader(f):
        loss.append(float(row[0]))

with open('src/data/'+read_name+'_tp_loss.csv') as f:
    for row in csv.reader(f):
        loss2.append(float(row[0]))

lossbest = 100
for i in range(survivor*generation):
    loss_all = (loss[i]+loss2[i])/2
    if loss_all <= lossbest:
        lossbest = loss_all
        best_pop = i
print(best_pop)
print(lossbest)

loss1 = np.array(loss).reshape(-1,survivor)
loss_ave1 = np.mean(loss1, axis=1)
x_err1 = np.std(loss1, axis=1)
#print(loss1.shape)
#print(loss_ave1.shape)
loss2 = np.array(loss2).reshape(-1,survivor)
loss_ave2 = np.mean(loss2, axis=1)
x_err2 = np.std(loss2, axis=1)

plt.figure()
fig, ax = plt.subplots()
ax.errorbar(gene,loss_ave1[:len(gene)], yerr=x_err1[:len(gene)],linestyle="None",capsize=5,label="spatial standard deviation",color="lightgreen")
ax.errorbar(gene,loss_ave2[:len(gene)], yerr=x_err2[:len(gene)],linestyle="None",capsize=5,label="temporal standard deviation",color="lightcoral")
plt.plot(gene,loss_ave1[:len(gene)],label="average spatial information",color="g")
plt.plot(gene,loss_ave2[:len(gene)],label="average temporal information",color="r")
plt.xlim(0,len(gene)-1)
#plt.yticks((0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
#plt.yticks((10,20,30,40,50,60,70,80,90,100))
#plt.xticks((0,1,2,3,4))
plt.ylim(0,2.5)
plt.xlabel('Generation',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.legend(loc=1)
plt.savefig('src/img/'+write_name+'_loss.svg')
plt.savefig('src/img/'+write_name+'_loss.png')
plt.savefig('src/img/'+write_name+'_loss.pdf')


