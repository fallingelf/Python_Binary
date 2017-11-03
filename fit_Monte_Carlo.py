import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
def ephemeris_linear(cycles, period , epoch):                                             #注意参数顺序
    return period * cycles + epoch                                                 #欲拟合的函数

def ephemeris_quadr(cycles, period, epoch, pdot):                                              #注意参数顺序
    return period * cycles + pdot * cycles**2 / 2 + epoch

def resample(cycles,eclips_time,error,threshold):
  factor_threshold=np.random.rand(len(cycles))*(error/max(error))
  mark=[i for i in range(len(cycles)) if factor_threshold[i] > threshold]
  print(mark)
  cycles=cycles[mark]
  eclips_time=eclips_time[mark]
  error=error[mark]
  return cycles,eclips_time,error

data_all=[[],[],[]]
data_all[0]=np.array([0,13,14,21,42,2760,2761,2768,7345,7804,43583,43590,43929,46379,46400,51543,52265,54114,56960,59346,59833])-int(59833/2)
data_all[1]=np.array([2449007.58900,2449009.42400,2449009.56600,2449010.55400,2449013.52100,2449397.42100,2449397.56150,2449398.55030,2450045.02352,2450109.85462,2455163.41633,2455164.40512,2455212.28680,2455558.33421,2455561.30025,2456287.71728,2456389.69528,2456650.85507,2457052.83501,2457389.84317,2457458.62849])-2449000
data_all[2]=np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.00005,0.00002,0.00002,0.0001,0.00002,0.00013,0.00013,0.00016,0.00031,0.00022,0.00019,0.00009,0.0001,0.00009,0.00073,0.00014])
#cycles,eclips_time,error=resample(cycles,eclips_time,error,threshold=0.07)



Period_linear_list=[]
Epoch_linear_list=[]
Period_quadr_list=[]
Epoch_quadr_list=[]
Pdot_quadr_list=[]
N=10000
for i in range(N):
  cycles, eclips_time, error = resample(data_all[0], data_all[1], data_all[2], threshold=0.05)
  random_list=np.random.rand(len(cycles))
  eclips_time_resample=eclips_time+random_list*error
  popt_linear= optimize.curve_fit(ephemeris_linear,cycles,eclips_time_resample)
  Period_linear_list.append(popt_linear[0][0])
  Epoch_linear_list.append(popt_linear[0][1])
  popt_quadr = optimize.curve_fit(ephemeris_quadr,cycles,eclips_time_resample)
  Period_quadr_list.append(popt_quadr[0][0])
  Epoch_quadr_list.append(popt_quadr[0][1])
  Pdot_quadr_list.append(popt_quadr[0][2])

Period_linear=np.mean(Period_linear_list)
Period_linear_err=np.std(Period_linear_list)
Epoch_linear=np.mean(Epoch_linear_list)
Epoch_linear_err=np.std(Epoch_linear_list)
Period_quadr=np.mean(Period_quadr_list)
Period_quadr_err=np.std(Period_quadr_list)
Epoch_quadr=np.mean(Epoch_quadr_list)
Epoch_quadr_err=np.std(Epoch_quadr_list)
Pdot_quadr=np.mean(Pdot_quadr_list)
Pdot_quadr_err=np.std(Pdot_quadr_list)

print('linearfit : ')
print("Epoch     Period    =", Epoch_linear+2449000,Period_linear)
print("Epoch_err Period_err=", Epoch_linear_err,Period_linear_err)
print('quadrofit : ')
print("Epoch     Period     Pdot    =", Epoch_quadr+2449000,Period_quadr,Pdot_quadr)
print("Epoch_err Period_err Pdot_err=", Epoch_quadr_err,Period_quadr_err,Pdot_quadr_err)



plt.subplot(211)
plt.plot(cycles, eclips_time+2449000,'*',label='eclips_time')
plt.plot(cycles, ephemeris_linear(cycles,Period_linear,Epoch_linear+2449000), 'r-', label='linearfit')
plt.plot(cycles, ephemeris_quadr(cycles,Period_quadr,Epoch_quadr+2449000,Pdot_quadr), 'b-', label='qoudrofit')
plt.xlabel('cycles')
plt.ylabel('HJD')
plt.legend()

plt.subplot(212)
O_C_linear=eclips_time+2449000-ephemeris_linear(cycles,Period_linear,Epoch_linear+2449000)
O_C_quadr=eclips_time+2449000-ephemeris_quadr(cycles,Period_quadr,Epoch_quadr+2449000,Pdot_quadr)
plt.plot(cycles,np.zeros(len(cycles)),'k-',label='base_line')
plt.plot(cycles, O_C_linear, 'r*', label='O-C_linear')
plt.plot(cycles, O_C_quadr,  'b*', label='O-C_quadr')
plt.xlabel('cycles')
plt.ylabel('O-C')
plt.legend()
#plt.show()

file_in=open('fit_result.txt','a+')
setting_txt='fit_Monte_Carlo'+'\n'+'num_resample='+str(N)+'\n'
linear_fit_txt='linear_fit:' + '\t' + str(Epoch_linear+2449000) + '\t' +  str(Period_linear) + '\t' +  str(Epoch_linear_err)+ '\t' +  str(Period_linear_err)+'\n'
quadr_fit_txt= 'quadr_fit:' + '\t' +  str(Epoch_quadr+2449000) + '\t' +  str(Period_quadr) + '\t' +  str(Pdot_quadr)+ '\t' +  str(Epoch_quadr_err)+ '\t' +  str(Period_linear_err) + '\t' + str(Pdot_quadr_err)+'\n'
file_in.write(setting_txt)
file_in.write(linear_fit_txt)
file_in.write(quadr_fit_txt)
file_in.close()