import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.stats.stats import linregress
import scipy.stats as stats


def ephemeris_linear(p,cycles):                                             #注意参数顺序
    period, epoch = p
    HJD=period*cycles+epoch                                                 #欲拟合的函数
    return HJD

def residuals_linear(p,cycles,eclips_time,error):
    ret = (ephemeris_linear(p,cycles)-eclips_time)/error                    #欲实施平方和极小化的数列---评价指标
    return ret

def ephemeris_quadr(p,cycles):                                              #注意参数顺序
    period, epoch, pdot = p
    return period * cycles + pdot * cycles**2 / 2 + epoch                   #欲拟合的函数

def residuals_quadr(p,cycles,eclips_time,error):
    return (ephemeris_quadr(p,cycles)-eclips_time)/error                    #欲实施平方和极小化的数列---评价指标


#极小时刻圈数，可以任意加减一个整数
cycles=np.array([0,13,14,21,42,2760,2761,2768,7345,7804,43583,43590,43929,46379,46400,51543,52265,54114,56960,59346,59833])-int(59833/2)   #避免精度问题
#对应圈数的极小时刻
eclips_time=np.array([2449007.58900,2449009.42400,2449009.56600,2449010.55400,2449013.52100,2449397.42100,2449397.56150,2449398.55030,2450045.02352,2450109.85462,2455163.41633,2455164.40512,2455212.28680,2455558.33421,2455561.30025,2456287.71728,2456389.69528,2456650.85507,2457052.83501,2457389.84317,2457458.62849])-2449000
#极小时刻的误差
error=np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.00005,0.00002,0.00002,0.0001,0.00002,0.00013,0.00013,0.00016,0.00031,0.00022,0.00019,0.00009,0.0001,0.00009,0.00073,0.00014])

#极小时刻圈数，可以任意加减一个整数
#cycles = np.array([0,13,14,21,42,2760,2761,2768,7345,7804,43583, 43590, 43929, 46379, 46400, 51543, 52265, 54114, 56960,59833]) - int( 59833 / 2)  # 避免精度问题
# 对应圈数的极小时刻
#eclips_time = np.array([2449007.58900,2449009.42400,2449009.56600,2449010.55400,2449013.52100,2449397.42100,2449397.56150,2449398.55030,2450045.02352,2450109.85462,2455163.41633, 2455164.40512, 2455212.28680, 2455558.33421,2455561.30025, 2456287.71728, 2456389.69528, 2456650.85507, 2457052.83501, 2457458.62849]) - 2449000
# 极小时刻的误差
#error = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.00005,0.00002,0.00002,0.0001,0.00002,0.00013, 0.00013, 0.00016, 0.00031, 0.00022, 0.00019, 0.00009, 0.0001, 0.00009, 0.00014])



p_linear_init = [0.14124373,8458.62849]                   #指定拟合参数的初值
p_quadr_init = [0.14124373,8458.62849,0]
pfit_linear, pcov_linear, infodict, errmsg, success= leastsq(residuals_linear, p_linear_init, args=(cycles,eclips_time,error),full_output=1)   #拟合函数
pfit_quadr, pcov_quadr, infodict, errmsg, success= leastsq(residuals_quadr, p_quadr_init, args=(cycles,eclips_time,error),full_output=1)   #拟合函数

#计算拟合参数的误差
if (len(eclips_time) > len(p_linear_init)) and pcov_linear is not None:
    s_sq = sum(residuals_linear(pfit_linear, cycles, eclips_time,error) ** 2) / (len(eclips_time) - len(p_linear_init))
    pcov_linear = pcov_linear * s_sq
else:
    pcov_linear = float("inf")

perr_linear = []
for i in range(len(pfit_linear)):
    try:
        perr_linear.append(np.absolute(pcov_linear[i][i]) ** 0.5)
    except:
        perr_linear.append(0.00)

#计算拟合参数的误差  multiply reduced covariance by the reduced chi squared
if (len(eclips_time) > len(p_quadr_init)) and pcov_quadr is not None:
    s_sq = sum(residuals_quadr(pfit_quadr, cycles, eclips_time,error) ** 2) / (len(eclips_time) - len(p_quadr_init))
    pcov_quadr = pcov_quadr * s_sq
else:
    pcov_quadr = float("inf")

perr_quadr = []
for i in range(len(pfit_quadr)):
    try:
        perr_quadr.append(np.absolute(pcov_quadr[i][i]) ** 0.5)
    except:
        perr_quadr.append(0.00)

perr_linear = np.array(perr_linear)
print('linearfit : ')
print("pfit_linear =", pfit_linear)
print("perr_linear =", perr_linear)
eclips_time_linearfit=ephemeris_linear(pfit_linear,cycles)
COD_linear=1 - sum((eclips_time-eclips_time_linearfit)**2)/sum((eclips_time_linearfit-np.mean(eclips_time))**2)   #计算可决系数(coefficient of determination)
print('COD_linear=%.20f'%(COD_linear))
perr_quadr = np.array(perr_quadr)
print('quadrofit : ')
print("pfit_quadr =", pfit_quadr)
print("perr_quadr =", perr_quadr)
eclips_time_quadrofit=ephemeris_quadr(pfit_quadr,cycles)
COD_quadr = 1 - sum((eclips_time - eclips_time_quadrofit) ** 2) / sum((eclips_time_quadrofit - np.mean(eclips_time)) ** 2)
print('COD_quadr=%.20f'%(COD_quadr))

ax1=plt.subplot(211)
plt.plot(cycles,eclips_time,'k*',label='eclips_time')
plt.plot(cycles,ephemeris_linear(pfit_linear,cycles), 'r-', label='linearfit')
plt.plot(cycles,ephemeris_quadr(pfit_quadr,cycles), 'b-', label='quadrofit')
plt.xlabel('cycles')
plt.ylabel('HJD(+2449000)')

#在第二张子图上显示O-C
ax2=plt.subplot(212)
OC_linearfit=eclips_time-ephemeris_linear(pfit_linear,cycles)
plt.plot(cycles,OC_linearfit,'r*')
plt.plot(cycles,np.zeros(len(cycles)),'k-')
plt.xlabel('cycles')
plt.ylabel('O-C')
plt.savefig('O-C')
#plt.show()

file_in=open('fit_result.txt','a+')
setting_txt='fit_leastsq'+'\n'
linear_fit_txt='linear_fit:' + '\t' + str(pfit_linear) + '\t' +  str(perr_linear)  + '\t' + str(COD_linear) + '\n'
quadr_fit_txt= 'quadr_fit:'  + '\t' + str(pfit_quadr)  + '\t' +  str(perr_quadr)   + '\t' + str(COD_quadr)  + '\n'
file_in.write(setting_txt)
file_in.write(linear_fit_txt)
file_in.write(quadr_fit_txt)
file_in.close()