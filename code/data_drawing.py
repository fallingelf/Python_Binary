import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


#file_list=os.listdir('C:\\Users\\falli\\Desktop\\meterial\\MNHya\\2009-2016')    #读取目录下文件列表
#data_file=[]
#for file in file_list:
#    if file[len(file)-4:len(file)]=='OPJ' or 'opj':
#        data_file.append(file[0:len(file)-4]+'.txt')
#        file_pid=open('C:\\Users\\falli\\Desktop\\meterial\\MNHya\\Python_deal\\data_all\\'+\
#                      file[0:len(file)-4]+'.txt','a+')             #当文件句柄被重新赋值后，之前的文件将会被关闭
#file_pid.close()


def data_save_draw(flag):    #flag指定历元数组
    filedir="C:\\Users\\falli\\Desktop\\meterial\\MNHya\\Data_Deal_Python\\Python_In\\data_all"
    file_list = os.listdir(filedir)
    data_all = []  # [0]为约化的HJD，[1]为相位,[2]较差星等
    for i in range(len(file_list)):
        data_all.append([])
        data_all[i]=[[None] for i in range(3)]
    data_name= []
    plot_mode = ('c.--', 'g.--', 'b.--', 'm.--', 'r.--', 'c.-', 'g.-', 'b.-', 'm.-', 'r.-', 'y.-', 'k.-')  # 指定画图样式
    pre_HJD_lib=[['20091127MNHya_N.txt', '20091128MNHya_N.txt', '20100115MNHya_N.txt', '20101227MNHya_N.txt', '20101230MNHya_N.txt', \
                  '20121225MNHya_N.txt', '20130406MNHya_N.txt', '20131223MNHya_N.txt', '20150129MNHya_N.txt', '20160101MNHya_N.txt', \
                  '20160117MNHya_V.txt', '20160310MNHya_N.txt'],\
                  [2455100, 2455100, 2455200, 2455500, 2455500, 2456200, 2456300, 2456600, 2457000, 2457300, 2457400,2457400]]
    Epoch_list =  [2449007.5882, 2453233.03745]
    Period_list = [0.14124373, 0.141243809727]
    Epoch = Epoch_list[flag]  # linearfit_leastsq.pfit_linear[1]+2449000
    Period = Period_list[flag]  # linearfit_leastsq.pfit_linear[0]
    pre_HJD=[]
    for i in range(len(file_list)):
        pre_HJD.append(pre_HJD_lib[1][pre_HJD_lib[0].index(file_list[i])])

    if len(file_list) == len(pre_HJD):
        i = 0
        for file in file_list:
            #      if file[len(file)-5] is not 'V':
            data = np.loadtxt(filedir +'\\'+ file)  # 直接读取txt文件中的数组
            data_name.append(file[0:len(file) - 4])
            data_all[i][0] = np.transpose(data)[0]  # reduced HJD
            data_all[i][1] = (data_all[i][0] + pre_HJD[i] - Epoch) / Period % 1  # 相位
            data_all[i][1] = np.where(data_all[i][1] < 0.5, data_all[i][1] + 1, data_all[i][1])  # 将0.5相位置于图形中间
            data_all[i][2] = np.transpose(data)[1]  # 较差星等数据
            order = np.lexsort((np.sort(data_all[i][1]), data_all[i][1]))  # np.lexsort((b, a))；返回b中的元素在a中的位置的数组
            data_all[i][0] = data_all[i][0][order]  # 按位置输出数组元素
            data_all[i][1] = data_all[i][1][order]
            data_all[i][2] = data_all[i][2][order]  # +i/11*4
            plt.plot(data_all[i][1], data_all[i][2], plot_mode[i], label=data_name[i])  # +'shifted+'+str(i/11*4)
            # ran=((random.random()*random.random()+random.random() ) %1 )*2-1
            # ram=((random.random()*random.random()+random.random() )%1 )*2-1
            # plt.annotate(data_name[i],xy=(data_all[i][1][int(len(data_all[i][2])*ran)],data_all[i][2][int(len(data_all[i][2])*ran)]),xytext=(data_all[i][1][int(len(data_all[i][2])*ran)]+0.1*ram,data_all[i][2][int(len(data_all[i][2])*ran)]+2*ram),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
            plt.annotate(data_name[i],
                         xy=(data_all[i][1][len(data_all[i][2]) - 1], data_all[i][2][len(data_all[i][2]) - 1]), xytext=(
                data_all[i][1][len(data_all[i][2]) - 1] + 0.002,
                data_all[i][2][len(data_all[i][2]) - 1] + (i - 6) * 0.3),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            # 用箭头直接标记曲线,选择在曲线结尾标记
            i = i + 1
    else:
        print('Matching Error!')
    plt.legend(loc='best')  # 使lable生效
    plt.xlabel('Phase')
    plt.ylabel('Mag.')
    plt.gca().invert_yaxis()     #反转y轴刻度
    plt.gcf().set_size_inches(37, 21)            #gca()返回当前的坐标实例（a matplotlib.axes.Axes instance），gcf()返回当前图像（matplotlib.figure.Figure instance）。
    #设置当前图像大小

    return data_name,data_all


def linear_interpol(data_all,num_data,which_Phase,which_Mag,N):  #which_Phase=1,which_Mag=2
    data_linear = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []],[[], []], [[], []], [[], []], [[], []], [[], []]]
    #print(num_data)
    Phase_std = np.linspace(0.5, 1.5, N + 1)
    for i in range(0,num_data):
        which_data = i
        Phase = data_all[which_data][which_Phase]
        Mag = data_all[which_data][which_Mag]
        Phase_start = next(Phase_std[i] for i in range(0, len(Phase_std) - 1) if Phase_std[i] >= min(Phase))
        Phase_end = next(Phase_std[len(Phase_std) - 1 - i] for i in range(0, len(Phase_std) - 1) if Phase_std[len(Phase_std) - 1 - i] <= max(Phase))
        Phase_new = np.linspace(Phase_start, Phase_end, (Phase_end-Phase_start) * N + 1)
        f_linear = interpolate.interp1d(Phase, Mag, kind=1)   #线性插值
        Mag_linear = f_linear(Phase_new)
        data_linear[i][0]=Phase_new
        data_linear[i][1]=Mag_linear
        #print(i,Phase_start, Phase_end,(Phase_start-0.5)*5000, (Phase_end-0.5)*5000)
    return data_linear

#def movingaverage(data_all):
#    for i in range(0,11):
#      data=data_all[i][1]
#
#      up_range=
#      low_range=
#      np.where(np.where(np.array(a) > 5, 0, a) < 3, 0, np.where(np.array(a) > 5, 0, a))
#      return MA_all,MA_mean
def Rearrange(data_linear,dimension,N):
    data_arrange=[]
    for i in range(dimension):
        data_arrange.append([])
        data_arrange[i]=[[None]*(N+1) for i in range(2)]
    for i in range(dimension):
        data_arrange[i][0]=np.linspace(0.5, 1.5, N + 1)
        start_point=(data_linear[i][0][0]-0.5)*N
        #print(i,data_linear[i][1])
        for j in range(len(data_linear[i][0])):
            data_arrange[i][1][int(round(start_point+j))]=data_linear[i][1][j]
    return data_arrange

def Meaning_data(data_arrange,dimension):
    meaning_mag =[]
    for j in range(len(data_arrange[0][0])):
      Mag_list=[]
      for i in range(dimension):  #0,1,2...,11
          if  data_arrange[i][1][j] is not None:
              Mag_list.append(data_arrange[i][1][j])
      if Mag_list == []:
         meaning_mag.append(None)
      else:
         meaning_mag.append(np.mean(Mag_list))
    return np.array(meaning_mag)

def Mag_Phase_plot(x,y,line_mode,line_text,position_index):  #position_index为数组,[0]表示位置，[1]偏移,[2]坐标尺度下缩放系数
    plt.plot(x,y,line_mode,label=line_text)
    plt.annotate(line_text,xy=(x[len(y)-1],y[len(y)-1]),xytext=(x[len(y)-1]+0.002,y[len(y)-1]+(position_index[0]-position_index[1])*position_index[2]),arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # 用箭头直接标记曲线,选择在曲线结尾标记
    plt.legend(loc='best')  # 使label生效
    plt.xlabel('Phase')
    plt.ylabel('Mag.')

#-------------------------------------------main function------------------------------------------------------

if __name__=='__main__':

    Num = 5000   #总区间插值点数
    plt.subplot(211)
    data_name, data_all=data_save_draw(1) # 0 or 1  #0代表old 参数，1代表新的参数。

    data_linear=linear_interpol(data_all,len(data_name),which_Phase=1,which_Mag=2, N=Num)

    plt.subplot(212)
    line_mode = ('c.--', 'g.--', 'b.--', 'm.--', 'r.--', 'c.-', 'g.-', 'b.-', 'm.-', 'r.-', 'y.-', 'k.-')  # 指定画图样式
    for i in range(0,len(data_name)):
        Mag_Phase_plot(data_linear[i][0],data_linear[i][1],line_mode=line_mode[i],line_text=data_name[i],position_index=np.array([i,len(data_name)-6,0.3]))
    plt.gca().invert_yaxis()
    plt.gcf().set_size_inches(37, 21)
    plt.savefig('C:\\Users\\falli\\Desktop\\meterial\\MNHya\\Data_Deal_Python\\Python_Out\\data_drawing', dpi=400)  # 设置保存图像精度

    data_arrange=Rearrange(data_linear,len(data_name),N=Num)

    meaning_mag=Meaning_data(data_arrange,len(data_name))
    meaning_mag=meaning_mag[np.array([i for i in range(len(meaning_mag)) if meaning_mag[i] is not None])]
    meaning_phase= np.linspace(0.5, 1.5, Num + 1)[np.array([i for i in range(len(meaning_mag)) if meaning_mag[i] is not None])]
    plt.plot(meaning_phase,meaning_mag,'k-')

    file_in = open('C:\\Users\\falli\\Desktop\\meterial\\MNHya\\Data_Deal_Python\\Python_Out\\data_drawing.txt', 'w+')
    file_in.write(str(data_arrange))
    file_in.close()


    plt.show()


