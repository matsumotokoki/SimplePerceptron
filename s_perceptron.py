import numpy as np
import math
from matplotlib import pyplot as plt
import copy


def error_function(v,x,y,omega):
    L = (((1/(1+math.exp(-omega[0]*x-omega[1]*y-1)))-v)**2)/2 #2乗誤差(微分可能にするため)
    return L

def sigmoid_func(x,y,omega):
    return 1/(1+math.exp(-(x*omega[0]+y*omega[1]+1))) #いたって普通のシグモイド関数

def Partial_differentiation(v,x,y,omega):
    d = sigmoid_func(x,y,omega)
    dx = (d-v)*d*x/(1-d) #2乗誤差のx偏微分
    dy = (d-v)*d*y/(1-d) #2乗誤差のy偏微分
    return dx,dy

def read_data(path):
    data = []
    rdata = []
    sdata = []
    pdata = []
    x =  []
    y = []
    v = []
    with open(path) as f:
        data = f.readlines()
        for i in range(len(data)):
            rdata.append(data[i].split('\n'))
        for i in range(len(data)):
            sdata.append(rdata[i][0])
        for i in range(len(data)):
            pdata.append(sdata[i].split())
    for i in range(len(data)):
        x.append(float(pdata[i][0]))
        y.append(float(pdata[i][1]))
        v.append(float(pdata[i][2]))
    return x,y,v

if __name__=="__main__":
    path = './input.txt'
    omega = [0.2,-0.2] #適当な2値
    eta = 0.01 #学習率
    tmp = 0
    x = []
    y = []
    v = []
    
    x,y,v=read_data(path)

    for i in range(len(x)):
        if v[i]:
            plt.plot(x[i],y[i],marker='.', markersize=10,color='blue')
        else:
            plt.plot(x[i],y[i],marker='.', markersize=10,color='red')
    
    for n in range(len(x)*3):
        i = n%100
        error_sum = 0
        try:
            dx,dy = Partial_differentiation(v[i],x[i],y[i],omega)
        except ZeroDivisionError:
            print("ZeroDivisionError!!\nstop!!")
            break
        omega[0] -= eta*dx
        omega[1] -= eta*dy
        for k in range(len(x)):
            error_sum += error_function(v[k],x[k],y[k],omega)
        if tmp > error_sum or i==0:
            tmp = error_sum
            k_omega = copy.copy(omega)
        print(str(n)+": "+str(error_sum)+": "+str(omega))
        for i in range(len(x)):
            if v[i]:
                plt.plot(x[i],y[i],marker='.', markersize=10,color='blue')
            else:
                plt.plot(x[i],y[i],marker='.', markersize=10,color='red')
        plot_x = np.arange(-2,5,0.1)
        plot_fx = -(omega[1]/omega[0])*plot_x
        plt.plot(plot_x,plot_fx)
        plt.axes().set_aspect('equal')
        plt.xlim([-2.1, 5.1])
        plt.ylim([-2.1, 6])
        plt.pause(0.1)
        plt.clf()
    print('omega :' + str(k_omega))
    print(omega)

    plot_x = np.arange(-2,5,0.1)
    plot_fx = -(omega[1]/omega[0])*plot_x
    plt.plot(plot_x,plot_fx)
    plt.xlim([-2.1, 5.1])
    plt.ylim([-2.1, 6])
    plt.show()

