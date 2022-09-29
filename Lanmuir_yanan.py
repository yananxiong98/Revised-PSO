"""
@author: Yanan Xiong
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit,least_squares
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Definitions of Global variables
C0 = 50.0
Cs = 4.0
t = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0,
             90.0, 105.0, 120.0, 150.0, 180.0, 210.0])
qt = np.array([0, 2.022644346, 4.609541933, 6.399231094, 7.236343423, 8.474510454,
              9.392376354, 9.933348215, 10.24436494, 10.87449072, 11.34656137, 11.62453718])
dt = 0.1
thresh = 1
ctc0 = np.array((C0-Cs*qt)/C0)
if thresh !=1:
    qt = np.append([0],qt[ctc0<=thresh])
    t = np.append([0],t[ctc0 <= thresh])

def PSO_getQ(t, qe, k2):
    return t/((t/qe)+(1/(k2*qe**2)))

# linear PSO-----------------------


def PSO_linear(t, qt):
    # get the m and c for the linearized curve
    t_qt = t[1:]/qt[1:]
    t1 = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(t1[1:], t_qt)
    c = float(model.intercept_)
    m = float(model.coef_)
    # calculate the parameters (qe and k2) from the c,m
    qe = 1/m
    k2 = 1/(c*qe*qe)

    # calculate the r^2 value
    q_model = PSO_getQ(t, qe, k2)
    r_sq = 1 - ((qt-q_model)**2).sum()/((qt-qt.mean())**2).sum()

    return qe, k2, r_sq


# nonlinear PSO-----------------------

def PSO_nonlinear(t, qt):
    # use the qe,k2 from linear method as initial guess
    p0 = np.array([PSO_linear(t, qt)[0], PSO_linear(t, qt)[1]])

    # find the optimised parameters qe,k2 fitting the PSO equation,
    # by using the given t,qt
    popt, pcov = curve_fit(PSO_getQ, t, qt, p0=p0)
    qe, k2 = popt

    # calculate the r^2 value
    q_model = PSO_getQ(t, qe, k2)
    r_sq = 1 - ((qt-q_model)**2).sum()/((qt-qt.mean())**2).sum()

    return qe, k2, r_sq

# ------revised PSO---------

# equation to get q by revised PSO


def rPSO_getQ(t,qe,kprime):
    # create time series with dt timestep
    t1 = np.arange(0,t[-1]+dt,dt)
    t_index = np.array(t*(1/dt),'i')
    # calculate qt for each timestep
    q = np.zeros(len(t1),dtype="float")
    q_model = []
    for i in range(1,len(t1)):
        q[i] = q[i-1]+(dt*(kprime*(C0-Cs*q[i-1])*(1-q[i-1]/qe)**2))
    # extract data points corresponding to the experimental data
    for j in t_index:
        q_model.append(q[j])
    return q_model


def rPSO_nonlinear(t,qt):           

    p1,p2,_ = PSO_nonlinear(t,qt)
    p1 = float(p1)
    p2 = float(p2*p1**2/C0**2)
    p0 = np.array([p1,p2])
    
    # find the optimised parameters qe,k2 fitting the PSO equation,
    # by using the given t,qt
    popt,pcov = curve_fit(rPSO_getQ, t, qt, p0=p0,bounds=([0.00000001,0],np.inf))
    qe,kprime = popt
    
    # calculate the r^2 value 
    q_model = rPSO_getQ(t,qe,kprime)
    r_sq = 1 - ((qt-q_model)**2).sum()/((qt-qt.mean())**2).sum()
    return qe,kprime,r_sq



def MonteCarlo(q_model,qe,k,func):
    # create the array to store the parameters obtained by the two MonteCarlo simulations
    k_MC = np.zeros(200)
    qe_MC = np.zeros(200)
    
    size = np.size(t)
    SSE = np.sum(np.square(q_model-qt))
    std = np.sqrt(SSE/(size-2))
    q_monte = np.zeros(size)
    
    # repeat the Monte Carlo simulation for 200 times 
    for i in range(200):
        q_monte = q_model + norm.ppf(np.random.rand(size), 0, std)
        # put the simulated data into models to obtain the parameters
        qe_MC[i],k_MC[i],_ = func(t,q_monte)
        
    # qe and k in confidence level of 95%
    qe_MC = sorted(qe_MC)
    k_MC = sorted(k_MC)
    qe_CL = qe_MC[5:195]
    k_CL = k_MC[5:195]

    # uncertainties
    qe_un = (qe_CL[-1]-qe_CL[0])/2
    k_un = (k_CL[-1]-k_CL[0])/2
    
    return qe_un,k_un

if __name__ == "__main__":
    print("manna: linearised PSO")
    qe,k2,r_sq= PSO_linear(t,qt)
    q_model= t/((t/qe)+(1/(k2*qe**2)))
    qe_un,k2_un = MonteCarlo(q_model,qe,k2,PSO_linear)
    print(f"qe: {'%.6f'%qe} +- {'%.6f'%qe_un}")
    print(f"k2: {'%.6f'%k2} +- {'%.6f'%k2_un}")
    print(f"R^2: {'%.6f'%r_sq}")
    print(f"qt_model: {q_model}")


    print("manna: non linearised PSO")
    qe,k2,r_sq = PSO_nonlinear(t,qt)
    q_model= t/((t/qe)+(1/(k2*qe**2)))
    qe_un,k2_un = MonteCarlo(q_model,qe,k2,PSO_nonlinear)
    print(f"qe: {'%.6f'%qe} +- {'%.6f'%qe_un}")
    print(f"k2: {'%.6f'%k2} +- {'%.6f'%k2_un}")
    print(f"R^2: {'%.6f'%r_sq}")
    print(f"qt_model: {q_model}")

    
    print('manna: nonlinear rPSO')
    qe,kprime,r_sq = rPSO_nonlinear(t,qt)
    q_model= rPSO_getQ(t,qe,kprime)
    qe_un,kprime_un = MonteCarlo(q_model,qe,kprime,rPSO_nonlinear)
    print(f"qe: {'%.6f'%qe} +- {'%.6f'%qe_un}")
    print(f"k2: {'%.6f'%kprime} +- {'%.6f'%kprime_un}")
    print(f"R^2: {'%.6f'%r_sq}")
    print(f"qt_model: {q_model}")

    plt.figure(figsize = (12,10),label="Manna PSO")
    plt.figure(1)
    ax1 = plt.subplot(221)
    plt.title('Linearized PSO Fitting (linear plot)',fontsize=13)
    qe,k2,r_sq = PSO_linear(t,qt)
    q_model= t/((t/qe)+(1/(k2*qe**2)))
    m = 1/qe
    c = 1/k2/qe/qe
    y = t[1:]/qt[1:]
    y_model=m*t+c
    r2 =  1 - ((y-y_model[1:])**2).sum()/((y-y.mean())**2).sum()
    plt.scatter(t,t/qt,label = "observed data")
    plt.plot(t,y_model,"green",label = "linearised PSO")
    ax1.legend(loc='upper left')
    ax1.legend("R^2: {r2}")
    ax1.legend(fontsize=12)
    ax1.set_xlabel('Time (min)',fontsize=13)
    ax1.set_ylabel('t/qt (min g mg-1)',fontsize=13)
    ax1.text(0.7*t.max(),0.2*y.max(),f"R^2: {'%.5f'%r2}",fontsize=12)
    new_ticks = np.linspace(0, 1.1*y.max(), 8)
    plt.yticks(new_ticks)
    for i in range(np.size(new_ticks)):
        if new_ticks[i] < 100:
            new_ticks[i] = format(new_ticks[i],'.2g')
        elif new_ticks[i] >= 100:
            new_ticks[i] = format(new_ticks[i],'.3g')
    plt.yticks(new_ticks,fontsize=10)
    plt.xticks(fontsize=10)


    ax2 = plt.subplot(222)
    plt.title('Linearised PSO (nonlinear plot)',fontsize=13)
    plt.scatter(t,qt,label = "observed data")
    t_p = np.linspace(0, 1.1*t.max(), 100)
    q_model= t_p/((t_p/qe)+(1/(k2*qe**2)))
    ax2.plot(t_p,q_model,"red",label = "linear PSO")
    ax2.legend(loc='upper left')
    ax2.legend("R^2: {r_sq}")
    ax2.legend(fontsize=12)
    ax2.set_xlabel('Time (min)',fontsize=13)
    ax2.set_ylabel('qt (mg g-1)',fontsize=13)
    ax2.text(0.7*t_p.max(),0.2*q_model.max(),f"R^2: {'%.5f'%r_sq}",fontsize=12)
    new_ticks = np.linspace(0, 1.1*qt.max(), 8)
    for i in range(np.size(new_ticks)):
        if new_ticks[i] < 100:
            new_ticks[i] = format(new_ticks[i],'.2g')
        elif new_ticks[i] >= 100:
            new_ticks[i] = format(new_ticks[i],'.3g')
    plt.yticks(new_ticks,fontsize=10)
    plt.xticks(fontsize=10)

    ax3 = plt.subplot(223)
    plt.title('Nonlinearised PSO',fontsize=13)
    plt.scatter(t,qt,label = "observed data")
    qe,k2,r_sq= PSO_nonlinear(t,qt)
    q_model= t_p/((t_p/qe)+(1/(k2*qe**2)))
    plt.plot(t_p,q_model,'red', label ="non linear PSO")
    ax3.legend(loc='upper left')
    ax3.legend(fontsize=12)
    ax3.set_xlabel('Time (mins)',fontsize=13)
    ax3.set_ylabel('qt (mg g-1)',fontsize=13)
    ax3.text(0.7*t_p.max(),0.2*q_model.max(),f"R^2: {'%.5f'%r_sq}",fontsize=12)
    new_ticks = np.linspace(0, 1.1*qt.max(), 8)
    for i in range(np.size(new_ticks)):
        if new_ticks[i] < 100:
            new_ticks[i] = format(new_ticks[i],'.2g')
        elif new_ticks[i] >= 100:
            new_ticks[i] = format(new_ticks[i],'.3g')
    plt.yticks(new_ticks,fontsize=10)
    plt.xticks(fontsize=10)

    ax4 = plt.subplot(224)
    plt.title('Nonlinear rPSO',fontsize=13)
    plt.scatter(t,qt,label = "observed data")
    qe,kprime,r_sq= rPSO_nonlinear(t,qt)
    q_model= rPSO_getQ(t_p,qe,kprime)
    plt.plot(t_p,q_model,"red",label ="non linear rPSO")
    ax4.legend(loc='upper left')
    ax4.legend("R^2: {r_sq}")
    ax4.legend(fontsize=12)
    ax4.set_xlabel('Time (mins)',fontsize=13)
    ax4.set_ylabel('qt (mg g-1)',fontsize=13)
    ax4.text(0.7*t_p.max(),0.2*q_model[-1],f"R^2: {'%.5f'%r_sq}",fontsize=12)
    new_ticks = np.linspace(0, 1.1*qt.max(), 8)
    for i in range(np.size(new_ticks)):
        if new_ticks[i] < 100:
            new_ticks[i] = format(new_ticks[i],'.2g')
        elif new_ticks[i] >= 100:
            new_ticks[i] = format(new_ticks[i],'.3g')
    plt.yticks(new_ticks,fontsize=10)
    plt.xticks(fontsize=10)
    plt.show()