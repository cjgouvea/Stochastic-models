# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def random_walk(x0,t,dt):
    #Quantidade de elementos
    T = int(t/dt)                      
    S = [0]*T
    S[0] = x0
    for i in range(1,T):
        #Craição de elemento aleatório
        eps = round(np.random.normal(0,1),2)
        dz = eps*(np.sqrt(dt))
        #St = St-1 + dz
        S[i]=S[i-1]+dz
        
    return S
    
def MAB(x0,t,dt,mi,sigma):
    #Quantidade de elementos
    T = int(t/dt)
    #Criação das listas                      
    S = [0]*T                          
    trend = [0]*T 
    #Definição do elemento Inicial                     
    S[0] = x0
    trend[0] = x0                      
    for i in range(1,T):
        #Geração de número aleatório
        eps = np.random.normal(0,1)
        #Produto de epsilon por raiz de dt
        dz = eps*np.sqrt(dt)        
        #Incremento delta S
        dS = mi*dt + sigma*dz  
        #S(t+1)        
        S[i]=S[i-1] + dS               
        #Tendência
        trend[i]=trend[i-1]+mi*dt      
    #Retorno 2 arrays: Série aleatória e tendência
    return (S,trend)
    
def MGB(x0,t,dt,mi,sigma,ic_init=0.5):
    #Quantidade de elementos
    T = int(t/dt) 
    #Criação das listas                      
    S = [0]*T
    trend = [0]*T
    ic_min,ic_max = [0]*T ,[0]*T 
    #Definição do elemento Inicial     
    S[0] = x0    
    trend[0] = x0                       
    ic_min[0] = 0  
    ic_max[0] = 0
    
    for i in range(1,T):
        #Geração de número aleatório
        eps = np.random.normal(0,1) 
        #Produto de epsilon por raiz de dt 
        dz = eps*(np.sqrt(dt))       
        #Definição dos coeficientes
        growth_coef = (mi-0.5*(sigma**2)) 
        vol_coef = sigma
        #S(t+dt)
        S[i]=round(S[i-1]*np.exp(growth_coef*dt + vol_coef*dz),3) 
        
        #E(t+dt)                                 
        trend[i]=trend[0]*(np.exp(mi*i*dt)) 
        #Tau = 1 para 66% de confiança        
        tau=1
        if i > ic_init*T:
            #Criação dos intervalos de Confiança
            ic_min[i] = trend[i]*(np.exp(-tau*sigma*np.sqrt((i-T/2)*dt)))
            ic_max[i] = trend[i]*(np.exp(tau*sigma*np.sqrt((i-T/2)*dt)))
     
        else: 
            #E(t+dt)
            ic_min[i] = trend[i]
            ic_max[i] = trend[i]
    #Retorno 4 arrays: Série aleatória e tendência e IC minimo e máximo      
    return (S, trend,ic_min,ic_max)

def MJD(x0,t,dt,mi_d,sigma_d,mi_j,sigma_j,lambda_):
    #Quantidade de elementos
    T = int(t/dt)
    #Criação das listas  
    S = [0]*T
    trend = [0]*T
    #Definição do elemento Inicial     
    S[0] = x0
    trend[0] = x0
    #Criação de uma distribuição de Poisson para a ocorrência dos saltos
    N = np.random.poisson(lambda_*dt, T)
    #Criação de uma distribuição de Poisson para o tamanho dos saltos
    #Jump-up
    Q = np.random.normal(mi_j,sigma_j**2,T)
    #Jump-Down
    Q_ = np.random.normal(-1*mi_j,sigma_j**2,T)
    for i in range(1,T):
        #Geração de número aleatório
        eps = round(np.random.normal(0,1),4)
        dW = eps*(np.sqrt(dt))
        #Calculo do coeficiente de Drift
        drift_coef = (mi_d-(sigma_d**2)/2)
        #O elemento N define a ocorrência dos saltos (0 ou 1)
        #enquanto o tamanho dos saltos é uma escolha aleatória entre Q e Q_
        jump = N[i]*np.random.choice([Q[i],Q_[i]])
        
        #Log-retorno entre o elemento t-1 e t
        R_dt = drift_coef*dt + sigma_d*dW + jump
        #Elemento St = St-1*R_dt
        S[i]=S[i-1]*np.exp(R_dt)
        #Valor Esperado
        E_Rdt = mi_d*dt
        trend[i] = trend[i-1]*np.exp(E_Rdt)
    #Retorno 2 arrays: Série aleatória e tendência    
    return (S,trend)

def MRM_OU(x0,t,dt,mi,sigma,k):
    #Quantidade de elementos
    T = int(t/dt)                       
    S = [0]*T
    #Definição do elemento Inicial   
    S[0] = x0                           
    trend = [mi]*T                   
    for i in range(1,T):
        e_minus_kdt = np.exp(-k*dt)
        #Variancia da distribuição Normal de retornos aleatórios
        sigma_eps2 = ((sigma**2)/(2*k))*(1-(e_minus_kdt**2))
        eps = round(np.random.normal(0,1),2)
        
        S[i]= round(S[i-1]*e_minus_kdt+(1-e_minus_kdt)*mi+np.sqrt(sigma_eps2)*eps,2)
    #Retorno 2 arrays: Série aleatória e tendência    
    return (S, trend)

def MRM_exp_OU(x0,t,dt,mi,sigma,k):
    T = int(t/dt)
    P = [0]*T
    P[0] = x0
    trend = [0]*T
    trend[0] = x0
    for i in range(1,T):
        e_minus_kdt = np.exp(-k*dt)
        e_minus_2kdt = np.exp(-2*k*dt)
        #Variancia da distribuição Normal de retornos aleatórios
        sigma_eps2 = np.sqrt((sigma**2)*(1-e_minus_2kdt)/(2*k))
        eps = round(np.random.normal(0,1),4)

        alpha1 = (sigma**2)/(2*k)
        alpha2 = mi - alpha1
        
        #Tvedt 1997:
        #Elemento Pt
        P[i]=np.exp(np.log(P[i-1])*e_minus_kdt + (alpha2)*(1-e_minus_kdt) + sigma_eps2*eps)
                
        
        #Tvedt 1997
        #Valor Esperado
        E_cond = np.exp(((e_minus_kdt**i)*np.log(trend[0]))+(alpha2*(1-(e_minus_kdt**i)))+((sigma**2)/(4*k))*(1-(np.exp(-2*k*i*dt))))
        trend[i]=E_cond        
    #Retorno 2 arrays: Série aleatória e tendência    
    return (P, trend)

def MRM_jumps(x0,t,dt,mi,sigma,k,lambda_,mi_j,sigma_j):
    T = int(t/dt)
    S = [0]*T
    S[0] = x0
    trend = [np.exp(mi)]*T
    #Criação de uma distribuição de Poisson para a ocorrência dos saltos
    N = np.random.poisson(lambda_*dt, T)
    #Criação de uma distribuição de Poisson para o tamanho dos saltos
    #Jump-up    
    Q = np.random.normal(mi_j,sigma_j**2,T)
    #Jump-down
    Q_ = np.random.normal(-1*mi_j,sigma_j**2,T)
    for i in range(1,T):
        #Geração de número aleatório
        eps = round(np.random.normal(0,1),2)
        
        e_minus_kdt = np.exp(-k*dt)
        sigma2_2k = (sigma**2)/(2*k)
        sigma_eps2 = np.sqrt((sigma**2)*(1-e_minus_kdt**2)/(2*k))
        #O elemento N define a ocorrência dos saltos (0 ou 1)
        #enquanto o tamanho dos saltos é uma escolha aleatória entre Q e Q_
        jump = N[i]*np.random.choice([Q[i],Q_[i]])
        
        if jump>0:
            #O salto domina o retorno
            S[i]=S[i-1]*np.exp(jump)
        else:
            #Retono aleatório
            R_dt = (1-e_minus_kdt)*(mi-sigma2_2k)+(e_minus_kdt-1)*np.log(S[i-1])+sigma_eps2*eps
            S[i] = S[i-1]*np.exp(R_dt)

    return (S, trend)


################  FERRAMENTAS  ################################
def log_retorno(S):
    #Transforma uma série de preços em uma série de log-retornos
    R = []
    for i in range(1,len(S)):
        R_dt = np.log(S[i]/S[i-1])
        R.append(R_dt)
    R.append(0)
    R_arr = np.array(R)
    return R_arr

def retorno(S):
    #Transforma uma série de preços em uma série de retornos
    R = []
    for i in range(1,len(S)):
        R_dt = S[i]/S[i-1]
        R.append(R_dt)
    R.append(1)
    R_arr = np.array(R)
    return R_arr

def jump_selection(data,up_lim,low_lim):
    #Detects the jumps based on the upper and lower bounds
    logR = log_retorno(data)
    dt=1/12
    jumps_list = []
    diff_list = []
    
    for i,r in enumerate(logR):
        if r>up_lim:
            jumps_list.append(i)
        elif r<low_lim:
            jumps_list.append(i)
        else:
            diff_list.append(i)
      
    logR_jumps = [logR[j] for j in jumps_list]
    S_jumps = [data[j] for j in jumps_list]
    
    logR_diff = [logR[d] for d in diff_list]
    S_diff = [data[d] for d in diff_list]
    
    lamb = len(logR_jumps)/((len(logR)-1)*dt)
    #Returns the frequency of jumps/log-returns jump series/log-returns diff series
        #jump series/diff series
    return (lamb,logR_jumps,logR_diff,S_jumps,S_diff)

def MGB_param_estimate(S,dt):
    #Parameter estimate for the MGB
    logR = log_retorno(S)
    mean = logR.mean()
    var=logR.var(ddof=1)
    muhat = (2*mean+var*dt)/(2*dt)
    sigmahat = np.sqrt(var/dt)
    
    return (muhat,sigmahat)

def MJD_param_estimate(S,dt,up_lim,low_lim):
    #Parameter estimate for the MJD
    S_lambda,R_j,R_d,S_j,S_d = jump_selection(S,up_lim,low_lim)
    mu_d,sig_d=MGB_param_estimate(np.array(S_d),dt)
    
    R_j_mod=np.abs(R_j)
    mu_j= R_j_mod.mean()
    sig_j = np.sqrt(R_j_mod.var(ddof=1))

    return (mu_d,sig_d,mu_j,sig_j,S_lambda)

def std_error(logS,logR,p_list):
    #Calculates the standard error of a regression
    N = len(logR)
    
    H_x = np.array(logS)*np.array(p_list[1])+np.array(p_list[0])
    diff = np.array(logR-H_x)
    
    var_E = (np.array(diff**2).sum())/(N)
    std_E = np.sqrt(var_E)
    
    return std_E

def GMR_param_estimate(S,dt):
    #Parameter estimate for the GMR (MRM_exp_OU)
    S_arr = np.array(S)
    logS = np.log(S_arr)
    logR = log_retorno(S)
    
    slope,intercept,r_value,p_value,error = stats.linregress(logS,logR)
    a = intercept
    b = slope + 1
    
    std_E = std_error(logS,logR,[intercept,slope])
    print('a = {}, slope = {} , b = {}'.format(a,slope,b))
    
    nabla = -(np.log(b)/dt)
    sigma = std_E*np.sqrt((2*np.log(b))/((b**2-1)*dt))    
    P_bar = np.exp((a/(1-b))+((sigma**2)/(2*nabla)))
    #S_bar2 = np.exp((a/(1-b))+((std_error**2)/(1-b**2)))
    #print(S_bar2-S_bar)
    print('Nabla = {}, sigma = {}, P_ = {}'.format(nabla,sigma,P_bar))
    
    return (intercept,slope,nabla,sigma,P_bar)


def MRMSP_param_estimate(S,dt,up_lim,low_lim):
    #Parameter estimate for the MRMSP
    
    S_lambda,R_j,R_d,S_j,S_d = jump_selection(S,up_lim,low_lim)
    R_j_mod=np.abs(R_j)
    
    mu_j= R_j_mod.mean()
    sig_j = np.sqrt(R_j_mod.var(ddof=1))
    
    logS_d = np.log(S_d)
    slope,intercept,r_value,p_value,error = stats.linregress(logS_d,R_d)            
    a = intercept
    b = slope +1
    print('a = {}, slope = {} , b = {}'.format(a,slope,b))
    std_E = std_error(logS_d,R_d,[intercept,slope])
    nabla = -np.log(b)/dt
    sigma = std_E*np.sqrt((2*np.log(b))/(((b**2)-1)*dt))
    P_bar = np.exp(((a/(1-b)))+((sigma**2)/(4*nabla)))
    print('Nabla = {}, sigma = {}, P_ = {}'.format(nabla,sigma,P_bar))
        
    return (intercept,slope,nabla,sigma,P_bar,mu_j,sig_j,S_lambda)
    

def plot_regression(data,reg_coef,up_lim,low_lim,title=' ',color='g'):
    #Plot the linear regression of the log-return series without jumps
    lamb,R_jumps,R_diff,S_jumps,S_diff = jump_selection(data,up_lim,low_lim)
    logS = np.log(np.array(S_diff))
    logR = np.array(R_diff)
    x = np.linspace(np.array(logS).min(), np.array(logS).max(), 100)
    y = reg_coef[0] + reg_coef[1]*x
    plt.scatter(logS,logR,alpha=0.7,color=color)
    plt.plot(x,y,'k',label = 'y={:.5f}x +{:.5f}'.format(reg_coef[1],reg_coef[0]))
    plt.legend(loc = 'upper right')
    
    plt.ylabel('ln[S(t)/S(t-1)]                   ',rotation=0)
    plt.xlabel('ln[S(t)]')
    
    plt.title(title)
    return 0

def plot_jumps(S_j,title):
    #Plot the log-return with the jump bounds and the jump-distribution
    S_j_mod=np.abs(S_j)
    mu_j= S_j_mod.mean()
    sig_j = np.sqrt(S_j_mod.var())
    
    plt.hist(S_j,density=True,bins=10,color='cornflowerblue')
    x = np.linspace(0,1, 100)
    plt.plot(x, stats.norm.pdf(x, mu_j, sig_j),color='')
    x2= np.linspace(-1, 0, 100)
    plt.plot(x2, stats.norm.pdf(x2, -1*mu_j, sig_j),color='r')
    plt.title('Distribuição do tamanho dos saltos da série '+str(title))
    plt.show()
    
    return (mu_j,sig_j)


def MLE_MJD(x,lamb,mu_d,sigma_d,mu_j,sigma_j):
    
    dt=1/12
    n=1000
    f=[]*n
    for k in range(n):
        mu = (mu_d-(sigma_d**2)/2)*dt + mu_j*k
        loc =  (sigma_d**2)*dt + (sigma_j**2)*k
        pk=stats.poisson.pmf(k,lamb*dt)
        phi = stats.norm.pdf(x,mu,loc)
        f[k] = np.log(pk*phi)
    
    func = sum(f)
    return (-1)*func

def df_mov(func,length,IC=False,**kwargs):
    #Simulate a length of the selected random series
    #func must be a string in the list: MAB,MGB,MRM_OU,GMR,MRMSP,MJD
    columns = ['S'+str(i) for i in range(1,length+1)] + ['trend']
    
    df = pd.DataFrame(columns=columns)
    for i in df:
        if func=='MAB':
            S,trend = MAB(kwargs['x0'],kwargs['t'],kwargs['dt'],kwargs['mi'],kwargs['sigma'])
            df[str(i)] = S
        
        elif func=='MGB':
            S, trend ,ic_min, ic_max = MGB(kwargs['x0'],kwargs['t'],kwargs['dt'],kwargs['mi'],kwargs['sigma'])
            df[str(i)] = S
            
        elif func=='MRM_OU':
            S, trend = MRM_OU(kwargs['x0'],kwargs['t'],kwargs['dt'],kwargs['mi'],kwargs['sigma'],kwargs['k'])
            df[str(i)] = S
            
        elif func=='GMR':
            S, trend = MRM_exp_OU(kwargs['x0'],kwargs['t'],kwargs['dt'],kwargs['mi'],kwargs['sigma'],kwargs['k'])
            df[str(i)] = S
            
        elif func=='MRMSP':
            S, trend = MRM_jumps(kwargs['x0'],kwargs['t'],kwargs['dt'],kwargs['mi'],kwargs['sigma'],kwargs['k'],kwargs['lambda_'],kwargs['mi_j'],kwargs['sigma_j'])
            df[str(i)] = S
        
        elif func=='MJD':
            S, trend = MJD(kwargs['x0'],kwargs['t'],kwargs['dt'],kwargs['mi_d'],kwargs['sigma_d'],kwargs['mi_j'],kwargs['sigma_j'],kwargs['lambda_'])
            df[str(i)] = S
            
    df['trend'] = trend
    if IC == True:
        df['ic_min']=ic_min
        df['ic_max']=ic_max
    
    return df


def concat_sims(S,df):
    #Function to concat the simulated group of series to the data
    size=len(df.columns)
    columns = ['S'+str(i) for i in range(1,size)] + ['trend']
    df_S = pd.DataFrame(columns=columns)
    for i in range(1,size):
        label = 'S'+str(i)
        df_S[label] = np.concatenate((S.values,df[label].values),axis=None)
    df_S['trend']=np.concatenate((S.values,df['trend'].values),axis=None)
    return df_S
    
    

def plot_sims(S,dt,trend=True,var=False,title='Modelo'):
    #plot the simulations from df_mov    
    y = S
    fig = plt.figure(figsize=[10,4.8])
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Valor S(t)')
    ax1.set_xlabel('Tempo (t)')
    ax1.set_title(title)
    
    cores=colors(len(S.columns)-1)
    j=0
    if var == True and trend == True :
        for i in y.drop(['trend','ic_min','ic_max'],axis=1):    
            plt.plot(y[i], marker='',color=cores[j], linewidth=1.2, alpha=0.7, figure=fig)
            j=j+1
        trend = S['trend']
        ax1.plot(y.index,trend,'k--')
        ic_min= S['ic_min']
        ic_max = S['ic_max']
        ax1.plot(y.index,ic_min,'grey')
        ax1.plot(y.index,ic_max,'grey')
        
    elif trend == True:
        for i in y.drop(['trend'],axis=1):    
            plt.plot(y[i], marker='',color=cores[j], linewidth=1.2, alpha=0.7, figure=fig)
            j=j+1
        trend = S['trend']
        ax1.plot(y.index,trend,cores[-1])
    
    else:
        for i in y.drop(['trend'],axis=1):    
            plt.plot(y[i], marker='',color=cores[j], linewidth=1.2, alpha=0.7, figure=fig)
            j=j+1    
  
    
def colors(length):
    #Define the range of colors so the trend and IC are always the same
    colors_list = ['r','g','b','c','m','y','indigo','grey','orange','lime','purple']
    if length>len(colors_list):
        colors_list = colors_list*length
    
    final_colors = colors_list[0:length-1]+['k','k--']
    
    return final_colors
