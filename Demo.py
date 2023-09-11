import numpy as np
from functions.models import mode_0, mode_1, mode_2, mode_3, mode_4, mode_5, mode_6
from functions.comparison import gaussian_closedform, gaussian_gradientdescent, poly_ERM,kernelregression_closedform
from functions.clossfunction import Closs
from functions.createdata import CreateData
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pylab as plt
plt.rcParams.update({'font.size':20})

def EDERM(data_type = 1,case  = 1,lamb1 = 0.9,lamb2 = 3,opt='sgd'):
    '''
    Data type:
        Linear data:    '1,2,3'
        Nonlinear data: '4,5,6,7' 
        
    Ic type:
        'correntropy','sigmoid','tanh','hinge',('modifiedsquare','exponential')
        
    loss type:
        'mse','closs'
        
    mode: 
        0 ERM
        1 linear single threshold - EDERM           lamb1 < \rho
        2 linear double thresholds- EDERM           lamb1 < \rho < lamb2
        3 linear single threshold - 0/1 indicator  lamb1 < \rho
        4 linear double thresholds- 0/1 indicator  lamb1 < \rho < lamb2
        5 Poly-nonlinear single threshold EDERM
        6 Gaussian Kernel-nonlinear single threshold EDERM
        
    task:
        'linear'      Linear regression cases
        'polynomial'  Polynomial regression cases
        'kernel'      Gaussian kernel regression cases
        
    optimizer:
        'fgd','sgd','adaGrad','adam','amsgd','rmsprop','NAG'
        
    lamb1 & lamb2  stand for the error density threshold
    '''
    if(task== 'linear'):
        # Linear regression cases
        if data_type in [1,2,3]:
            X, y, Xtst, ytst = CreateData(data_type,'gaussian')
            
            ##ERM  Ordinary Linear Regression
            mode_0(X, y, Xtst, ytst,iters=1000,learning_step=0.002,tol = 1/10**10,losstype='mse',h=2,delta=2,online = True )
            
            ## EDERM(S,C)  square loss + Correntropy-surrogate indicator
            ##  lamb1 < \rho
            mode_1(X, y, Xtst, ytst,iters=1000,lamb=lamb1,learning_step=0.002,tol = 1/10**8,Ictype='correntropy',losstype='mse',h=1,delta=1,online =True, optimizer=opt)
            
            ## EDERM(C,C)  Correntropy loss + Correntropy-surrogate indicator
            ##  lamb1 < \rho
            mode_1(X, y, Xtst, ytst,iters=1000,lamb=lamb1,learning_step=0.002,tol = 1/10**8,Ictype='correntropy',losstype='closs',h=1,delta=1,online =True, optimizer=opt)
            
            ## EDERM(S,C)  square loss + Correntropy-surrogate indicator
            ##  lamb1 < \rho  < lamb2
            mode_2(X, y, Xtst, ytst,iters=1000,lamb=lamb1,lamb2 = lamb2, learning_step=0.002,tol = 1/10**8,Ictype='correntropy',losstype='mse',h=1,delta=2,online = True, optimizer=opt)
            
            ## EDERM(C,C)  Correntropy loss + Correntropy-surrogate indicator
            ##  lamb1 < \rho  < lamb2
            mode_2(X, y, Xtst, ytst,iters=1000,lamb=lamb1,lamb2 = lamb2, learning_step=0.002,tol = 1/10**8,Ictype='correntropy',losstype='mse',h=1,delta=2,online = True, optimizer=opt)
        else:
            print("Please choose linear data for linear regression (Type 1 2 3).")
    
    elif(task =='polynomial'):
        # Polynomial regression cases
        if data_type in [4]:
            X, y, Xtst, ytst = CreateData(data_type,'gaussian')
            #ERM
            poly_ERM(X,y,Xtst,ytst, online = True,dimensions = 4)
            #EDERM    Here fgd optimizer is suggested
            mode_5(X, y, Xtst, ytst,dimensions = 4,iters=2000,lamb=0.8,learning_step=0.02,tol = 1/10**10,Ictype = "correntropy",losstype='mse',h=1,delta=2,online = True, optimizer=opt,batchsize=10)
        else:
            print("Please choose nonlinear data for polynomial regression (Type 4).")
            
    elif(task =='kernel'):
        # Gaussian kernel regression cases
        if data_type in [4,5,6,7]:
            X, y, Xtst, ytst = CreateData(data_type,'gaussian')
            #ERM
            gaussian_gradientdescent(X, y, Xtst, ytst,outlier = True, online = True)
            #EDERM
            mode_6(X, y, Xtst, ytst,kernel = 'gaussian',delta_kernel = 0.8,iters=2000,lamb=1,learning_step=0.01,tol = 1/10**10,Ictype = "correntropy",losstype='mse',h=1,delta=2,online = True, optimizer=opt)
        else:
            print("Please choose nonlinear data for gaussian kernel regression (Type 4 5 6).")

if __name__=='__main__':
    # Generate data for regression
    # Type 1 2 3 -> linear regression; Type 4 -> polynomial regression; Type 4 5 6 7 -> gaussian kernel regression
    # Type 1 2 3 4 5 -> the curves are visible
    data_type = 3
    
    # Regression Task: Linear, polynomial or gaussian kernel regression
    task = 'linear'  # 'linear', 'polynomial' or 'kernel'
    
    # Error density threshold \lambda1 ( and \lambda2 for extended EDERM)
    lamb1=1
    lamb2=3
    
    # Extended accelerate optimizers (fgd,sgd,adam...)
    optimizer = 'adaGrad'  # 'fgd','sgd','adaGrad','adam','amsgd','rmsprop','NAG'
    
    # Run demo
    EDERM(data_type, task, lamb1, lamb2,optimizer)