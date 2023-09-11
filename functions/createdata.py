import numpy as np
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
import warnings
from sklearn import datasets

plt.rcParams.update({'font.size':20})
sns.set_style("whitegrid")
random_state=42
warnings.simplefilter('ignore')


def abline(a, label_, c=None):
    axes   = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = a * x_vals 
    plt.plot(x_vals, y_vals, label=label_, color=c, zorder=1)
 
def CreateData(case,noise_type = 'gaussian'):
    if(case == 1):
        # 1 dimension linear regression  for type 1
        features                      = 1
        [feature_mean,feature_std]    = [0  ,1]
        [noise_mean,noise_std]        = [0  ,1]
        [outlierx_mean,outlierx_std]  = [-2 ,1]
        [outliery_mean,outliery_std]  = [2,0.5]
        num_sample                    = 270
        num_outlier                   = 30 
        total_samples = num_sample + num_outlier
        
        theta = np.random.randint(5,6,features) 
        X     = np.random.normal(feature_mean,feature_std, [num_sample,features])
        if(noise_type == 'gaussian'):
            y     = np.dot(X,theta)    + np.random.normal(noise_mean,noise_std, num_sample)  
        elif(noise_type == 'student'):
            y     = np.dot(X,theta)    + np.random.standard_t(2, num_sample)
            
        Xtst  = np.random.normal(feature_mean,feature_std, [total_samples,features])
        ytst  = np.dot(Xtst,theta) + np.random.normal(noise_mean,noise_std, total_samples)
    
        X_out = np.random.normal(outlierx_mean,outlierx_std, [num_outlier,features])
        y_out = np.random.normal(outliery_mean,outliery_std, [num_outlier,1])
    
        plt.plot
        plt.figure(figsize=(8, 8))
        plt.scatter(X, y, s=2, label='Main Data')
        plt.scatter(X_out, y_out, s=5, label="Outliers")
        abline(theta, 'Ground Truth', "red")
        plt.legend()
        plt.show()
            
        X     = np.append(X, X_out)
        y     = np.append(y, y_out)   
        y     = np.append(y,[])     #list to array 
        ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((total_samples, features))
        Xtst  = Xtst.reshape((total_samples, features))
        
        
        
    elif(case == 2):
        # 1 dimension linear regression  for type 2
        samplestrn     = 100 
        samplesoutlier = 30
        features       = 1
        theta = np.random.randint(2,3) 
        X       = np.random.normal(0,10, samplestrn-samplesoutlier)
        y       = np.dot(X,theta) + np.random.normal(0,1, samplestrn-samplesoutlier)
        Xtst    = np.arange(-5, 5, 10/samplestrn)
        ytst    = 2*Xtst + np.random.normal(0,1, samplestrn)
        
        rand    = np.random.normal(5,2, samplesoutlier) + np.random.normal(0,1, samplesoutlier)
        X_out   = np.random.normal(-15,10, samplesoutlier)
        y_out   = np.dot(X_out,theta) +  rand
            
        plt.plot
        plt.figure(figsize=(8, 8))
        plt.scatter(X, y, s=2, label='Main Data')
        plt.scatter(X_out, y_out, s=5, label="Outliers")
        abline([2], 'Ground Truth', "red")
        plt.legend()
        plt.show()
        
        X     = np.append(X, X_out)
        y     = np.append(y, y_out)   
        y     = np.append(y,[])     #list to array 
        ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((samplestrn, features))
        Xtst  = Xtst.reshape((samplestrn, features))
        
        
    elif(case == 3):
        # 1 dimension linear regression  for type 3
        samplestrn     = 100 
        samplesoutlier = 30
        features       = 1
        X    = np.arange(-5, 5, 10/samplestrn)
        y    = 5*X + np.random.normal(0,1, samplestrn)
        Xtst = np.arange(-5, 5, 10/samplestrn)
        ytst = 5*X + np.random.normal(0,1, samplestrn)
        
        out   = random.sample(range(1,samplestrn),samplesoutlier)
        y[out]= X[out]*-10 + np.random.normal(3,1, samplesoutlier)

        plt.plot
        plt.figure(figsize=(8, 8))
        # plt.title("Original Setting",fontsize= 'xx-large') 
        plt.scatter(X, y, s=2, label='Main Data')
        plt.scatter(X[out], y[out], s=5, label="Outliers")
        abline([5], 'Ground Truth', "red")
        plt.legend()
        plt.show()
        
        y     = np.append(y,[])     #list to array 
        ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((samplestrn, features))
        Xtst  = Xtst.reshape((samplestrn, features))
        
    
    elif(case == 4):
        # 1 dimension nonlinear regression  quadratic  for  f1 & Type 4
        samplestrn    = 100 
        samplestst    = 100
        features      = 1
        outliers      = 10
        X    = np.arange(-5, 5, 10/samplestrn)
        y    = X + 2 * X ** 2 + np.random.normal(0,0.5, samplestrn)  #N(0,1) noise
        l    =  random.sample(range(1,100), outliers)
        y[l]  += np.random.normal(20,1, outliers)
        # y[l]  += np.random.normal(50,10, outliers)
        Xtst = np.arange(-3, 3, 6/samplestst)
        ytst = Xtst + 2 * Xtst ** 2 + np.random.normal(0,0.5, samplestst) #N(0,1) noise
        y    = y.reshape([y.shape[0], 1])
        ytst = ytst.reshape([ytst.shape[0], 1])
        
        X_out = X[l]
        y_out = y[l]
        
        plt.plot
        plt.figure(figsize=(8, 8))
        X0    = np.arange(-5, 5, 0.1)
        y0    = X0 + 2 * X0 ** 2 
        plt.plot(X0, y0,c='red',label='Groud Truth')
        plt.scatter(X, y, s=10, label='Main Data')
        plt.scatter(X_out, y_out, s=10, label="Outliers")
        plt.legend()
        plt.show()
        
        # y     = np.append(y,[])     #list to array 
        # ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((samplestrn, features))
        Xtst  = Xtst.reshape((samplestrn, features))
        
    elif(case == 5):
        # 1 dimension nonlinear regression   sinc  for  f2
        samplestrn    = 100 
        samplestst    = 100
        features      = 1
        outliers      = 10
        X    = np.arange(-4, 4, 8/samplestrn)
        y    = np.sin(math.pi*X)/(math.pi*X) + np.random.normal(0,0.1, samplestrn)
        l    =  random.sample(range(1,100), outliers)
        # y[l] +=  np.random.normal(20,1, outliers)
        # y[l] +=  np.random.normal(50,10, outliers)
        y[l] +=  np.random.standard_t(2, size=outliers)*10
        Xtst = np.arange(-4, 4, 8/samplestst)
        ytst = np.sin(math.pi*X)/(math.pi*X) + np.random.normal(0,0.1, samplestrn)
        y    = y.reshape([y.shape[0], 1])
        ytst = ytst.reshape([ytst.shape[0], 1])
        
        X_out = X[l]
        y_out = y[l]
        
        plt.plot
        plt.figure(figsize=(8, 8))
        X0    = np.arange(-4, 4, 0.01)
        y0    = np.sin(math.pi*X0)/(math.pi*X0) 
        plt.plot(X0, y0,c='red',label="Ground Truth")
        plt.scatter(X, y, s=10, label='Main Data')
        plt.scatter(X_out, y_out, s=10, label="Outliers")
        plt.legend()
        plt.show()
        
        y     = np.append(y,[])     #list to array 
        ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((samplestrn, features))
        Xtst  = Xtst.reshape((samplestrn, features))
        

        
    elif(case == 6):
        # Friedman nonlinear high-dimensional data  for f3
        samplestrn   = 1000
        samplestst   = 1000
        outlier_prop = 0.1
        num_outlier  = int(samplestrn*outlier_prop)
    
        X, y = datasets.make_friedman2(n_samples=samplestrn,random_state=1, noise=0)
        Xtst, ytst = datasets.make_friedman2(n_samples=samplestst,random_state=1, noise=0)
        features = X.shape[1]
        
        if(noise_type == 'gaussian'):
            y   += np.random.normal(0,1, samplestrn)
        else:
            y   += np.random.standard_t(2, size=samplestrn)
            
        l     =  random.sample(range(1,samplestrn), num_outlier)
        y[l] +=  np.random.standard_t(2, size=num_outlier)*1000
        
        y     = np.append(y,[])     #list to array 
        ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((samplestrn, features))
        Xtst  = Xtst.reshape((samplestst, features))
        

        
    elif(case == 7):
        # Friedman nonlinear high-dimensional data  for f4
        samplestrn   = 1000
        samplestst   = 1000
        outlier_prop = 0.1
        num_outlier  = int(samplestrn*outlier_prop)
        
        X, y = datasets.make_friedman3(n_samples=samplestrn,random_state=1, noise=0)
        Xtst, ytst = datasets.make_friedman3(n_samples=samplestst,random_state=1, noise=0)
        features = X.shape[1]
        
        if(noise_type == 'gaussian'):
            y   += np.random.normal(0,1, 1000)
        else:
            y   += np.random.standard_t(2, size=1000)
        
        l     =  random.sample(range(1,samplestrn), num_outlier)
        y[l] +=  np.random.standard_t(2, size=num_outlier)*1000
        
        y     = np.append(y,[])     #list to array 
        ytst  = np.append(ytst,[])  #list to array 
        X     = X.reshape((samplestrn, features))
        Xtst  = Xtst.reshape((samplestrn, features))

    
    return X, y, Xtst, ytst

# CreateData(case=3,noise_type = 'gaussian')