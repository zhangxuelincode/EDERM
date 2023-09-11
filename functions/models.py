import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from functions.indicators import indicator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size':20})


def abline(a,  label_, c=None ):
    plt.rcParams.update({'font.size':20})
    axes       = plt.gca()
    x_vals     = np.array(axes.get_xlim())
    x_vals     = x_vals.reshape((x_vals.shape[0], 1))
    intercept  = np.ones((x_vals.shape[0], 1))
    x_vals     = np.concatenate((x_vals, intercept), axis=1) 
    y_vals     = np.dot(x_vals , a)
    plt.plot(x_vals[:,0:-1], y_vals, label=label_, color=c)
    

def kerneldensity(loss,fai): 
    ''' 
        loss:   y-y_pred  
    '''
    lossinds    = loss.argsort()
    sorted_loss = loss[lossinds[:]]
    x2          = sorted_loss
    y2          = fai[lossinds[:]]   # final ρ
    plt.figure(figsize=(8, 8))
    plt.plot(x2,y2,label='Final density',linewidth=2,color='r',marker='o', markerfacecolor='blue',markersize=4)
    plt.hist(loss, bins = 20,density = True)
    plt.grid(True) 
    plt.xlabel('Error Value') 
    plt.ylabel('Density') 
    plt.legend() 
    plt.show()


def mode_0(X, y, Xtst, ytst,iters,learning_step,tol,losstype,h,delta,online):
    '''
        ERM regression with square loss
        We have the following
            F = || Y - XW || 2,2
           dF = X.T * (Y - XW)
    '''
    #init params
    intercept  = np.ones((X.shape[0], 1))
    X_         = np.concatenate((X, intercept), axis=1)
    intercept  = np.ones((Xtst.shape[0], 1))
    Xtst_      = np.concatenate((Xtst, intercept), axis=1)
    n          = X_.shape[0]
    features   = X_.shape[1]
    theta_hat  = np.zeros(features)
    for k in range(iters):
        grad   =  np.zeros(features)
        loss   =  np.zeros(n)
        dloss  =  np.zeros([n,features])
        y_pred =  np.dot(X_, theta_hat)
        error  =  (y-y_pred) 
        olderr = mse(y,y_pred)
        for i in range(0,n):
            if(losstype == 'mse'):
                loss[i] = error[i]**2
                dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
            elif(losstype == 'closs'):
                loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
            grad    +=  dloss[i] / n
        
        theta_hat = theta_hat - learning_step * grad
        newerr = mse(y,np.dot(X_, theta_hat))
        if(online == True):
            print("iterations",k,"Errors",newerr,"test_R2",r2(ytst,np.dot(Xtst_,theta_hat)))
        if(abs(newerr - olderr) < tol):
            break
    
    print("training set")    
    y_pred = np.dot(X_, theta_hat)
    loss = (y-y_pred)* (y-y_pred) 
    print(" max loss: {}, min loss: {}, avg loss: {}, variance: {}".format( max(loss), min(loss), np.mean(loss), np.var(loss)))
    
    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.scatter(X, y, s=2, label='data distribution')
        abline(theta_hat,  'ERM', "red")
        plt.grid(True) 
        plt.legend()

    print("test set")    
    ytst_pred = np.dot(Xtst_,theta_hat)
    loss = (ytst-ytst_pred)* (ytst-ytst_pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format( max(loss), min(loss), np.mean(loss), np.var(loss)))
    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.scatter(Xtst, ytst, s=2, label='Data Distribution')
        abline(theta_hat,  'ERM', "red")
        plt.grid(True) 
        plt.legend()

def mode_1(X, y, Xtst, ytst,iters,lamb,learning_step,tol,Ictype,losstype,h,delta,online,flag='LS',optimizer='fgd',batchsize=10):
    '''
        correntropy-Induced approximated Indicator single mode
        linear regression y = x*w + b
        λ<ρ restrict the indicator function I  = I(ρ>λ)
        We have the following
            F = sum(f(zi) * I)/n = sum(f(zi) * I(ρ>λ))/n 
           dF = sum(df*I + f*dI )/n 
    '''
    #init params
    intercept  = np.ones((X.shape[0], 1))
    X_         = np.concatenate((X, intercept), axis=1)
    intercept  = np.ones((Xtst.shape[0], 1))
    Xtst_      = np.concatenate((Xtst, intercept), axis=1)
    n          = X_.shape[0]
    features   = X_.shape[1]
    theta_hat  = np.zeros(features)
    
    #for adagrad
    eps=0.00001
    adagrad = np.zeros(features)
    # for adam
    n, dim = X_.shape
    b1 = 0.9  
    b2 = 0.999  
    e = 0.00000001  
    mt = np.zeros(dim)
    vt = np.zeros(dim)
    #for AmsGD
    v=0
    cache=0
    #for rmsprop
    cache = np.zeros(features)
    e = 0.00000001
    #for NAG
    v = np.zeros(features)
    
    for k in range(iters):
        Ic     =  np.zeros(n)
        R      =  np.zeros(iters)   
        mainx  =  []  #save important points
        mainy  =  []
        grad   =  np.zeros(features)
        fai    =  np.zeros(n) #final error density

        dfai   =  np.zeros([n,features])
        loss   =  np.zeros(n)
        dloss  =  np.zeros([n,features])
        Ic     =  np.zeros(n)
        Points =  np.zeros(n)
        dIc    =  np.zeros([n,features])
        y_pred =  np.dot(X_, theta_hat)
        error  =  (y-y_pred) 
        errors =  (y-y_pred)* (y-y_pred) 
        olderr = mse(y,y_pred)
        if(optimizer == 'fgd'):
            for i in range(0,n):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0   
            theta_hat = theta_hat - learning_step * grad 
        elif(optimizer == 'sgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            # index = np.random.randint(0, n)
            # grad    =  dloss[index]*Ic[index]+loss[index]*dIc[index] 
            theta_hat = theta_hat - learning_step * grad 
        elif(optimizer == 'adaGrad'):
            for i in np.random.randint(0, n, 10):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            gradient2=grad*grad
            adagrad+=gradient2
            theta_hat = theta_hat - learning_step*n*grad/np.sqrt(adagrad + eps)
        elif(optimizer == 'adam'):
            for i in np.random.randint(0, n, 10):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * ((error[i]-error[j])/(h*h)) *(X_[i].T)
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * (grad**2)
            mtt = mt / (1 - (b1**(k + 1)))
            vtt = vt / (1 - (b2**(k + 1)))
            vtt_sqrt = np.array([math.sqrt(vtt[0]),
                                 math.sqrt(vtt[1])]) 
            theta_hat = theta_hat - learning_step * mtt / (vtt_sqrt + e)
        elif(optimizer == 'amsgd'):
            for i in np.random.randint(0, n, 10):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * ((error[i]-error[j])/(h*h)) *(X_[i].T)
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
    
            mu = 0.9
            decay_rate = 0.999
            eps = 1e-8
            v = mu*v + (1-mu)*grad
            vt = v/(1-mu**(k+1))
            cache = decay_rate *cache +(1-decay_rate )* (grad**2)
            cachet = cache/(1-decay_rate**(k+1))
            theta_hat = theta_hat - (learning_step  / (np.array([math.sqrt(cachet[0]),math.sqrt(cachet[1])]) + eps))*vt
        elif(optimizer == 'rmsprop'):
            for i in np.random.randint(0, n, 10):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * ((error[i]-error[j])/(h*h)) *(X_[i].T)
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
    
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
                
            decay_rate = 0.9
            cache = decay_rate * cache + (1 - decay_rate) * grad**2
            cache_sqrt = np.array([math.sqrt(cache[0]), math.sqrt(cache[1])])  
            theta_hat = theta_hat - learning_step *  grad / (cache_sqrt + e)
        elif(optimizer == 'NAG'):
            for i in np.random.randint(0, n, 10):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))* ((error[i]-error[j])/(h*h)) *(X_[i].T)
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
    
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
                
            #nesterov_momentumGD   
            mu = 0.99 
            pre_v = v
            v = mu*v # 1
            v +=  - learning_step*grad # 2
            #x += v + mu*(v - pre_v) # 3
            theta_hat = v + mu*(v - pre_v) # 3
    
        newerr = mse(y,np.dot(X_, theta_hat))
        if(online == True):
            print("Iterations",k,"Training Errors",newerr,"test_R2",r2(ytst,np.dot(Xtst_,theta_hat)))
        if(abs(newerr - olderr) < tol):
            break
        
        
    #show density info
    kerneldensity(error,fai)
    
    print("training set")
    y_pred = np.dot(X_, theta_hat)
    loss = (y-y_pred)* (y-y_pred)
    print("ρ>{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb, max(loss), min(loss), np.mean(loss), np.var(loss)))

    if(features==2):
        plt.figure(figsize=(8, 8))
        # plt.title("Training Set: λ = {} error = {}".format(lamb, np.mean(loss)),fontsize= 'xx-large')
        plt.scatter(X, y, s=2, label='Data Distribution')
        if(flag=='LS'):
            plt.scatter(mainx, mainy, s=2, label="Main Points")
            abline(theta_hat,  'EDERM', "red")
        else:
            abline(theta_hat,  'Correntropy', "red")
        plt.grid(True)
        plt.legend()

    print("test set")    
    ytst_pred = np.dot(Xtst_,theta_hat)
    loss = (ytst-ytst_pred)* (ytst-ytst_pred)
    print("ρ>{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb, max(loss), min(loss), np.mean(loss), np.var(loss)))
    # print("max error density: {}, min ED: {}， average ED: {}".format(max(fai), min(fai), np.mean(fai)))
    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.scatter(Xtst, ytst, s=2, label='Data distribution')
        if(flag=='LS'):
            abline(theta_hat,  'EDERM', "red")
        else:
            abline(theta_hat,  'Correntropy', "red")
        plt.grid(True) 
        plt.legend()
    print("The linear weight: ",theta_hat[-2],"the constrant:",theta_hat[-1])


def mode_2(X, y, Xtst, ytst,iters,lamb,lamb2,learning_step,tol,Ictype,losstype,h,delta,online,optimizer='fgd',batchsize=10):
    '''
        correntropy-Induced approximated Indicator double mode
        linear regression y = x*w + b
        λ1<ρ<λ2 restrict the indicator function I  = I(ρ>λ1) - I(ρ>λ2) 
        We have the following
            F = sum(f(zi) * I)/n = sum(f(zi) * (I(ρ>λ1) - I(ρ>λ2) ))/n 
           dF = sum(df*I + f*dI )/n 
              = sum(df*(I(ρ>λ1) - I(ρ>λ2) ) + f*(dI(ρ>λ1) - dI(ρ>λ2) ))/n
    '''

    #init params
    intercept  = np.ones((X.shape[0], 1))
    X_         = np.concatenate((X, intercept), axis=1)
    intercept  = np.ones((Xtst.shape[0], 1))
    Xtst_      = np.concatenate((Xtst, intercept), axis=1)
    n          = X_.shape[0]
    features   = X_.shape[1]
    theta_hat  = np.zeros(features)
    
    #for adagrad
    eps=0.00001
    adagrad = np.zeros(features)
    # for adam
    n, dim = X_.shape
    b1 = 0.9  
    b2 = 0.999  
    e = 0.00000001  
    mt = np.zeros(dim)
    vt = np.zeros(dim)
    #for AmsGD
    v=0
    cache=0
    #for rmsprop
    cache = np.zeros(features)
    e = 0.00000001
    #for NAG
    v = np.zeros(features)
    
    for k in range(iters):
        R      =  np.zeros(iters)
        mainx  =  []
        mainy  =  [] 
        grad    =  np.zeros(features)
        fai     =  np.zeros(n) #final error density

        dfai    =  np.zeros([n,features])
        loss    =  np.zeros(n)
        dloss   =  np.zeros([n,features])
        Ic      =  np.zeros(n)
        Ic1     =  np.zeros(n)
        Ic2     =  np.zeros(n)
        dIc     =  np.zeros([n,features])
        dIc1    =  np.zeros([n,features])
        dIc2    =  np.zeros([n,features])
        y_pred  = np.dot(X_, theta_hat)
        error   = (y-y_pred) 
        errors  = (y-y_pred)* (y-y_pred) 
        Points =  np.zeros(n)
        olderr = mse(y,y_pred)
        if(optimizer == 'fgd'):
            for i in range(0,n):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                if(fai[i]>lamb and fai[i]<lamb2):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0   
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i]
            theta_hat = theta_hat -learning_step * grad 
        elif(optimizer == 'sgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            # index = np.random.randint(0, n)
            # grad    =  dloss[index]*Ic[index]+loss[index]*dIc[index] 
            theta_hat = theta_hat - learning_step * grad 
        elif(optimizer == 'adaGrad'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            gradient2=grad*grad
            adagrad+=gradient2
            theta_hat = theta_hat - learning_step*n*grad/np.sqrt(adagrad + eps)
        elif(optimizer == 'adam'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * (grad**2)
            mtt = mt / (1 - (b1**(k + 1)))
            vtt = vt / (1 - (b2**(k + 1)))
            vtt_sqrt = np.array([math.sqrt(vtt[0]),
                                 math.sqrt(vtt[1])]) 
            theta_hat = theta_hat - learning_step * mtt / (vtt_sqrt + e)
        elif(optimizer == 'amsgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            mu = 0.9
            decay_rate = 0.999
            eps = 1e-8
            v = mu*v + (1-mu)*grad
            vt = v/(1-mu**(k+1))
            cache = decay_rate *cache +(1-decay_rate )* (grad**2)
            cachet = cache/(1-decay_rate**(k+1))
            theta_hat = theta_hat - (learning_step  / (np.array([math.sqrt(cachet[0]),math.sqrt(cachet[1])]) + eps))*vt
        elif(optimizer == 'rmsprop'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            decay_rate = 0.9
            cache = decay_rate * cache + (1 - decay_rate) * grad**2
            cache_sqrt = np.array([math.sqrt(cache[0]), math.sqrt(cache[1])])  
            theta_hat = theta_hat - learning_step *  grad / (cache_sqrt + e)
        elif(optimizer == 'NAG'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * \
                        (error[i]-error[j])/(h*h) *(X_[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                    dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                Ic1[i],dIc1[i]     = indicator(Ictype, fai[i], dfai[i], lamb,delta)
                Ic2[i],dIc2[i]     = indicator(Ictype, fai[i], dfai[i], lamb2,delta)
                Ic[i]    = Ic1[i]-Ic2[i]
                dIc[i]   = dIc1[i]-dIc2[i]
                R[k]    +=  errors[i]*Ic[i]
                grad    +=  dloss[i]*Ic[i]+loss[i]*dIc[i] 
            mu = 0.99 
            pre_v = v
            v = mu*v # 1
            v +=  - learning_step*grad # 2
            #x += v + mu*(v - pre_v) # 3
            theta_hat = v + mu*(v - pre_v) # 3    
        newerr = mse(y,np.dot(X_, theta_hat))
        if(online == True):
            print("Iterations",k,"Training Errors",newerr,"test_R2",r2(ytst,np.dot(Xtst_,theta_hat)))
        if(abs(newerr - olderr) < tol):
            break

    kerneldensity(error,fai)
        
    print("training set")    
    loss = (y-y_pred)**2
    print("{}<ρ<{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb,lamb2, max(loss), min(loss), np.mean(loss), np.var(loss)))
    
    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.title("Training Set:  {}<λ<{} ".format(lamb,lamb2),fontsize= 'xx-large')
        plt.scatter(X, y, s=2, label='Data Distribution')
        plt.scatter(mainx, mainy, s=2, label="Main Points")
        abline(theta_hat, 'EDERM', "red")
        plt.grid(True) 
        plt.legend()
     
    print("test set")    
    ytst_pred = np.dot(Xtst_,theta_hat)
    loss = (ytst-ytst_pred)* (ytst-ytst_pred)
    print("{}<ρ<{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb,lamb2, max(loss), min(loss), np.mean(loss), np.var(loss)))


    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.title("Test Set:  {}<λ<{} ".format(lamb,lamb2),fontsize= 'xx-large')
        plt.scatter(Xtst, ytst, s=2, label='Data Distribution')
        abline(theta_hat, 'EDERM', "red")
        plt.grid(True) 
        plt.legend()
        

def mode_3(X, y, Xtst, ytst,iters,lamb,learning_step,tol,losstype,h,delta,online):
    '''
        0/1 Indicator   single mode
        linear regression y = x*w + b
        λ<ρ restrict the indicator function I  = I(ρ>λ)    I=1/0
        We have the following
            F = sum(f(zi) * I)/n = sum(f(zi) * I(ρ>λ))/n 
           dF = sum(df*I )/n 
    '''
    
    #init params
    intercept  = np.ones((X.shape[0], 1))
    X_         = np.concatenate((X, intercept), axis=1)
    intercept  = np.ones((Xtst.shape[0], 1))
    Xtst_      = np.concatenate((Xtst, intercept), axis=1)
    # X_         = X
    # Xtst_      = Xtst
    n          = X_.shape[0]
    features   = X_.shape[1]
    theta_hat  = np.zeros(features)
    for k in range(iters):
        Ic     =  np.zeros(n)
        mainx  =  []  #save important points
        mainy  =  []
        grad   =  np.zeros(features)
        fai    =  np.zeros(n) #final error density

        loss   =  np.zeros(n)
        dloss   =  np.zeros([n,features])
        Ic     =  np.zeros(n)
        Points =  np.zeros(n)
        y_pred =  np.dot(X_, theta_hat)
        error  =  (y-y_pred) 
        olderr = mse(y,y_pred)
        for i in range(0,n):
            for j in range(0,n):
                fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
            if(losstype == 'mse'):
                loss[i] = error[i]**2
                dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
            elif(losstype == 'closs'):
                loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
                
            if(fai[i]>lamb):
                mainx.append(X[i])
                mainy.append(y[i])
                Ic[i]=1
                Points[i]=1
            
            grad    += dloss[i] + 0;
        theta_hat = theta_hat - learning_step * grad 
        newerr = mse(y,np.dot(X_, theta_hat))
        if(online == True):
            print("Iterations",k,"Training Errors",newerr,"test_R2",r2(ytst,np.dot(Xtst_,theta_hat)))
        if(abs(newerr - olderr) < tol):
            break
    kerneldensity(error,fai)
    
    print("training set")    
    y_pred = np.dot(X_, theta_hat)
    loss = (y-y_pred)* (y-y_pred) 
    print("ρ>{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb, max(loss), min(loss), np.mean(loss), np.var(loss)))
    
        
    if(features==2):
        plt.figure(figsize=(8, 8))
        # plt.title("Training Set: λ = {}".format(lamb),fontsize= 'xx-large')
        plt.scatter(X, y, s=2, label='Data Distribution')
        plt.scatter(mainx, mainy, s=2, label="Main Points")
        abline(theta_hat, 'EDERM', "red")
        plt.grid(True) 
        plt.legend()

    print("test set")    
    ytst_pred = np.dot(Xtst_,theta_hat)
    loss = (ytst-ytst_pred)* (ytst-ytst_pred)
    print("ρ>{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb, max(loss), min(loss), np.mean(loss), np.var(loss)))
    
    if(features==2):
        plt.figure(figsize=(8, 8))
        # plt.title("Test Set: λ = {}".format(lamb),fontsize= 'xx-large')
        plt.scatter(Xtst, ytst, s=2, label='Data Distribution')
        abline(theta_hat, 'EDERM', "red")
        plt.grid(True) 
        plt.legend()
    
    # return error,theta_hat,fai,Ic,Points,mainx,mainy

def mode_4(X, y, Xtst, ytst,iters,lamb,lamb2,learning_step,tol,losstype,h,delta,online):
    '''
        0/1  Indicator  double mode
        linear regression y = x*w + b
        λ1<ρ<λ2 restrict the indicator function I  = I(λ1<ρ<λ2) = 1/0
        We have the following
            F = sum(f(zi) * I(λ1<ρ<λ2))/n 
           dF = sum(df*I)/n 
    '''
    
    #init params
    intercept  = np.ones((X.shape[0], 1))
    X_         = np.concatenate((X, intercept), axis=1)
    intercept  = np.ones((Xtst.shape[0], 1))
    Xtst_      = np.concatenate((Xtst, intercept), axis=1)
    n          = X_.shape[0]
    features   = X_.shape[1]
    theta_hat  = np.ones(features)
    Ic     =  np.zeros(n)
    error  =  np.zeros(n)
    mainx  =  []
    mainy  =  []        
    for k in range(iters):
        grad    =  np.zeros(features)
        fai     =  np.zeros(n)
        #
        dfai    =  np.zeros([n,features])
        loss    =  np.zeros(n)
        dloss   =  np.zeros([n,features])
        Ic      =  np.zeros(n)
        y_pred  = np.dot(X_, theta_hat)
        error   = (y-y_pred) 
        Points =  np.zeros(n)
        olderr = mse(y,y_pred)
        for i in range(0,n):
            for j in range(0,n):
                fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
            if(losstype == 'mse'):
                loss[i] = error[i]**2
                dloss[i]= (2*X_[i].T*(y_pred[i]-y[i]))
            elif(losstype == 'closs'):
                loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta))) 
                dloss[i]= ( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])
            if(fai[i]>lamb and fai[i]<lamb2):
                Ic[i]=1
                Points[i]=1
                mainx.append(X[i])
                mainy.append(y[i])  
            grad    += 2*X[i].T*(y_pred[i]-y[i])*Ic[i];
            
        theta_hat = theta_hat -learning_step * grad
        newerr = mse(y,np.dot(X_, theta_hat))
        if(online == True):
            print("Iterations",k,"Training Errors",newerr,"test_R2",r2(ytst,np.dot(Xtst_,theta_hat)))
        if(abs(newerr - olderr) < tol):
            break
    kerneldensity(error,fai)
    
    print("training set")  
    y_pred = np.dot(X_, theta_hat)
    loss = (y-y_pred)* (y-y_pred) 
    print("{}<ρ<{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb,lamb2, max(loss), min(loss), np.mean(loss), np.var(loss)))
    
    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.title("Training Set:  {}<λ<{} ".format(lamb,lamb2),fontsize= 'xx-large')
        plt.scatter(X, y, s=2, label='Data Distribution')
        plt.scatter(mainx, mainy, s=2, label="Main Points")
        abline(theta_hat, 'EDERM', "red")
        plt.grid(True) 
        plt.legend()
     
    print("test set")    
    ytst_pred = np.dot(Xtst_,theta_hat)
    loss = (ytst-ytst_pred)* (ytst-ytst_pred)
    print("{}<ρ<{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb,lamb2, max(loss), min(loss), np.mean(loss), np.var(loss)))

    if(features==2):
        plt.figure(figsize=(8, 8))
        plt.title("Test Set:  {}<λ<{} ".format(lamb,lamb2),fontsize= 'xx-large')
        plt.scatter(Xtst, ytst, s=2, label='Data Distribution')
        abline(theta_hat, 'EDERM', "red")
        plt.grid(True) 
        plt.legend()


def mode_5(X, y, Xtst, ytst,dimensions,iters,lamb,learning_step,tol,Ictype,losstype,h,delta,online,optimizer='fgd',batchsize=10):
    '''
        Poly-based non-linear regression
    '''
    poly          = PolynomialFeatures(dimensions) # change the degree to meet the features of data. 
    X_            = poly.fit_transform(np.array(X)/2)
    Xtst_         = poly.fit_transform(np.array(Xtst)/2)

    n = X_.shape[0]
    features = X_.shape[1]
    theta_hat = np.zeros(features)
    
    #for adagrad
    eps=0.00001
    adagrad = np.zeros(features)
    # for adam
    n, dim = X_.shape
    b1 = 0.9  
    b2 = 0.999  
    e = 0.00000001  
    mt = np.zeros(dim)
    vt = np.zeros(dim)
    #for AmsGD
    v=0
    cache=0
    #for rmsprop
    cache = np.zeros(features)
    e = 0.00000001
    #for NAG
    v = np.zeros(features)
    
    for k in range(iters):
        Ic = np.zeros(n)
        mainx = []  # save important points
        mainy = []
        grad  = np.zeros(features)
        fai   = np.zeros(n)  # final error density

        dfai  = np.zeros([n, features])
        loss  = np.zeros(n)
        dloss = np.zeros([n, features])
        # dloss = []
        Ic = np.zeros(n)
        Points = np.zeros(n)
        dIc = np.zeros([n, features])
        y_pred = np.dot(X_, theta_hat.reshape(features,1))
        error = (y - y_pred)
        old_error = mse(y,y_pred)
        
        if(optimizer == 'fgd'):
            for i in range(0, n):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0
            theta_hat = theta_hat - learning_step * grad / n
        elif(optimizer == 'sgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0
            theta_hat = theta_hat - learning_step * grad  / n
        elif(optimizer == 'adaGrad'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0
            gradient2=grad*grad
            adagrad+=gradient2
            theta_hat = theta_hat - learning_step*n*grad/np.sqrt(adagrad + eps)
        elif(optimizer == 'adam'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * (grad**2)
            mtt = mt / (1 - (b1**(k + 1)))
            vtt = vt / (1 - (b2**(k + 1)))
            vtt_sqrt = np.array([math.sqrt(vtt[0]),
                                 math.sqrt(vtt[1])]) 
            theta_hat = theta_hat - learning_step * mtt / (vtt_sqrt + e)
        elif(optimizer == 'amsgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0
            mu = 0.9
            decay_rate = 0.999
            eps = 1e-8
            v = mu*v + (1-mu)*grad
            vt = v/(1-mu**(k+1))
            cache = decay_rate *cache +(1-decay_rate )* (grad**2)
            cachet = cache/(1-decay_rate**(k+1))
            theta_hat = theta_hat - (learning_step  / (np.array([math.sqrt(cachet[0]),math.sqrt(cachet[1])]) + eps))*vt
        elif(optimizer == 'rmsprop'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0
            decay_rate = 0.9
            cache = decay_rate * cache + (1 - decay_rate) * grad**2
            cache_sqrt = np.array([math.sqrt(cache[0]), math.sqrt(cache[1])])  
            theta_hat = theta_hat - learning_step *  grad / (cache_sqrt + e)
        elif(optimizer == 'NAG'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0, n):
                    fai[i] = fai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) * (error[i]-error[j])/(h*h) *(X[j].T-X[i].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= (2*X_[i].T*(y_pred[i]-y[i])).reshape(features)
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (( math.exp(error[i]**2 / (-2*delta*delta)))  * X_[i].T*(y_pred[i]-y[i])).reshape(features)
                dfai[i] = dfai[i].T
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad += dloss[i] * Ic[i] + loss[i] * dIc[i]
                if (fai[i] > lamb):
                    Points[i] = 1
                    mainx.append(X_[i])
                    mainy.append(y[i])
                else:
                    Points[i] = 0 
            mu = 0.99 
            pre_v = v
            v = mu*v # 1
            v +=  - learning_step*grad # 2
            #x += v + mu*(v - pre_v) # 3
            theta_hat = v + mu*(v - pre_v) # 3    
        
        y_pred = np.dot(X_,theta_hat).reshape(len(y),1)
        new_error = mse(y,y_pred)
        if(online == True):
            print("Iterations",k,"Training Errors",new_error,"test_R2",r2(ytst,np.dot(Xtst_,theta_hat)))
        if abs(new_error - old_error) <= tol:
            break
    ytst_pred  = np.dot(Xtst_,theta_hat).reshape(Xtst_.shape[0],1) 

    loss_train = mse(y,y_pred)
    loss_test  = mse(ytst,ytst_pred)
    print( "Trn Error: "+str(loss_train))
    print( "Tst Error: "+str(loss_test))
    
    plt.figure("Training Set")
    plt.figure(figsize=(8, 8))
    plt.scatter(X,y,marker='o',color='red',label="Training data")
    plt.plot(X,y_pred,'b',label = 'EDERM') 
    plt.legend(loc=4)# make legend
    plt.show()
    
    plt.figure("Test Set")
    plt.figure(figsize=(8, 8))
    plt.scatter(Xtst,ytst,marker='o',color='r',label="Test data")
    plt.plot(Xtst,ytst_pred,'b',label = 'EDERM')
    plt.legend(loc=4)
    plt.show()
    
def kernel_function(kernel,delta,Xtrn,X):
    gram = np.zeros([X.shape[0],Xtrn.shape[0]])
    if(X.shape[1]==1):
        if(kernel == 'gaussian'):
            for i in range(X.shape[0]):
                for j in range(Xtrn.shape[0]):  # gram test_samples * trn_samples
                    gram[i,j] = math.exp( (X[i]-Xtrn[j])**2 /(-2*delta**2) )
            return gram
    else:
        if(kernel == 'gaussian'):
            for i in range(X.shape[0]):
                for j in range(Xtrn.shape[0]):  # gram test_samples * trn_samples
                    gram[i,j] = math.exp( np.linalg.norm(X[i]-Xtrn[j])**2 /(-2*delta**2) )
            return gram
    
def mode_6(X, y, Xtst, ytst,kernel,delta_kernel,iters,lamb,learning_step,tol,Ictype,losstype,h,delta,online,optimizer='fgd',batchsize=10):
    '''
        Kernel regression, here we focus on gaussian regression with gradient descent method.
        We compare the performance of our proposed EDERM with traditional LSE and correntropy loss.
    '''    
    gram       = kernel_function('gaussian',delta_kernel,X,X).reshape(len(y),len(y))
    #init params
    n          = X.shape[0]
    features   = X.shape[0]
    alpha  = np.zeros(X.shape[0])

    #for adagrad
    eps=0.00001
    adagrad = np.zeros(features)
    # for adam
    n, dim = gram.shape
    b1 = 0.9  
    b2 = 0.999  
    e = 0.00000001  
    mt = np.zeros(dim)
    vt = np.zeros(dim)
    #for AmsGD
    v=0
    cache=0
    #for rmsprop
    cache = np.zeros(features)
    e = 0.00000001
    #for NAG
    v = np.zeros(features)
    
    for iterations in range(iters):
        Ic     =  np.zeros(n)
        mainx  =  []  #save important points
        mainy  =  []
        grad   =  np.zeros(n)
        fai    =  np.zeros(n) #final error density

        dfai   =  np.zeros([n,n])
        loss   =  np.zeros(n)
        dloss  =  np.zeros([n,n])
        Ic     =  np.zeros(n)
        Points =  np.zeros(n)
        dIc    =  np.zeros([n,n])
        y_pred =  np.dot(gram, alpha)
        error  =  (y-y_pred)
        old_error = mse(y,y_pred)
        if(optimizer == 'fgd'):
            for i in range(0,n):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0   
            grad = grad.reshape([n,1])
            alpha = alpha - learning_step * grad
        elif(optimizer == 'sgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0  
            alpha = alpha - learning_step * grad
        elif(optimizer == 'adaGrad'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0 
            gradient2=grad*grad
            adagrad+=gradient2
            alpha = alpha - learning_step*n*grad/np.sqrt(adagrad + eps)
        elif(optimizer == 'adam'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0 
            mt = b1 * mt + (1 - b1) * grad
            vt = b2 * vt + (1 - b2) * (grad**2)
            mtt = mt / (1 - (b1**(iterations + 1)))
            vtt = vt / (1 - (b2**(iterations + 1)))
            vtt_sqrt = np.array([math.sqrt(vtt[0]),
                                 math.sqrt(vtt[1])]) 
            alpha = alpha - learning_step * mtt / (vtt_sqrt + e)
        elif(optimizer == 'amsgd'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0 
            mu = 0.9
            decay_rate = 0.999
            eps = 1e-8
            v = mu*v + (1-mu)*grad
            vt = v/(1-mu**(iterations+1))
            cache = decay_rate *cache +(1-decay_rate )* (grad**2)
            cachet = cache/(1-decay_rate**(iterations+1))
            alpha = alpha - (learning_step  / (np.array([math.sqrt(cachet[0]),math.sqrt(cachet[1])]) + eps))*vt
        elif(optimizer == 'rmsprop'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0 
            decay_rate = 0.9
            cache = decay_rate * cache + (1 - decay_rate) * grad**2
            cache_sqrt = np.array([math.sqrt(cache[0]), math.sqrt(cache[1])])  
            alpha  = alpha - learning_step *  grad / (cache_sqrt + e)
        elif(optimizer == 'NAG'):
            for i in np.random.randint(0, n, batchsize):
                for j in range(0,n):
                    fai[i]  = fai[i]  + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h))
                    dfai[i] = dfai[i] + (1/(n*h))* math.exp((error[i]-error[j])**2/(-2*h*h)) \
                        * ((error[i]-error[j])/(h*h)) *(gram[i,:].T)
                if(losstype == 'mse'):
                    loss[i] = error[i]**2
                    dloss[i]= 2*gram[i,:].T * (y_pred[i]-y[i])
                elif(losstype == 'closs'):
                    loss[i] = delta * delta * (1 - math.exp(error[i]**2 / (-2*delta*delta)))
                    dloss[i]= (math.exp(error[i]**2 / (-2*delta*delta)))  * gram[i,:].T * (y_pred[i]-y[i])
                Ic[i],dIc[i]     = indicator(Ictype, fai[i], dfai[i], lamb, delta)
                grad    +=  dloss[i]*Ic[i]  +  loss[i]*dIc[i]
                if(fai[i]>lamb):
                    Points[i]=1
                    mainx.append(X[i])
                    mainy.append(y[i])
                else:
                    Points[i]=0 
            mu = 0.99 
            pre_v = v
            v = mu*v # 1
            v +=  - learning_step*grad # 2
            #x += v + mu*(v - pre_v) # 3
            alpha = v + mu*(v - pre_v) # 3    
        
        alpha_hat = alpha
        y_pred =  np.dot(gram, alpha_hat).reshape(len(y),1)
        new_error = mse(y,y_pred)
        # if abs(new_error - old_error) <= tol:
        #     break
        # if(online == True):
        #     print("iteration: ",iterations,"   Error: ",new_error)
        if(online == True):
            print("Iterations",iterations,"Training Errors",new_error,"test_R2",r2(ytst,np.dot(kernel_function('gaussian',delta_kernel,X,Xtst),alpha_hat).reshape(len(ytst),1)))
        if abs(new_error - old_error) <= tol:
            break
                
    print("training set")
    y_pred =  np.dot(gram, alpha_hat).reshape(len(y),1)
    loss = (y-y_pred)* (y-y_pred)
    print(np.mean(loss))
    # print("ρ>{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb, max(loss), min(loss), np.mean(loss), np.var(loss)))
   
    plt.figure(figsize=(8, 8))
    plt.scatter(X,y,marker='o',color='r',label ="Training Data")
    plt.plot(X,y_pred, label="EDERM")
    plt.legend()
    plt.show()

    print("test set")
    gram_tst = kernel_function('gaussian',delta_kernel,X,Xtst)
    ytst_pred = np.dot(gram_tst, alpha_hat).reshape(len(ytst),1)
    loss = (ytst-ytst_pred)* (ytst-ytst_pred)
    print(np.mean(loss))
    # print("ρ>{}, max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(lamb, max(loss), min(loss), np.mean(loss), np.var(loss)))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(Xtst,ytst, marker='o',color='r',label ="Test Data")
    plt.plot(Xtst,ytst_pred,label="EDERM")
    plt.legend()
    plt.show()
    