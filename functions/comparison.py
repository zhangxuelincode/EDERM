import math
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
warnings.filterwarnings("ignore")


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

def gaussian_gradientdescent(X, y, Xtst, ytst,outlier = True , online = False,delta = 0.5):
    iters = 2000
    lr    = 0.005
    samplestrn = X.shape[0]
    samplestst = Xtst.shape[0]
    intercept  = np.ones((X.shape[0], 1))
    X_=X
    Xtst_=Xtst
    X         = np.concatenate((X, intercept), axis=1)
    Xtst     = Xtst.reshape((samplestst, Xtst.shape[1]))
    intercept  = np.ones((Xtst.shape[0], 1))
    Xtst         = np.concatenate((Xtst, intercept), axis=1)
    
    X     = X.reshape((samplestrn, X.shape[1]))
    y     = y.reshape((samplestrn, 1))
    Xtst  = Xtst.reshape((samplestrn, Xtst.shape[1]))
    ytst  = ytst.reshape((samplestrn, 1))
        

    gram       = kernel_function('gaussian',delta,X,X)
    #init params
    n          = X.shape[0]
    alpha_hat  = np.zeros([X.shape[0],1])
    
    y = y.reshape(len(y),1)

    #init params
    n      = len(y)
    alpha_hat = np.zeros([len(y),1])

    for k in range(iters):
        y_pred =  np.dot(gram, alpha_hat).reshape(len(y),1)
        ytst_pred = np.dot(kernel_function('gaussian', 1, X, Xtst), alpha_hat).reshape(len(ytst),1)
        if(online == True):
            print("iteration: ",k,"   Error: ",mse(y_pred,y),"test_R2",r2(ytst,ytst_pred))
        grad = np.dot(gram.T , (y_pred-y))
        grad = grad.reshape([n,1])
        alpha_hat = alpha_hat - lr * grad

    print("training set")
    y_pred =  np.dot(gram, alpha_hat)
    loss = np.array(y-y_pred) * np.array(y-y_pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
    if(X_.shape[1]==1):
        plt.figure(figsize=(8, 8))
        plt.scatter(X_,y)
        # plt.title("Training Set",fontsize= 'large') 
        plt.scatter(X_, y, marker='o',color='r', label="Training Data")
        plt.plot(X_, y_pred, 'b', label='ERM')  
        plt.legend()
        plt.show()
    
    print("test set")
    gram_tst = kernel_function('gaussian', 1, X, Xtst)
    ytst_pred =  np.dot(gram_tst, alpha_hat)
    loss = np.array(ytst - ytst_pred) * np.array(ytst - ytst_pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss),np.var(loss)))
    if(X_.shape[1]==1):
        plt.figure(figsize=(8, 8))
        plt.scatter(Xtst_, ytst)
        # plt.title("Test Set",fontsize= 'large') 
        plt.scatter(Xtst_, ytst, marker='o',color='r', label="Test Data")
        plt.plot(Xtst_, ytst_pred, 'b', label='ERM')  
        plt.legend()
        plt.show()
    return y_pred,ytst_pred


def Cal_grad(y_pred, y, X, j):
    total = 0.0
    for i in range(0, len(y)):
        total += ((y_pred[i] - y[i]) * X[i][j])
    return total / X.shape[0]

def BatchGradientDescent(Y, X_mat,step_size, tol ,iters,online,Xtst, ytst):
    my_copy = np.zeros(len(Y))
    error = mse(Y, my_copy)
    theta = np.zeros([X_mat.shape[1], 1])
    X = np.array(X_mat)
    for k in range(iters):
        new_theta = []
        for i in range(0, len(theta)):
            grad = Cal_grad(my_copy, Y, X, i)
            flag = theta[i] - step_size * grad
            new_theta.append(flag)
        theta = np.array(new_theta).reshape(len(theta), 1)
        my_copy = np.dot(X_mat, theta).reshape(len(Y), 1)
        new_error = mse(Y, my_copy)
        if abs(new_error - error) <= tol:
            break
        error = new_error
        if(online == True):
            print("iteration: ",k,"   Error: ",mse(my_copy,Y),'r2_test',r2(ytst,np.dot(Xtst, theta).reshape(len(Y), 1)))
    return my_copy, theta

def poly_ERM(X, y, Xtst, ytst,outlier = True, online = True,dimensions = 4,step_size = 0.05,tol=10**-8,iters =2000):
    samples = 100
    X     = X.reshape((samples,  ))
    Xtst  = Xtst.reshape((samples,  ))
    X_standard    = np.array(X) / 2.5 
    Xtst_standard = np.array(Xtst) / 2.5  
    X_ = []
    Xtst_ = []
    for i in range(dimensions + 1):
        X_.append(X_standard ** i)
        Xtst_.append(Xtst_standard ** i)
    X_ = np.mat(X_).T
    Xtst_ = np.mat(Xtst_).T

    y_pred, theta = BatchGradientDescent(y, X_,step_size, tol ,iters,online,Xtst_, ytst)

    ytst_pred = np.dot(Xtst_, theta)


    print("Training set")
    loss = np.array(y-y_pred) * np.array(y-y_pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
    
    print("Test set")
    loss = np.array(ytst-ytst_pred) * np.array(ytst-ytst_pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
    # loss_train = mse(y, y_pred)  
    # loss_test  = mse(ytst, ytst_pred)  
    # print("Trn Error: " + str(loss_train))
    # print("Tst Error: " + str(loss_test))
    plt.figure(figsize=(8, 8))
    # plt.title("Training Set",fontsize= 'large') 
    # plt.xlabel('x axis')  # make axis labels
    # plt.ylabel('y axis')
    plt.scatter(X, y, marker='o',color='r',label="Training Data")
    plt.plot(X, y_pred, 'b', label='ERM')  
    plt.legend(loc=4)  # make legend
    plt.show()
    
    plt.figure(figsize=(8, 8))
    # plt.title("Test Set",fontsize= 'large') 
    # plt.xlabel('x axis')  # make axis labels
    # plt.ylabel('y axis')
    plt.scatter(Xtst, ytst, marker='o',color='r',label="Test Data")
    plt.plot(Xtst, ytst_pred, 'b', label='ERM')  
    plt.legend(loc=4)  # make legend
    plt.show()
    return y_pred,ytst_pred


class KernelRegression:
    def __init__(self,delta):
        self.delta  = delta;
    def fit(self,X,Y):
        self.X      = X;
        self.Y      = Y;
        return self;
    def predict(self,data):
        size        = self.X.shape[0]; 
        u           = cdist(self.X , data)**2; 
        kernel_dist = self.rbf_kernel(u);
        sum         = np.sum(kernel_dist,axis = 1).reshape((size,1));
        weight      = kernel_dist/sum;
        pred        = np.dot(weight.T,self.Y).reshape(data.shape[0],1)
        return pred
    def rbf_kernel(self,u):
        return np.exp(-u/(self.delta**2))
    
def gaussian_closedform(outlier = True, delta = 0.5):  
    samplestrn    = 100 
    samplestst    = 50
    X    = np.linspace(-5, 5,samplestrn)
    y    = X    + 2 *  X**2    +  np.random.normal(0,1, samplestrn)
    l = [5,10,15,20,30,40,50,70]
    if(outlier == True):
        y[l] += 20
    Xtst = np.linspace(-2.5, 2.5,samplestst)
    ytst = Xtst + 2 *  Xtst**2 +  np.random.normal(0,1, samplestst)
    
    
    y    = y.reshape([y.shape[0],1])
    ytst = ytst.reshape([ytst.shape[0],1])
    X     = X.reshape((X.shape[0],1))
    Xtst  = Xtst.reshape((Xtst.shape[0],1))
    
    KR=KernelRegression(delta)
    KR.fit(X,y)

    pred= KR.predict(X)
    print("Training set")
    loss = np.array(y-pred) * np.array(y-pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
    plt.plot
    plt.scatter(X,y)
    # plt.title("Training Set",fontsize= 'large') 
    plt.plot(X,y,label ="True")
    plt.plot(X,pred,label="Prediction")
    plt.legend()
    plt.show()
    
    pred=KR.predict(Xtst)
    print("Test set")
    loss = np.array(ytst-pred) * np.array(ytst-pred)
    print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
    plt.plot
    plt.scatter(Xtst,ytst)
    # plt.title("Test Set",fontsize= 'large')
    plt.plot(Xtst,ytst,label ="True")
    plt.plot(Xtst,pred,label="Prediction")
    plt.legend()
    plt.show()
    
def kernel_closedform(x, xtst, y_noise, h, h0):
    weight_e = lambda t:  (1-t**2)*3/4 
    weight_t = lambda t:  (1-t**3)**3 
    
    num = len(x)
    y_rec_e = np.zeros(num)
    y_rec_t = np.zeros(num)
    y_rec_g = np.zeros(num)
    for i in range(num):
        dist = np.abs(x-xtst[i])/h  
        # epanechnikov
        w_e = weight_e(dist)*np.where(dist<=1,1,0)              
        y_rec_e[i] = np.sum(y_noise*w_e)/np.sum(w_e)
        # tri-cube
        w_t = weight_t(dist)*np.where(dist<=1,1,0) 
        y_rec_t[i] = np.sum(y_noise*w_t)/np.sum(w_t)
    
    # gaussian kernel    
    gaussian_kernel = lambda d,h: np.exp(-dist**2/(2*(h**2))) #/(np.sqrt(2*np.pi)*h)
    for i in range(num):
        dist = np.abs(x-xtst[i])
        w = gaussian_kernel(dist,h0)
        y_rec_g[i] = np.sum(y_noise*w)/np.sum(w)        
    return y_rec_g, y_rec_e, y_rec_t

def kernelregression_closedform(kernel = 'gaussian', outlier = True, h = 0.5, delta = 0.3): 
    trnsample = 100
    tstsample = 100
    h  = h
    h0 = delta
    x  = np.linspace(-5,5,trnsample)
    y       = x    + 2 *  x**2    
    y_noise = y + np.random.normal(0,1, y.shape[0])
    l = [5,10,15,20,30,40,50,70]
    if(outlier == True):
        y_noise[l] += 20    
    xtst    = np.linspace(-3,3,tstsample)
    ytst    = x    + 2 *  x**2    
    
    if(kernel == 'gaussian'):
        y_rec_g, y_rec_e, y_rec_t = kernel_closedform(x,x,y_noise,h,h0)
        print("Training set")
        loss = np.array(y_noise-y_rec_g) * np.array(y_noise-y_rec_g)
        print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
        plt.figure
        plt.plot(x,y)
        plt.plot(x,y_noise,'yo',markerfacecolor='none')
        plt.plot(x,y_rec_g,'m')
        plt.legend(labels=['original data','noise data','gaussian'],loc='upper right')
        plt.show()
        
        y_rec_g, y_rec_e, y_rec_t= kernel_closedform(x,xtst,y_noise,h,h0)
        print("Test set")
        loss = np.array(ytst-y_rec_g) * np.array(ytst-y_rec_g)
        print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
        plt.figure
        plt.plot(xtst,ytst)
        plt.plot(xtst,ytst,'yo',markerfacecolor='none')
        plt.plot(xtst,y_rec_g,'m')
        plt.legend(labels=['original data','noise data','gaussian'],loc='upper right')
        plt.show()
    elif(kernel == 'epanechnikov'):
        y_rec_g, y_rec_e, y_rec_t = kernel_closedform(x,x,y_noise,h,h0)
        print("Training set")
        loss = np.array(y_noise-y_rec_e) * np.array(y_noise-y_rec_e)
        print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
        plt.figure
        plt.plot(x,y)
        plt.plot(x,y_noise,'yo',markerfacecolor='none')
        plt.plot(x,y_rec_e,'k')
        plt.legend(labels=['original data','noise data','epanechnikov'],loc='upper right')
        plt.show()
        
        y_rec_g, y_rec_e, y_rec_t= kernel_closedform(x,xtst,y_noise,h,h0)
        print("Test set")
        loss = np.array(ytst-y_rec_e) * np.array(ytst-y_rec_e)
        print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
        plt.figure
        plt.plot(xtst,ytst)
        plt.plot(xtst,ytst,'yo',markerfacecolor='none')
        plt.plot(xtst,y_rec_e,'k')
        plt.legend(labels=['original data','noise data','epanechnikov'],loc='upper right')
        plt.show()
    elif(kernel == 'tri-cube'):
        y_rec_g, y_rec_e, y_rec_t = kernel_closedform(x,x,y_noise,h,h0)
        print("Training set")
        loss = np.array(y_noise-y_rec_t) * np.array(y_noise-y_rec_t)
        print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
        plt.figure
        plt.plot(x,y)
        plt.plot(x,y_noise,'yo',markerfacecolor='none')
        plt.plot(x,y_rec_t,'r')
        plt.legend(labels=['original data','noise data','tri-cube'],loc='upper right')
        plt.show()
        
        y_rec_g, y_rec_e, y_rec_t= kernel_closedform(x,xtst,y_noise,h,h0)
        print("Test set")
        loss = np.array(ytst-y_rec_t) * np.array(ytst-y_rec_t)
        print("max loss: {}, min loss: {}, avg loss: {}, variance: {}".format(max(loss), min(loss), np.mean(loss), np.var(loss)))
        plt.figure
        plt.plot(xtst,ytst)
        plt.plot(xtst,ytst,'yo',markerfacecolor='none')
        plt.plot(xtst,y_rec_t,'r')
        plt.legend(labels=['original data','noise data','tri-cube'],loc='upper right')
        plt.show()
        
# gaussian_closedform(outlier = True)
# gaussian_gradientdescent(outlier = True, online = False)
# poly_ERM(outlier = True, online = True)
# kernelregression_closedform(outlier = True)