import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

def calculate(x,lamb,delta):
    if 1-lamb + x > 0:
        Ic = (1-math.exp((1-lamb+x)**2/(-2*delta*delta)))/(1-math.exp(1/(-2*delta*delta)))
    else:
        Ic = 0
    return Ic

def draw_closs(delta):
    Ic = [0]*100
    index = 0
    for i in np.linspace(-5,5,100):
        # x = i
        # Ic[index] = (1-math.exp((1-x)**2/(-2*delta*delta)))/(1-math.exp(1/(-2*delta*delta)))
        # index += 1
        x = i 
        Ic[index] = calculate(0,x,delta)
        index += 1
        
    x1          = [-5,0,0,5]
    y1          = [1 ,1,0,0]
    x2          = np.linspace(-5,5,100)
    y2          = Ic   # final ρ
    plt.figure(figsize=(8, 8))
    plt.plot(x1,y1,linewidth=2,label='Original',color='b',marker='o', markerfacecolor='blue',markersize=4)
    plt.plot(x2,y2,linewidth=2,label='Approximate',color='r',marker='o', markerfacecolor='blue',markersize=4)
    plt.grid(True) 
    plt.xlabel('α') 
    plt.ylabel('Error') 
    plt.show()
    
def draw_single(delta,lamb):
    Ic = [0]*100
    index = 0
    for i in np.linspace(-5,5,100):
        x = i
        Ic[index] = calculate(x,lamb,delta)
        index += 1
    x1          = [-5,lamb,lamb,lamb+5]
    y1          = [0,0,1,1]
    x2          = np.linspace(-5,5,100)
    y2          = Ic   # final ρ
    plt.figure(figsize=(8, 8))
    plt.plot(x1,y1,linewidth=2,label='Original',color='b',marker='o', markerfacecolor='blue',markersize=4)
    plt.plot(x2,y2,linewidth=3,label='Approximate',color='r',marker='o', markerfacecolor='blue',markersize=4)
    plt.grid(True) 
    plt.xlabel('Density') 
    plt.ylabel('Ic Value') 
    plt.show()
    
def draw_double(delta,lamb,lamb2):
    Ic  = [0]*100
    Ic1 = [0]*100
    Ic2 = [0]*100
    index = 0
    for i in np.linspace(-5,5,100):
        x = i
        Ic1[index] = calculate(x,lamb,delta)
        Ic2[index] = calculate(x,lamb2,delta)
        Ic[index]  = Ic1[index]-Ic2[index]
        index += 1
    
    x1          = [-5,lamb,lamb,lamb2,lamb2,lamb2+5]
    y1          = [0,0,1,1,0,0]
    x2          = np.linspace(-5,5,100)
    y2          = Ic   # final ρ
    plt.figure(figsize=(8, 8))
    plt.plot(x1,y1,linewidth=2,label='Original',color='b',marker='o', markerfacecolor='blue',markersize=4)
    plt.plot(x2,y2,linewidth=3,label='Approximate',color='r',marker='o', markerfacecolor='blue',markersize=4)
    plt.grid(True) 
    plt.xlabel('Density') 
    plt.ylabel('Ic Value') 
    plt.show()
    
def Closs(delta = 1, lamb  = 1, lamb2 = 2):
    draw_closs(delta)
    draw_single(delta,lamb)
    draw_single(delta,lamb2)
    draw_double(delta,lamb,lamb2)


# Closs(delta = 1, lamb  = 1, lamb2 = 2)