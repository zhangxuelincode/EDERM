import math
import warnings
warnings.filterwarnings("ignore")

def indicator(name, fai, dfai, lamb,delta):
    if(name == 'correntropy'):
        Ic, dIc = Ic_correntropy(fai, dfai, lamb, delta)
        return Ic, dIc
    elif(name == 'modifiedsquare'):
        Ic, dIc = Ic_modifiedsquare(fai, dfai, lamb, delta)
        return Ic, dIc
    elif(name == 'exponential'):
        Ic, dIc = Ic_exponential(fai, dfai, lamb, delta)
        return Ic, dIc
    elif(name == 'sigmoid'):
        Ic, dIc = Ic_sigmoid(fai, dfai, lamb, delta)
        return Ic, dIc
    elif(name == 'tanh'):
        Ic, dIc = Ic_tanh(fai, dfai, lamb, delta)
        return Ic, dIc
    elif(name == 'hingeIC'):
        Ic, dIc = Ic_hinge(fai, dfai, lamb, delta)
        return Ic, dIc
    elif(name == 'noIC'):
        Ic, dIc = Ic_none(fai, dfai, lamb, delta)
        return Ic, dIc
 
def Ic_none(fai, dfai, lamb, delta):
    Ic   = 1
    dIc  = 0
    return Ic, dIc

def Ic_correntropy(fai, dfai, lamb, delta):
    if(1-lamb +fai>0): 
        Ic   = (1-math.exp((1-lamb+fai)**2/(-2*delta*delta)))/(1-math.exp(1/(-2*delta*delta)))
        dIc  = math.exp((1-lamb+fai)**2/(-2*delta*delta))/(1-math.exp(1/(2*delta*delta)))\
            * ((1-lamb+fai)/(delta*delta)) * dfai
        return Ic, dIc
    else:
        Ic   = 0
        dIc  = 0
    return Ic, dIc

def Ic_modifiedsquare(fai, dfai, lamb, delta):
    if(fai>lamb): 
        Ic   = delta*(fai - lamb)**2
        dIc  = 2 * delta*(fai - lamb) * dfai
    else:
        Ic   = 0
        dIc  = 0
    return Ic, dIc

def Ic_exponential(fai, dfai, lamb, delta):
    Ic   = math.exp(delta * (fai - lamb))
    dIc  = math.exp(delta * (fai - lamb)) * delta * dfai
    return Ic, dIc

def Ic_sigmoid(fai, dfai, lamb, delta):
    Ic   = 1 / (1 + math.exp(-delta * (fai - lamb)))
    dIc  = delta * math.exp(-delta * (fai - lamb)) * dfai / (1 + math.exp(-delta * (fai - lamb)))**2
    return Ic, dIc

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def Ic_tanh(fai, dfai, lamb, delta):
    Ic   = 1 + tanh(delta * (fai - lamb))
    dIc  = delta * dfai *(1 - (tanh(delta * (fai - lamb)))**2 )
    return Ic, dIc

def Ic_hinge(fai, dfai, lamb, delta):
    # Ic   = max(0,delta*(fai-lamb))
    # dIc  = bool((fai-lamb)>0)*delta
    if(fai>lamb): 
        Ic   = delta*(fai - lamb)
        dIc  = delta*dfai
    else:
        Ic   = 0
        dIc  = 0
    return Ic, dIc


# def Ic_ramp(fai, dfai, lamb, delta):
#     if(fai >= lamb): 
#         Ic   = 1*delta
#         dIc  = 0
#         return Ic, dIc
#     elif(fai <= lamb - 1):
#         Ic   = 0
#         dIc  = 0
#     else:
#         Ic   = (1 - lamb + fai)*delta
#         dIc  = 1*delta
#     return Ic, dIc

