"""
Created on Thu Jun 27 2024, 16:30:54 

@author: Amoy Ashesh
"""

import numpy as np
import matplotlib.pyplot as plt

def derv_fwd_diff(x,y,x0):
    """
    This function computes the forward difference(crude numerical first derivative) of a given set of data points
    (x,y) at a point x0.
    
    Parameters:
    -----------
    x : numpy array
        The data points in the x-axis.
    y : numpy array
        The data points in the y-axis.
    x0 : float
        The point at which the forward difference is to be computed.
    
    Returns:
    --------
    y_prime : float
        The forward difference at the point x0.
    """
    if not (x0 in x):
        raise ValueError("x0 is not in the list of x values")
    i = x.index(x0)
    h = x[i+1]-x[i] 
    y_prime = (y[i+1]-y[i])/h
    return y_prime

def derv_central_diff(x,y,x0):
    """
    This function computes the central difference(crude numerical second derivative) of a given set of data points
    (x,y) at a point x0.
    
    Parameters:
    -----------
    x : numpy array
        The data points in the x-axis.
    y : numpy array
        The data points in the y-axis.
    x0 : float
        The point at which the central difference is to be computed.
    
    Returns:
    --------
    y_prime : float
        The central difference at the point x0.
    """
    if not (x0 in x):
        raise ValueError("x0 is not in the list of x values")
    i = x.index(x0)
    h = x[i+1]-x[i] 
    y_prime = (y[i+1]-y[i-1])/(2*h)
    return y_prime

def derv_richardson_extrapolation(x0:float,fx,h=0.00001):
    """
    This function computes the Richardson extrapolation of a given function f(x) at a point x0.
    (This is the most accurate numerical first derivative method in this module)
    
    Parameters:
    -----------
    x0 : float
        The point at which the Richardson extrapolation is to be computed.
    fx : function
        The function f(x) whose Richardson extrapolation is to be computed.
    h : float
        (Optional)
        The step size for the numerical differentiation. Default is 0.00001.
    
    Returns:
    --------
    fx_prime : float
        The Richardson extrapolation at the point x0.
    """
    
    x = [x0-2*h,x0-h,x0,x0+h,x0+2*h]
    y = [fx(y) for y in x]
    y_prime = (8*y[3]-8*y[1]-y[4]+y[0])/(12*h)
    return y_prime

def phase_space_plot(x0,v0,fxv,dt=0.1,n=1000,plot=True):
    """
    This function plots the phase space of a given set of data points (x,y). It also plots the X-T plot.
    It returns the lists of positions and velocities.
    
    Parameters:
    -----------
    x0 : float
        The initial position.
    v0 : float
        The initial velocity.
    fxv : function
        The function f(x,v) which defines the differential equation.
    dt : float
        (Optional)
        The step size for the numerical integration. Default is 0.1.
    n : int
        (Optional)
        The number of iterations for the numerical simulation. Default is 1000.
    plot : bool
        (Optional)
        If True, it plots the phase space and X-T plot. Default is True.
    
    Returns:
    --------
    The phase space plot and the X-T plot.

    x : list
        The list of positions.
    v : list
        The list of velocities.
        
    """
    
    t0 = 0
    t_arr = [t0]
    for i in range(n):
        t0 += dt
        t_arr.append(t0)
        
    x = []
    v = []
    x.append(x0)
    v.append(v0)
    for i in range(n):
        x.append(x[-1]+dt*v[-1])
        v.append(v[-1]+dt*(fxv(x[-1],v[-1])))

    if plot:
        plt.plot(x,v)
        plt.title("Phase-space diagram")
        plt.xlabel("X")
        plt.ylabel("V")
        plt.show()
        plt.plot(t_arr,x)
        plt.title("X-T Plot")
        plt.xlabel("Time")
        plt.ylabel("X")
        plt.show()
    
    return x,v

def trapezoidal(x,h,f):
    """
    Function to compute the integral of a given function using the trapezoidal rule.
    
    Parameters:
    -----------
    x : numpy array
        The array of x values.
    h : float
        The step size.
    f : numpy array
        The array of f(x) values.
    
    Returns:
    --------
    comp_trap : float
        The computed integral.
    """
    N = int((x[-1]-x[0])/h)
    sm = (f[0]+f[-1])/2
    for n in range(1,N):
        sm += f[n]
    comp_trap = h*sm
    return comp_trap

def simpson(x,h,f):
    """
    Function to compute the integral of a given function using the Simpson's rule.
    
    Parameters:
    -----------
    x : numpy array
        The array of x values.
    h : float
        The step size.
    f : numpy array
        The array of f(x) values.
    
    Returns:
    --------
    comp_smp : float
        The computed integral.
    """
    N = int((x[-1]-x[0])/h)
    comp_smp = 0
    for i in range(int(N/2)):
        t = h*(f[2*i]+(4*f[(2*i)+1])+f[(2*i)+2])/3
        comp_smp += t
    return comp_smp

def simpson_3_8th(x,h,f):
    """
    Function to compute the integral of a given function using the Simpson's 3/8th rule.
    
    Parameters:
    -----------
    x : numpy array
        The array of x values.
    h : float
        The step size.
    f : numpy array
        The array of f(x) values.
    
    Returns:
    --------
    comp_smp : float
        The computed integral.
    """
    N = int((x[-1]-x[0])/h)
    comp_smp = 0
    for i in range(int(N/3)):
        t = 3*h*(f[3*i]+(3*f[(3*i)+1])+(3*f[(3*i)+2])+f[(3*i)+3])/8
        comp_smp += t
    return comp_smp

def gaussian_4_point(a,b,f):
    """
    Function to compute the integral of a given function using the Gaussian 4-point rule.
    
    Parameters:
    -----------
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    f : function
        The function f(x) to be integrated.
    
    Returns:
    --------
    I : float
        The computed integral.
    """

    # Rescaling
    t = [-0.86114,-0.33998,0.33998,0.86114]
    x = [(((b-a)*t1/2)+((b+a)/2)) for t1 in t]
    # Gaussian Integration
    I = ((b-a)/2)*((0.34785*f(x[0]))+(0.65215*f(x[1]))+(0.65215*f(x[2]))+(0.34785*f(x[3])))
    
    return I

def gaussian_integration(a,b,f,weights,nodes):
    """
    Function to compute the integral of a given function using the Gaussian Integration rule.
        
    Parameters:
    -----------
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    f : function
        The function f(x) to be integrated.
    weights : numpy array
        (Default = [0.236927,0.478629,0.568889,0.478629,0.236927])
        The array of weights.
    nodes : numpy array
        (Default = [-0.90618,-0.538469,0,0.538469,0.90618])
        The array of nodes.
    
    Returns:
    --------
    I : float
        The computed integral.
    """
    # Rescaling
    x = [(((b-a)*t1/2)+((b+a)/2)) for t1 in nodes]
    # Gaussian Integration
    I = 0
    for i in range(len(nodes)):
        I += ((b-a)/2)*(weights[i]*f(x[i]))
    return(I)

def gauss_laguarre(f,weights=[0.521756,0.398667,0.0759424,0.00361176,0.00002337],nodes=[0.26356,1.4134,3.59643,7.808581,12.6408]):
    """
    Function to compute the integral of a given function using the Gaussian Laguarre rule.

    Parameters:
    -----------
    f : function
        The function f(x) to be integrated.
    weights : numpy array
        (Default = [0.521756,0.398667,0.0759424,0.00361176,0.00002337])
        The array of weights.
    nodes : numpy array
        (Default = [0.26356,1.4134,3.59643,7.808581,12.6408])
        The array of nodes.
    
    
    Returns:
    --------
    I : float
        The computed integral.
    """
    I = 0
    for i in range(len(nodes)):
        I += weights[i]*f(nodes[i])*np.exp(nodes[i])
    return(I)

def gauss_hermite(f,weights=[0.0199532,0.393619,0.945309,0.393619,0.0199532],nodes=[-2.02018,-0.958572,0,0.958572,2.02018]):
    """
    Function to compute the integral of a given function using the Gaussian Hermite rule.
    
    Parameters:
    -----------
    f : function
        The function f(x) to be integrated.
    weights : numpy array
        (Default = [0.0199532,0.393619,0.945309,0.393619,0.0199532])
        The array of weights.
    nodes : numpy array
        (Default = [-2.02018,-0.958572,0,0.958572,2.02018])
        The array of nodes.
    
    Returns:
    --------
    I : float
        The computed integral
    """
    I = 0
    for i in range(len(nodes)):
        I += weights[i]*f(nodes[i])*((np.exp(nodes[i]))**2)
    return(I)