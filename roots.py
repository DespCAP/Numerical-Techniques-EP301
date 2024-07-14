"""
Created on Thu Jul 14 2024, 05:56:15

@author: Amoy Ashesh
"""
import math 
import numpy as np

def bisection(r1,r2):
    """
    This function computes the root of a given function using the bisection method.
    
    Parameters:
    -----------
    r1 : float
        The first guess of the root.
    r2 : float
        The second guess of the root.
    
    Returns:
    --------
    x : float
        The root of the function.
    """
    x = (r1+r2)/2
    return x

def bracketing(f,a,b,error=1e-6,print_intermediate=False):
    """
    This function computes the bracketing points of a given function.
    
    Parameters:
    -----------
    f : function
        The function whose bracketing points are to be computed.
    a : float
        The first guess of the bracketing point.
    b : float
        The second guess of the bracketing point.
    error : float
        The error tolerance.
    print_intermediate : bool
        If True, prints the intermediate bracketing points.
    
    Returns:
    --------
    x1 : float
        The bracketing point of the function.
    """
    # Initial root
    x1 = bisection(a,b)
    xi = a
    while (abs(f(x1))>error):
        val = f(x1)*f(a)
        if val<0:
            xi = a
            x1 = bisection(xi,x1)
        else:
            xi = b 
            x1 = bisection(xi,x1)
        if print_intermediate:
            print("Bracketing points:",(a,b),"Root:",x1)
    return x1

# False Position
def fp(f,r0,r1):
    """
    This function computes the root of a given function using the false position method.
    """
    x = r1 - (f(r1)*(r1-r0)/(f(r1)-f(r0)))
    return x

def bracketing_fp(f,a,b,error=1e-6):
    """
    This function computes the bracketing points of a given function using the false position method.
    
    Parameters:
    -----------
    f : function
        The function whose bracketing points are to be computed.
    a : float
        The first guess of the bracketing point.
    b : float
        The second guess of the bracketing point.
    error : float
        (Optional)
        The error tolerance. Default is 1e-6.
    
    Returns:
    --------
    x1 : float
        The root of the function.
    """
    # Initial root
    x1 = fp(f,a,b)
    xi = a
    while (abs(f(x1))>error):
        val = f(x1)*f(xi)
        if val<0:
            xi = a
            x1 = fp(f,xi,x1)
        else:
            xi = b 
            x1 = fp(f,xi,x1)
    return x1

def secant(f,x0,x1,error=1e-6):
    """
    This function computes the root of a given function using the secant method.
    
    Parameters:
    -----------
    f : function
        The function whose root is to be computed.
    x0 : float
        The first guess of the root.
    x1 : float
        The second guess of the root.
    error : float
        (Optional)
        The error tolerance. Default is 1e-6.
    
    Returns:
    --------
    x2 : float
        The root of the function.
    """
    xi = x1
    x2 = fp(f,x0,x1)
    while (abs(f(x2))>error):
        temp = x2.copy()
        x2 = fp(f,xi,x2)
        xi = temp.copy()
    return x2

def nwrp(f,f_prime,x0,error=1e-6):
    """
    This function computes the root of a given function using the Newton-Raphson method.
    
    Parameters:
    -----------
    f : function
        The function whose root is to be computed.
    f_prime : function
        The derivative of the function.
    x0 : float
        The initial guess of the root.
    error : float
        (Optional)
        The error tolerance. Default is 1e-6.
    
    Returns:
    --------
    x1 : float
        The root of the function.
    """
    x1 = x0 - (f(x0)/f_prime(x0))
    while(abs(f(x1))>error):
        x1 = x1 - (f(x1)/f_prime(x1))
    return x1

def scan_br(fn,h,arr):
    """
    This function computes the bracketing points of a given function for any given number of roots .
    
    Parameters:
    -----------
    fn : function
        The function whose bracketing points are to be computed.
    h : float
        The step size.
    arr : list
        The range of the function.
    
    Returns:
    --------
    br : list
        The bracketing points of the function.
    """
    br = []
    a = arr[0]
    b = arr[1]
    for x in np.arange(a,b,h):
        val = fn(x)*fn(x-h)
        if val<0:
            br.append([x+2*h,x-2*h])
    return(br)

def poly(coeff,x):
    """
    Helper function to compute the value of a polynomial at a given point.
    """
    coeff = coeff[::-1]
    n = len(coeff)
    y = 0 
    for i in range(n):
        y += coeff[i]*(math.pow(x,i))
    return y

def derv(coeff,x):
    """
    Helper function to compute the derivative of a polynomial at a given point.
    """
    coeff = coeff[::-1]
    n = len(coeff)
    y = 0
    for i in range(1,n):
        y += coeff[i]*math.factorial(i)*math.pow(x,i-1)
    return y

def deflate(a):
    """
    This function computes the roots of a polynomial using the Newton-Raphson method. 
    
    Parameters:
    -----------
    a : list
        The coefficients of the polynomial.\n
        Example: For the equation: x^2 - 4x + 2 = 0;\n 
        a = [1,-4,2]
        
    Returns:
    --------
    roots : list
        The roots of the polynomial.
    """
    roots = []
    b = a.copy()
    # Deflation
    x = 0
    n = len(a)
    for i in range(n):
        r = nwrp(poly,derv,x,b)
        roots.append(r)
        x = roots[-1]+1
        nrr = len(b)
        temp = b.copy()
        b = []
        b.append(temp[0])
        for j in range(nrr-2):
            b.append(temp[j+1]+(r*b[-1]))
        # b.append(-a[-1]/r)
        if len(b)==1:
            break
    return roots

def dx(x,y,fn,h=0.0000001):
    """
    Helper function to compute the x-derivative of a function at a given point.
    """
    arr = [x-2*h,x-h,x,x+h,x+2*h]
    z = [(fn(t,y)) for t in arr]
    y_prime = (8*z[3]-8*z[1]-z[4]+z[0])/(12*h)
    return(y_prime)

def dy(x,y,fn,h=0.0000001):
    """
    Helper function to compute the y-derivative of a function at a given point.
    """
    arr = [y-2*h,y-h,y,y+h,y+2*h]
    z = [(fn(x,t)) for t in arr]
    y_prime = (8*z[3]-8*z[1]-z[4]+z[0])/(12*h)
    return(y_prime)

def j_matrix(f,g,x,y):
    """
    Helper function to compute the Jacobian matrix of a given system of equations.
    """
    m = [[dx(x,y,f),dy(x,y,f)],[dx(x,y,g),dy(x,y,g)]]
    return(m)

def solve(x,y,f,g):
    """
    Helper function to solve a system of Non-Linear simultaneous equations.
    """
    j = j_matrix(f,g,x,y)
    j_inv = np.linalg.inv(j)
    [xf,yf] = [x,y] - np.dot(j_inv,[f(x,y),g(x,y)])
    return(xf,yf)

def nl_solver(f,g,error=1e-8):
    """
    This function computes the roots of a system of Non-Linear simultaneous equations.
    
    Parameters:
    -----------
    f : function
        The first equation of the system.
    g : function
        The second equation of the system.
    error : float
        (Optional)
        The error tolerance. Default is 1e-8.
    
    Returns:
    --------
    x : float
        The root of the first equation.
    y : float
        The root of the second equation.
    """
    while((abs(f(x,y))+abs(g(x,y)))>error):
        x,y = solve(x,y,f,g)
    return(x,y)

def muller(x,fn,error = 0.00000001):
    """
    This function computes the root of a given function using the Muller's method.
    
    Parameters:
    -----------
    x : list
        The initial guesses of the root.
        Close points to the root to be determined.
    fn : function
        The function whose root is to be computed.
    error : float
        (Optional)
        The error tolerance. Default is 0.00000001.
    
    Returns:
    --------
    x[-1] : float
        The root of the function.
    """
    y = [fn(t) for t in x]
    # Iterative solver
    while (abs(fn(x[-1]))>error):
        n = 3
        # Initiate diveded difference table 
        f = []
        f.append(y[-3:])
        x = x[-3:]
        # Fill the array
        for i in range(0,n-1):
            rw = []
            for j in range(0,n-i-1):
                yn = (f[i][j+1] - f[i][j])/(x[j+1+i]-x[j])
                rw.append(yn)
            f.append(rw)
        # Define alpha and beta for muller's method
        b = f[-1][0]
        a = b*(x[-1]-x[-2]) + f[-2][0]
        # Update root 
        xn = x[-1] + (-a+((a**2 - (4*b*y[-1]))**0.5))/(2*b)
        yn = fn(xn)
        x.append(xn)
        y.append(yn)
    return(x[-1])