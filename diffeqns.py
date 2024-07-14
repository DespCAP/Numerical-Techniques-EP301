"""
Created on Thu Jul 09 2024, 14:51:15 

@author: Amoy Ashesh
"""

import numpy as np
import matplotlib.pyplot as plt

def sde(f,x_val,e_range,tolerance=0.6,error=0.09,h=0.01,plot=True):
    """
    Function to solve the Schrodinger equation using the 2 simultaneous differential equations solver method.\n
    d^2(ψ)/dx^2 = -2Eψ (ℏ/m  = 1)
    
    Parameters:
    -----------
    f : function
        The function f(x,ψ,E) which defines the differential equation.
    x_val : numpy array
        The array of x values.
    e_range : numpy array
        The array of energy values.
    tolerance : float
        (Optional)
        The tolerance for the solution. Default is 0.6.
    error : float
        (Optional)
        The error tolerance for the solution. Default is 0.09.
    h : float
        (Optional)
        The step size for the numerical integration. Default is 0.01.
    plot : bool
        (Optional)
        If True, it plots the solutions. Default is True.
    
    Returns:
    --------
    sol : list
        The list of solutions.
    """
    sol = []
    Energies = []
    # Solving 2 simultaneous ODEs
    for E in e_range:
        # Initial boundary condition
        psi = [0]
        # Assumption
        psi_prime = [1]
        
        for x in x_val:
            # Solving dψ/dx = Φ
            m1 = psi_prime[-1]
            m2 = psi_prime[-1]+(m1*h/2)  
            m3 = psi_prime[-1]+(m2*h/2)
            m4 = psi_prime[-1]+(m3*h)             
            # Solving dΦ/dx = f(r,ψ)
            M1 = f(x,psi[-1],E)
            M2 = f(x,psi[-1]+(m1*h/2),E)
            M3 = f(x,psi[-1]+(m2*h/2),E)
            M4 = f(x,psi[-1]+(m3*h),E)   
            
            y = psi[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
            y1 = psi_prime[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
            
            psi.append(y)
            psi_prime.append(y1)
        if abs(psi[-1])<error:
            if len(sol)>0:
                if (abs(Energies[-1]-E)>tolerance):
                    print("Energy =",E)
                    sol.append(psi)
                    Energies.append(E)
            else:
                print("Energy =",E)
                sol.append(psi)
                Energies.append(E)
    if plot:
        for s in sol:
            plt.plot(x_val,s[1:])
            plt.xlabel("X")
            plt.ylabel("ψ") 
            plt.title("Solutions of the Schrodinger equation")
            plt.legend(["E="+str(Energies[list(sol).index(s)])])
            plt.show()
            
    return sol

def shooting(f,x_val,b0,bl,v0,v1,tolerance=0.00001,plot=True):
    """
    Function to solve the boundary value problem using the shooting method.\n
    dv/dx = f(x,y) and dy/dx = v
    
    Parameters:
    -----------
    f : function
        The function f(x,y) which defines the differential equation.
    x_val : numpy array
        The array of x values.
    b0 : float
        The initial boundary condition.
    bl : float
        The final boundary condition.
    v0 : float
        The initial guess for the slope.
    v1 : float
        The final guess for the slope.
    tolerance : float
        (Optional)
        The tolerance for the solution. Default is 0.00001.
    plot : bool
        (Optional)
        If True, it plots the solutions. Default is True.
    
    Returns:
    --------
    p : list
        The list of solutions.
    sol : list
        The list of solutions.
    sol_v : list
        The list of velocities.
    """
    
    h = abs(x_val[1] - x_val[0])
    # Range of shooting points
    sol = []
    p = []
    sol_v = []
    # V0
    y = [b0]
    v = [v0]
    for x in x_val:
        # Solving dy/dx = v
        m1 = v[-1]
        m2 = v[-1]+(m1*h/2)  
        m3 = v[-1]+(m2*h/2)
        m4 = v[-1]+(m3*h)             
        # Solving dv/dx = f(x,y)
        M1 = f(x,y[-1])
        M2 = f(x,y[-1]+(m1*h/2))
        M3 = f(x,y[-1]+(m2*h/2))
        M4 = f(x,y[-1]+(m3*h))   
        
        t = y[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
        t1 = v[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
        
        y.append(t)
        v.append(t1)
    bt0 = y[-1]
    p.append(y)
    # V1
    y = [b0]
    v = [v1]
    for x in x_val:
        # Solving dy/dx = v
        m1 = v[-1]
        m2 = v[-1]+(m1*h/2)  
        m3 = v[-1]+(m2*h/2)
        m4 = v[-1]+(m3*h)             
        # Solving dv/dx = f(x,y)
        M1 = f(x,y[-1])
        M2 = f(x,y[-1]+(m1*h/2))
        M3 = f(x,y[-1]+(m2*h/2))
        M4 = f(x,y[-1]+(m3*h))   
        
        t = y[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
        t1 = v[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
        
        y.append(t)
        v.append(t1)
    bt1 = y[-1]
    p.append(y)
    # V2
    v2 = v1 + (bl-bt1)*(v1-v0)/(bt1-bt0)
    y = [b0]
    v = [v2]
    for x in x_val:
        # Solving dy/dx = v
        m1 = v[-1]
        m2 = v[-1]+(m1*h/2)  
        m3 = v[-1]+(m2*h/2)
        m4 = v[-1]+(m3*h)             
        # Solving dv/dx = f(x,y)
        M1 = f(x,y[-1])
        M2 = f(x,y[-1]+(m1*h/2))
        M3 = f(x,y[-1]+(m2*h/2))
        M4 = f(x,y[-1]+(m3*h))   
        
        t = y[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
        t1 = v[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
        
        y.append(t)
        v.append(t1)
    if abs(y[-1]-bl)<tolerance:
        sol.append(y)
        sol_v.append(v2)
    p.append(y)   

    # Loop for better guesses
    while(abs(y[-1]-bl)>tolerance):
        v0 = v2-tolerance
        v1 = v2+tolerance
        # V0
        y = [b0]
        v = [v0]
        for x in x_val:
            # Solving dy/dx = v
            m1 = v[-1]
            m2 = v[-1]+(m1*h/2)  
            m3 = v[-1]+(m2*h/2)
            m4 = v[-1]+(m3*h)             
            # Solving dv/dx = f(x,y)
            M1 = f(x,y[-1])
            M2 = f(x,y[-1]+(m1*h/2))
            M3 = f(x,y[-1]+(m2*h/2))
            M4 = f(x,y[-1]+(m3*h))   
            
            t = y[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
            t1 = v[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
            
            y.append(t)
            v.append(t1)
        bt0 = y[-1]
        p.append(y)
        # V1
        y = [b0]
        v = [v1]
        for x in x_val:
            # Solving dy/dx = v
            m1 = v[-1]
            m2 = v[-1]+(m1*h/2)  
            m3 = v[-1]+(m2*h/2)
            m4 = v[-1]+(m3*h)             
            # Solving dv/dx = f(x,y)
            M1 = f(x,y[-1])
            M2 = f(x,y[-1]+(m1*h/2))
            M3 = f(x,y[-1]+(m2*h/2))
            M4 = f(x,y[-1]+(m3*h))   
            
            t = y[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
            t1 = v[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
            
            y.append(t)
            v.append(t1)
        bt1 = y[-1]
        p.append(y)
        # V2
        v2 = v1 + (bl-bt1)*(v1-v0)/(bt1-bt0)
        y = [b0]
        v = [v2]
        for x in x_val:
            # Solving dy/dx = v
            m1 = v[-1]
            m2 = v[-1]+(m1*h/2)  
            m3 = v[-1]+(m2*h/2)
            m4 = v[-1]+(m3*h)             
            # Solving dv/dx = f(x,y)
            M1 = f(x,y[-1])
            M2 = f(x,y[-1]+(m1*h/2))
            M3 = f(x,y[-1]+(m2*h/2))
            M4 = f(x,y[-1]+(m3*h))   
            
            t = y[-1] + (h*(m1+(2*m2)+(2*m3)+m4)/6)             
            t1 = v[-1] + (h*(M1+(2*M2)+(2*M3)+M4)/6)
            
            y.append(t)
            v.append(t1)
        if abs(y[-1]-bl)<tolerance:
            sol.append(y)
            sol_v.append(v2)
        p.append(y)  
        
    # Display solution(s)
    if plot:
        i = 0
        l = []
        for s in sol:
            plt.plot(x_val,s[1:])
            l.append("v(0) = "+str(sol_v[i]))
            i+=1
        plt.legend(l)
        plt.title("Solution(s)")
        plt.show()

        # Display different solutions for different values of V(0)
        for s in p:
            plt.plot(x_val,s[1:])
        plt.title("Different solutions for different values of V(0)")
        plt.show()
    return p,sol,sol_v

def heat_diffusion(xn,tn,ui0,tau=0.1,h=0.1,a=0.5,n=50,plot=True):
    """
    Function to solve the heat diffusion equation using the finite difference method.\n
    ∂u/∂t = a*∂^2u/∂x^2
    
    Parameters:
    -----------
    xn : float
        The final x value.
    tn : float
        The final t value. 
    ui0 : list
        The initial condition.
    tau : float
        (Optional)
        The time step. Default is 0.1.
    h : float
        (Optional)
        The space step. Default is 0.1. 
    a : float
        (Optional)
        The diffusion constant. Default is 0.5.
    n : int
        (Optional)
        The number of iterations. Default is 50.
        
    Returns:
    --------
    u : numpy array
        The solution.
    """
    
    t = np.linspace(0,tn,n)
    x = np.linspace(0,xn,n)
    ui0 = [np.exp(-k*k) for k in x]
    if plot:
        plt.plot(x,ui0)
        plt.title("u(x,0)")
        plt.show()
    
    u = np.zeros((n,n))

    for i in range(n):
        u[i][0] = ui0[i]


    for j in range(n-1):
        for i in range(1,n-1):
                u[i][j+1] = ((1-(2*tau*a/(h*h)))*u[i][j]) + ((tau*a/(h*h))*u[i-1][j]) + ((tau*a/(h*h))*u[i+1][j])

    y = []
    for i in range(n):
        y.append(u[i][1])

    if plot:
        plt.plot(x,y)
        plt.title(f"u(x,{tau})")
        plt.show()
        
    return u