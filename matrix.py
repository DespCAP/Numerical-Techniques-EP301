"""
Created on Thu Jun 27 2024, 11:49:31 

@author: Amoy Ashesh
"""

import numpy as np

def gauss_elimination(N,A,B):
    """
    A function to solve a system of linear equations using the
    Gauss Elimination method.\n
    The equations are of the form:\n
    A.X = B
    
    Parameters
    ----------
    N : int
        The number of equations.
        
    A : 2D list
        The coefficient matrix.\n
        Example:
        [[a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]]
    
    B : list
        The constant matrix.\n
        Example:
        [b1, b2, b3]
    
    Returns
    -------
    d : list
        The solution to the system of linear equations.
        
    """
    
    # Number of equations
    n=N
    # Checking if the matrix is square
    for i in range(n):
        if len(A[i])!=n:
            raise ValueError("The matrix is not square")
    # Checking if the number of equations and constants match
    if len(B)!=n:
        raise ValueError("The number of equations and constants do not match")
            
    # Creating the augmented matrix
    m = np.zeros((n,n+1))
    for i in range(n):
        for j in range(n):
            m[i][j] = A[i][j]
        m[i][n] = B[i]
    
    print("Entered Matrix:")
    for x in m:
        print(x)
        
    # Gauss Elimination
    for k in range(0,n-1):
        for i in range(k+1,n):
            for j in range(n,k-1,-1):
                m[i][j] = m[i][j] - ((m[i][k]*m[k][j])/m[k][k])
    d = {}
    for k in range(n-1,-1,-1):
        t = 0
        diff = 0
        for j in range(k+1,n):
            diff += m[k][j]*d[j]
        t = (m[k][-1] - diff)/m[k][k]
        d[k] = t
    d = list(np.around(list(d.values())[::-1],8))
    print("\nReduced Matrix:")
    for x in m:
        print(list(np.around(list(x),8)))
    print("\nSolution:")
    for x in d:
        print(x)
    return d

def sw_rws(a,b):
    """
    A function to swap two rows of a matrix.
    """
    temp = a.copy()
    a = b.copy()
    b = temp.copy()
    return a,b
    
def mx_elem_rw(m,k,n):
    """
    A function to find the row with the maximum element in a given column.
    """
    mxx = abs(m[k][k])
    mxi = k
    for x in range(k,n):
        for y in range(0,n):
            if abs(m[x][y])>mxx:
                mxx = abs(m[x][y])
                mxi = x
    return mxi

def gauss_elimination_pivoting(N,A,B):
    """
    A function to solve a system of linear equations using the
    Gauss Elimination with Pivoting method.\n
    The equations are of the form:\n
    A.X = B
    
    Parameters
    ----------
    N : int
        The number of equations.
        
    A : 2D list
        The coefficient matrix.\n
        Example:
        [[a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]]
    
    B : list
        The constant matrix.\n
        Example:
        [b1, b2, b3]
    
    Returns
    -------
    d : list
        The solution to the system of linear equations.
    
    """
    
    # Number of equations
    n=N
    # Checking if the matrix is square
    for i in range(n):
        if len(A[i])!=n:
            raise ValueError("The matrix is not square")
    # Checking if the number of equations and constants match
    if len(B)!=n:
        raise ValueError("The number of equations and constants do not match")
            
    # Creating the augmented matrix
    m = np.zeros((n,n+1))
    for i in range(n):
        for j in range(n):
            m[i][j] = A[i][j]
        m[i][n] = B[i]
    print("Entered Matrix:")
    for x in A:
        print(x)
    
    # Pivoting
    for k in range(0,n-1):
        i = mx_elem_rw(m,k,n)
        m[k],m[i] = sw_rws(m[k],m[i])
    print("\nPivoted Matrix:")
    for x in m:
        print(x)
    # Elimination
    for k in range(0,n-1):
        for i in range(k+1,n):
            for j in range(n,k-1,-1):
                m[i][j] = m[i][j] - ((m[i][k]*m[k][j])/m[k][k])
    d = {}
    for k in range(n-1,-1,-1):
        t = 0
        diff = 0
        for j in range(k+1,n):
            diff += m[k][j]*d[j]
        t = (m[k][-1] - diff)/m[k][k]
        d[k] = t
    d = list(np.around(list(d.values())[::-1],8))
    print("\nReduced Matrix:")
    for x in m:
        print(list(np.around(list(x),8)))
    print("\nSolution:")
    print(d)
    return(d)
    
def gauss_jordan(N,A,B):
    """
    A function to solve a system of linear equations using the
    Gauss Jordan Elimination method.\n
    The equations are of the form:\n
    A.X = B
    
    Parameters
    ----------
    N : int
        The number of equations.
        
    A : 2D list
        The coefficient matrix.\n
        Example:
        [[a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]]
    
    B : list
        The constant matrix.\n
        Example:
        [b1, b2, b3]
    
    Returns
    -------
    sol : list
        The solution to the system of linear equations.
    
    """
    # Number of equations
    n=N
    # Checking if the matrix is square
    for i in range(n):
        if len(A[i])!=n:
            raise ValueError("The matrix is not square")
    # Checking if the number of equations and constants match
    if len(B)!=n:
        raise ValueError("The number of equations and constants do not match")
            
    # Creating the augmented matrix
    m = np.zeros((n,n+1))
    for i in range(n):
        for j in range(n):
            m[i][j] = A[i][j]
        m[i][n] = B[i]
    print("Enterred Matrix:")
    for x in m:
        print(x)
        
    # Create upper triangular matrix
    for k in range(0,n-1):
        for i in range(k+1,n):
            for j in range(n,k-1,-1):
                m[i][j] = m[i][j] - ((m[i][k]*m[k][j])/m[k][k])
    # Make diagonal matrix
    for k in range(n-1,-1,-1):
        m[k][-1] = m[k][-1]/m[k][k]
        m[k][k] =  1
        for i in range(k-1,-1,-1):
            for j in range(n,0,-1):
                m[i][j] = m[i][j] - (m[i][k]*m[k][j])
    # Round-off
    m1 = []
    for x in m:
        rw=[]
        for y in x:
            y = round(y,6)
            rw.append(y)
        m1.append(rw)
    m = m1
    print("\nReduced Matrix:")
    for x in m:
        print(x)
    sol = []
    for x in m:
        sol.append(x[-1])
    print("\nSolution = ",sol)
    return(sol)
    
def lu_decomp(N,A,B):
    """
    A function to solve a system of linear equations using the
    LU-decomposition method.\n
    The equations are of the form:\n
    A.X = B
    
    Parameters
    ----------
    N : int
        The number of equations.
        
    A : 2D list
        The coefficient matrix.\n
        Example:
        [[a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]]
    
    B : list
        The constant matrix.\n
        Example:
        [b1, b2, b3]
    
    Returns
    -------
    x : list
        The solution to the system of linear equations.
    
    """
    # Number of equations
    n=N
    # Checking if the matrix is square
    for i in range(n):
        if len(A[i])!=n:
            raise ValueError("The matrix is not square")
    # Checking if the number of equations and constants match
    if len(B)!=n:
        raise ValueError("The number of equations and constants do not match")
    
    a = A
    b = B
    # Initialize U
    u = []
    for i in range(0,n):
        rw = []
        for j in range(0,n):
            rw.append(0)
        u.append(rw)
    # Initialize L
    l = []
    for i in range(0,n):
        rw = []
        for j in range(0,n):
            rw.append(0)
        l.append(rw)
    for i in range(0,n):
        l[i][i] = 1    
    # First row elements
    for i in range(0,n):
        u[0][i] = a[0][i]
    # First column elements
    for i in range(1,n):
        l[i][0] = a[i][0]/a[0][0]
    # Rest of the matrix
    for i in range(0,n):
        for j in range(0,n):
            if ((i<=j) and i!=0):
                pr = 0
                for k in range(0,i):
                    pr += l[i][k]*u[k][j]
                u[i][j] = a[i][j] - pr
            elif ((i>j) and j!=0):
                prr = 0
                for k in range(0,j):
                    prr += l[i][k]*u[k][j]
                l[i][j] = (a[i][j] - prr)/u[j][j]
    print("L:")
    for x in l:
        print(x)
    print("\nU:")
    for x in u:
        print(x)
    # Calculate Z
    z = {}
    for k in range(0,n):
        diff = 0
        for j in range(0,k):
            diff += l[k][j]*z[j]
        z[k] = (b[k] - diff)/l[k][k]
    z = list(np.around(list(z.values()),8))
    print("\nZ:")
    print(z)
    # Calculate X
    x = {}
    for k in range(n-1,-1,-1):
        diff = 0
        for j in range(k+1,n):
            diff += u[k][j]*x[j]
        x[k] = (z[k] - diff)/u[k][k]
    x = list(np.around(list(x.values())[::-1],8))
    print("\nSolution:")
    print(x)
    return(x)

# Function to check lists
def check(x,xn,n):
    """
    A function to check if two lists are equal.
    
    """
    tr = True    
    for i in range(0,n):
        temp = (round(x[i],16)==round(xn[i],16))
        tr = (tr and temp)
    return not(tr)

def jacobi(N,A,B,print_iter=False,max_iter=2000):
    """
    A function to solve a system of linear equations using the
    Jacobi Iterative method.\n
    The equations are of the form:\n
    A.X = B
    
    WARNING: The method may not converge for all matrices.
    ----------
    
    Parameters
    ----------
    
    N : int
        The number of equations.
        
    A : 2D list
        The coefficient matrix.\n
        Example:
        [[a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]]
    
    B : list
        The constant matrix.\n
        Example:
        [b1, b2, b3]
        
    print_iter : bool
        (Optional)
        Whether to print the iterations or not.
    
    max_iter : int
        (Optional)
        The maximum number of iterations.
    
    Returns
    -------
    x : list
        The solution to the system of linear equations.
    
    """
    a = A
    b = B
    
    # Number of equations
    n=N
    # Checking if the matrix is square
    for i in range(n):
        if len(A[i])!=n:
            raise ValueError("The matrix is not square")
    # Checking if the number of equations and constants match
    if len(B)!=n:
        raise ValueError("The number of equations and constants do not match")
    
    x = []
    xn = [0,0,0]
    
    # Initial "guess" solution
    for i in range(0,n):
        x.append(b[i]/a[i][i])
    print("Start",x)
    # Iterating in loops
    iterations = 0
    while (check(x,xn,n)):
        xn = [0,0,0]
        for i in range(0,n): 
            diff = 0       
            for j in range(0,n):
                if j!=i:
                    diff += a[i][j]*x[j]
            xn[i] = (b[i]-diff)/a[i][i]
        t = check(x,xn,n)
        if not(t):
            break
        for i in range(0,n):
            x[i] = xn[i]
        if not(check(x,xn,n)):
            xn = [0,0,0]
        iterations += 1
        if print_iter:
            print(iterations,x)
        if iterations>max_iter:
            raise ValueError("Method did not converge")
    print("---Final---")
    print(x)
    return(x)