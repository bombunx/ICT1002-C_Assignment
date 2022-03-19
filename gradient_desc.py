from mimetypes import init
from re import X
from tkinter import Y
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from matplotlib import rcParams


def func(x):
    dim = len(x)
    #1st test case ### Matya's function 
    #y = 0.26 * (x[0] * x[0]+ x[1] * x[1]) - 0.48 *x[0] * x[1]
    
    #2nd test case ### Himmelblau's function 
    y = (x[0] * x[0] + x[1] - 11) * (x[0] * x[0] + x[1] - 11)  + (x[0] + x[1] * x[1] - 7) * (x[0] + x[1] * x[1] - 7)
    return y

def fprime(x):
    #1st test case ### Matya's function 
    #return np.array([0.52 * x[0] - 0.48 * x[1], 0.52 * x[1] - 0.48 * x[0]])
    
    #2nd test case ### Himmelblau's function 
    return np.array([2 * (x[0] * x[0] + x[1] -11) * 2 * x[0] + 2 * (x[0] + x[1] * x[1] -7), 2 * (x[0] * x[0] + x[1] -11) + 2 * (x[0] + x[1] * x[1] -7) * 2 * x[1]])

x0 = np.array([3,2])
    
# def plotFunc(x0):
#     x = np.arange(-10, 10, 0.025)
#     y = np.arange(-10, 10, 0.025)
#     X, Y = np.meshgrid(x, y)
#     Z = np.zeros(X.shape)
#     mesh_size = range(len(X))
#     for i, j in product(mesh_size, mesh_size):
#         x_coor = X[i][j]
#         y_coor = Y[i][j]
#         Z[i][j] = func(np.array([x_coor, y_coor]))

#     fig = plt.figure(figsize=(6,6))
#     ax = fig.gca(projection='3d')
#     ax.set_title('Matya function')
#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')
#     ax.set_zlabel('$f(x_1, x_2)$')
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     plt.tight_layout()

# def plotPath(xs, ys, x0):
#     plotFunc(x0)
#     plt.plot(xs, ys, linestyle='--', marker='o', color='orange')
#     plt.plot(xs[-1], ys[-1], 'ro')

# plotFunc(x0)
# plt.show()


def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.
    
    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    pk : array
        Search direction.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.
    
    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0*pk)
    
    while not phi_a0 <= phi0 + c1*alpha0*derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0*pk)
    
    return alpha0, phi_a0


def GradientDescentSimple(func, fprime, x0, alpha, tol=1e-5, max_iter=1000):
    # initialize x, f(x), and -f'(x)
    xk = x0 #starting x values
    fk = func(xk) # starting y values
    gfk = fprime(xk) # gradient (downwards slope)
    gfk_norm = np.linalg.norm(gfk) # absolute value of gradient
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk] # curve_x and curve_y to store all the x and y axis
    curve_y = [fk]
    # take steps
    while gfk_norm > tol and num_iter < max_iter: # while abs(gradient) > tolerance/threshold (self determined) and within counter of 1000
        # calculate new x, f(x), and -f'(x)
        pk = -gfk # -ve gradient
        
        #Without momentum term 1a
        fk = func(xk) # new y value
        
        #With momentum term 1b
        # alpha, fk = ArmijoLineSearch(func, xk, pk, gfk, fk, alpha0=alpha) # alpha determines the momentum of point x going downwards -> determines how big ur next step is
        
        xk = xk + alpha * pk # new x value
        gfk = fprime(xk) # new gradient
        gfk_norm = np.linalg.norm(gfk) # new abs(gradient)
        
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.6f}'.format(num_iter, fk, xk, gfk_norm))
    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(fk, xk))
    
    return np.array(curve_x), np.array(curve_y) # to be stored in txt file when doing in C

# xs, ys = GradientDescentSimple(func, fprime, x0, alpha=0.1)
#plotPath(xs, ys, x0)

def plot(xs, ys):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(-10, 10, 0.025)
    y = np.arange(-10, 10, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = func(np.array([x_coor, y_coor]))
    plt.suptitle('Gradient Descent Method')

    ax1.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
    ax1.plot(xs[-1,0], xs[-1,1], 'ro')
    ax1.set(
        title='Path During Optimization Process',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax1.contour(X,Y,Z)
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    ax2.plot(ys, linestyle='--', marker='o', color='orange')
    ax2.plot(len(ys)-1, ys[-1], 'ro')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(
        title='Objective Function Value During Optimization Process',
        xlabel='Iterations',
        ylabel='Objective Function Value'
    )
    plt.tight_layout()
    plt.show()

xs, ys = GradientDescentSimple(func, fprime, x0, 7)
plot(xs, ys)