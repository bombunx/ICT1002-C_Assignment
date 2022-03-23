from cmath import sqrt
from mimetypes import init
from re import X
from tkinter import Y
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from matplotlib import rcParams

#### ----- Define functions and variables ----- #### 

dim = 2
grad = [0,0]
hessian_vecshaped = [0,0,0,0]


def func(x):
    y = 0.26 * (x[0] * x[0]+ x[1] * x[1]) - 0.48 *x[0] * x[1]
    # p1= 1.5 - x[0] +x[0]*x[1];
    # p2= 2.25 - x[0] +x[0]*x[1]*x[1]; 
    # p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1];   
  
    # y = p1*p1 + p2*p2 + p3*p3;
    return y

def gradient(x):
    y = func(x)
    grad[0] = 0.52 * x[0] - 0.48 * x[1]
    grad[1] = 0.52 * x[1] - 0.48 * x[0]
    
    hessian_vecshaped[0] = 0.52
    hessian_vecshaped[3] = 0.52
    hessian_vecshaped[1] = -0.48
    hessian_vecshaped[2] = hessian_vecshaped[1]
    # p1= 1.5 - x[0] +x[0]*x[1];
    # p2= 2.25 - x[0] +x[0]*x[1]*x[1]; 
    # p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1]; 
  
    # grad[0] = 2*p1*(-1+x[1]) + 2*p2*(-1+x[1]*x[1])  + 2*p3*(-1+x[1]*x[1]*x[1]); 
    # grad[1] = 2*p1*x[0] +  2*p2*2*x[0]*x[1] + 2*p3*3*x[0]*x[1]*x[1]; 
    
    
    # q1 = -1+x[1];
    # q2 = -1+x[1]*x[1];
    # q3 = -1+x[1]*x[1] *x[1];  
    # hessian_vecshaped[0] = 2*q1*q1 + 2*q2*q2 + 2*q3*q3;  
    # hessian_vecshaped[3] = 2*x[0]*x[0] + 8*x[0]*x[0]*x[1]*x[1] + 2*p2*2*x[0] + 18*x[0]*x[0]*x[1]*x[1]*x[1]*x[1] + 2*p3*6*x[0]*x[1];
    # hessian_vecshaped[1] = 2*x[0]*q1 +2*p1 + 4*x[0]*x[1]*q2 + 2*p2*2*x[1]+ 6*x[0]*x[1]*x[1]*q3 + 2*p3*3*x[1]*x[1];
    # hessian_vecshaped[2] = hessian_vecshaped[1+2*0];       
    return y


#### ----- Plot graph ----- #### 



#### 3D Graph -------

def plotFunc(x0):
    x = np.arange(-4.5, 4.5, 0.025) # X domain
    y = np.arange(-4.5, 4.5, 0.025) # Y domain 
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = func(np.array([x_coor, y_coor]))

    fig = plt.figure(figsize=(6,6))
    ax = fig.gca(projection='3d')
    ax.set_title('Matya function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.tight_layout()


#### ----- Contour and Step Taken Graph

def plot(xs, ys):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(-4.5, 4.5, 0.025)
    y = np.arange(-4.5, 4.5, 0.025)
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

#### ----- Simple Gradient Descent ----- #####

def GradientDescentSimple(func, gradient, x, alpha, tol=1e-5, max_iter=1000):
    # initialize x, f(x), and -f'(x)
    
    current_x = x #starting x values
    current_y = gradient(current_x) # starting y values
    current_gradient_xy = np.array(grad) # gradient (downwards slope)
    norm_gradient = np.linalg.norm(current_gradient_xy) # absolute value of gradient
    
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [current_x] # curve_x and curve_y to store all the x and y axis
    curve_y = [current_y]
    # take steps
    while norm_gradient > tol and num_iter < max_iter: # while abs(gradient) > tolerance/threshold (self determined) and within counter of 1000
        # calculate new x, f(x), and -f'(x)
        negative_gradient = -current_gradient_xy # -ve gradient
        
        #Without momentum term 1a
        current_y = gradient(current_x) # new y value
        
        current_x = current_x + alpha * negative_gradient # new x value
        current_gradient_xy = np.array(grad) # new gradient
        norm_gradient = np.linalg.norm(current_gradient_xy) # new abs(gradient)
        #norm_grad_test = sqrt(current_gradient_xy[0] * current_gradient_xy[0] + current_gradient_xy[1] * current_gradient_xy[1])
        # increase number of steps by 1, save new x and f(x)
        
        
        
        num_iter += 1
        curve_x.append(current_x)
        curve_y.append(current_y)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {}'.format(num_iter, current_y, current_x, grad))
    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(current_y, current_x))
    
    return np.array(curve_x), np.array(curve_y) # to be stored in txt file when doing in C


#### ----- Momentum Gradient Descent ----- ####

#### ----- Momentum 

def ArmijoLineSearch(func, current_x, negative_gradient, current_gradient_xy, phi, alpha0, rho=0.5, c1=1e-4):
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
    derphi = np.dot(current_gradient_xy, negative_gradient)
    phi_a = gradient(current_x + alpha0*negative_gradient)
    
    while not phi_a <= phi + c1*alpha0*derphi:
        alpha0 = alpha0 * rho
        phi_a = gradient(current_x + alpha0*negative_gradient)
    
    return alpha0, phi_a


def GradientDescent(func, gradient, x, alpha, tol=1e-5, max_iter=2000):
    # initialize x, f(x), and -f'(x)
    
    current_x = x #starting x values
    current_y = gradient(current_x) # starting y values
    current_gradient_xy = np.array(grad) # gradient (downwards slope)
    norm_gradient = np.linalg.norm(current_gradient_xy) # absolute value of gradient
    
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [current_x] # curve_x and curve_y to store all the x and y axis
    curve_y = [current_y]
    # take steps
    while norm_gradient > tol and num_iter < max_iter: # while abs(gradient) > tolerance/threshold (self determined) and within counter of 1000
        # calculate new x, f(x), and -f'(x)
        negative_gradient = -current_gradient_xy # -ve gradient
        
        #With momentum term 1b
        alpha, current_y = ArmijoLineSearch(func, current_x, negative_gradient, current_gradient_xy, current_y, alpha0=alpha)
        
        current_x = current_x + alpha * negative_gradient # new x value
        current_gradient_xy = np.array(grad) # new gradient
        norm_gradient = np.linalg.norm(current_gradient_xy) # new abs(gradient)
        
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(current_x)
        curve_y.append(current_y)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.6f}'.format(num_iter, current_y, current_x, norm_gradient))
    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(current_y, current_x))
    
    return np.array(curve_x), np.array(curve_y) # to be stored in txt file when doing in C


#### ----- Gradient Descent with Newton's Algorithm ----- ####

#### ----- Newton's Algorithm 
def Newtons_Method(func,max_iter,x,tol=1e-5,epsilon=10**-2):

    # Initialise values
    current_x = x
    current_y = gradient(current_x)
    current_gradient_xy = np.array(grad)
    negative_gradient = -current_gradient_xy
    
    current_hessian = np.array(hessian_vecshaped).reshape(2,2)
    norm_gradient = np.linalg.norm(current_gradient_xy)
    norm_hessian = np.linalg.norm(current_hessian)
    # save current x and y values
    curve_x = [current_x]           
    curve_y = [current_y]          
    
    num_iter = 0
    
    while norm_gradient > tol and num_iter < max_iter:
        # evaluate the gradient and hessian
        
        A = current_hessian + epsilon*np.eye(current_x.size) # A = hessian + epsilon * identity matrix
        negative_gradient = -current_gradient_xy
        
        # ∇f(x)+ Hess(f(x))h=0 ==  h= −Hess(f(x))^−1 ∇f(x)
        current_x = np.linalg.solve(A,(np.dot(A,current_x) + negative_gradient))
        
        
        current_y = gradient(current_x)
        current_gradient_xy = np.array(grad)
        current_hessian = np.array(hessian_vecshaped).reshape(2,2)

        norm_gradient = np.linalg.norm(current_gradient_xy)
        norm_hessian = np.linalg.norm(current_hessian)
        # record weight and cost
        num_iter += 1
        curve_x.append(current_x)
        curve_y.append(current_y)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.6f}'.format(num_iter, current_y, current_x, norm_gradient))
        # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(current_y, current_x))
    return np.array(curve_x), np.array(curve_y) # to be stored in txt file when doing in C



#### ----- Main function ----- ####


# Initialise start values
x = np.array([1,1])
xs, ys = GradientDescentSimple(func, gradient, x, 7)
#xs, ys = GradientDescent(func, gradient, x, alpha=0.1)
#xs, ys = Newtons_Method(func,100,x)
# plotFunc(x)
# plot(xs,ys)
# plt.show()