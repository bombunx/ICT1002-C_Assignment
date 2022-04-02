import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from matplotlib import rcParams


temp_x = []
curve_x = []
curve_y = []

def func(x):
    dim = len(x)
    #1st test case ### Matya's function 
    #y = 0.26 * (x[0] * x[0]+ x[1] * x[1]) - 0.48 *x[0] * x[1]
    
    #2nd test case ### Himmelblau's function 
    #y = (x[0] * x[0] + x[1] - 11) * (x[0] * x[0] + x[1] - 11)  + (x[0] + x[1] * x[1] - 7) * (x[0] + x[1] * x[1] - 7)
    
    #3rd test case ### Beale's function 
    p1= 1.5 - x[0] + x[0]*x[1];
    p2= 2.25 - x[0] +x[0]*x[1]*x[1]; 
    p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1];   
  
    y = p1*p1 + p2*p2 + p3*p3;
    return y

def plot(xs, ys):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    x = np.arange(-4.5, 4.5, 0.5)
    y = np.arange(-4.5, 4.5, 0.5)
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

with open('output.csv') as file:
    content = file.readlines()

for item in content:
    for char in item:
        # store y values in curve_y
        if char == "y":
            index = item.index(char)
            curve_y.append(item[index+4:index+12])
        # get x values from output file
        if char == "x":
            index = item.index(char)
            x = item[index+5:index+24].split(",  ")
            temp_x.append(x)

#  store x values in curve_x
for item in temp_x:
    x0 = np.array([float(item[0]),float(item[1])])
    curve_x.append(x0)

curve_x = np.array(curve_x)
curve_y = np.array(curve_y)

plot(curve_x, curve_y)