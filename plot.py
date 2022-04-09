import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from matplotlib import rcParams


temp_x = []
curve_x = []
curve_y = []

x_1 = []
x_2 = []
y = []
z = []

file = open('funcSurface.txt', 'r')
num_lines = sum(1 for line in open('funcSurface.txt', 'r'))

content = file.readlines()

#  get x_1 values
for index in range(len(content[0])):
    if content[0][index] == '[':
        index_start = index + 1
    if content[0][index] == "]":
        index_end = index - 1
        x1 = content[0][index_start:index_end].split(" ")
        
for item in x1:
    x_1.append(float(item))
x_1 = np.array(x_1)

#  get x_2 values
for index in range(len(content[1])):
    if content[1][index] == '[':
        index_start = index + 1
    if content[1][index] == "]":
        index_end = index - 1
        x2 = content[1][index_start:index_end].split(" ")
        
for item in x2:
    x_2.append(float(item))
x_2 = np.array(x_2)

#  get y values
for curr_line in range(num_lines-2): 
    for index in range(len(content[curr_line+2])): # start from 3rd line (y = [...])
        if content[curr_line+2][index] == '[':
            index_start = index + 1
        if content[curr_line+2][index] == "]":
            index_end = index - 1
            y0 = content[curr_line+2][index_start:index_end].split(" ")
            temp_y = []
            for item in y0:
                temp_y.append(float(item))
            y = np.array(temp_y)
            z.append(y) 
z = np.array(z)


def plot(xs, ys):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    X, Y = np.meshgrid(x_1, x_2)    
    plt.suptitle('Gradient Descent Method')

    ax1.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
    ax1.plot(xs[-1,0], xs[-1,1], 'ro')
    ax1.set(
        title='Path During Optimization Process',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax1.contour(X,Y,z)
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
            curve_y.append(float(item[index+4:index+12]))
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