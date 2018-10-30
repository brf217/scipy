#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:30:40 2018

@author: feebr01
"""

import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


# define variables
y, m, x, b, n = sp.symbols(' y m x b n', real = True)

################################################################ regression minimization example
# find cost function and partial derivatives
cost = x**2 + y**2
x_cost = sp.diff(cost, x)
y_cost = sp.diff(cost, y)



# define variables
x_points = [-1,-2,1,1,2,3,4,5,6,7,8,9,10,11]
y_points = [-1,-2,1,1,2,3,4,5,6,7,8,9,10,11]
x_init = 5
y_init = 6
n_values = len(x_points)
learn_rate = .1

check_x = []
check_y = []

for i in range(200):
    x_temp = x_cost.subs(x, x_init)
    y_temp = y_cost.subs(y, y_init)

    x_init = x_init - x_temp*learn_rate
    y_init = y_init - y_temp*learn_rate
    
    print(x_init, y_init)
    check_x.append(x_init)
    check_y.append(y_init)
    plt.plot(check_x)
    plt.plot(check_y)

result = cost.subs([(x, x_temp), (y, y_temp)])
print(f'Result of cost function is {result} at x = {x_temp} and y = {y_temp}')
    




