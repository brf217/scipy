
#import sympy as sp
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import sympy as sp
import seaborn as sns

#define variables
y, m, x, b, n = sp.symbols('y m x b n')

################################################################################## Gradient Descent
# find cost function and partial derivatives
cost = (y-((m*x)+b))**2 # sum of squared errors
m_cost = sp.diff(cost, m)
b_cost = sp.diff(cost, b)

# define variables
x_points = [1,1,2,3,4,5,6,7,8,9,10,11,13,17,13,20,28]
y_points = [1,2,3,1,4,5,6,7,8,10,13,12,13,16,19,23,31]

learn_rate = .002

# initial hypothesis of no relationship between x and y
m_init = 0
b_init = 0

# get initial value of y_pred
y_pred_init = list(map(lambda x: m_init * x  + b_init, x_points))


m_temp = defaultdict(list)
b_temp = defaultdict(list)

# initiate open lists for recordkeeping at each iteration
check_m = []
check_b = []
check_error = []


for i in range(1000):
    total_error = 0
    # calculate value of cost function slopes
    for n in range(len(x_points)):
        m_temp[i].append(m_cost.subs(
                [(x, x_points[n]), (b, b_init), (m, m_init), (y, y_points[n])]))
        b_temp[i].append(b_cost.subs(
                [(x, x_points[n]), (b, b_init), (m, m_init), (y, y_points[n])]))
        
    # calculate total error
        total_error += cost.subs(
                [(x, x_points[n]), (b, b_init), (m, m_init), (y, y_points[n])])   
    
       
    # adjust slope values depending on outcome of gradient formulas
        m_init = m_init - np.mean(m_temp[i])*learn_rate
        b_init = b_init - np.mean(b_temp[i])*learn_rate
    
    # check values as minimization is performed
    print(m_init, b_init)
    check_m.append(m_init)
    check_b.append(b_init)
    check_error.append(total_error)

  
    
# fit predicted y values using model
y_pred = list(map(lambda x: m_init * x  + b_init, x_points))

    
    

################################################################################### GD Visualization
sns.set_style('darkgrid')
sns.set_palette('muted')
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (10,12), tight_layout = True)
ax1.plot(x_points,y_points, 'o', label = 'x_points')
ax1.plot(x_points, y_pred, label = 'y_pred_fitted')
ax1.plot(x_points, y_pred_init, '--', label = 'y_pred_initial')
ax1.set(title = 'Final Fit (Gradient Descent)', xlabel = 'x_value', ylabel = 'Y_value')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)

ax2.plot(check_m, 'x',label = 'slope_coefficient(m)' )
ax2.plot(check_b, 'x',label = 'intercept(b)')
ax2.set(title = 'Values of m & b @ Iteration', xlabel = 'Iteration', ylabel = 'Value')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels)

#ax2.plot(check_m)
#ax2.plot(check_b)

ax3.plot(check_error)
ax3.set(title = 'Total Model Error', xlabel = 'Iteration', ylabel = 'Total Sum of Squared Error (SSE)')
plt.legend()



################################################################## Check Using Matrix Implementation
int_array = np.ones((len(x_points),1), dtype = int)
x_array = np.array(x_points).reshape(len(x_points), 1)
y_array = np.array(y_points).reshape(len(x_points), 1)

x_matrix = sp.Matrix(np.c_[int_array, x_array])
y_matrix = sp.Matrix(y_array)

# matrix inversion technique
reg_fit_matrix = (x_matrix.T*x_matrix).inv() * x_matrix.T*y_matrix


################################################################## Check matrix vs. gradient descent
matrix_intercept = float(reg_fit_matrix[0])
matrix_slope = float(reg_fit_matrix[1])

gradient_intercept = float(b_init)
gradient_slope =float(m_init)

# plot matrix version
y_pred_matrix = list(map(lambda x: matrix_slope * x  + matrix_intercept, x_points))

sns.set_style('darkgrid')
sns.set_palette('muted')
fig2, (ax) = plt.subplots(figsize = (11.3,6))
ax.plot(x_points,y_points, 'o', label = 'x_points')
ax.plot(x_points, y_pred, label = 'y_pred_gradient')
ax.plot(x_points, y_pred_matrix, '--', label = 'y_pred_matrix')
ax.set(title = 'Gradient vs. Matrix Fit', xlabel = 'x_value', ylabel = 'Y_value')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)



matrix_error = 0
for n in range(len(x_points)):
    matrix_error += cost.subs(
            [(x, x_points[n]), (b, matrix_intercept), (m, matrix_slope), (y, y_points[n])])   


gradient_error = 0
for n in range(len(x_points)):
    gradient_error += cost.subs(
            [(x, x_points[n]), (b, gradient_intercept), (m, gradient_slope), (y, y_points[n])]) 

print(matrix_error, gradient_error)


