import math
from math import log10, floor
import numpy
import decimal
from decimal import Decimal

numpy.set_printoptions(precision=5, suppress=True, linewidth=100)

# ========== QUESTION 1 ==========

def inner_work(t, y, h):
    
    temp = t - (y**2)
    
    t_inc = t + h
    y_inc = y + h * temp
    c_inc = t - (y**2)

    return temp + c_inc
    
def euler_method(left, right, num_iter, original_y):
    
    h = (right - left)/num_iter
    
    for cur_iteration in range(0, num_iter):
        
        t = left
        y = original_y
        
        inner = inner_work(t, y, h)
        
        next_y = y + ((h/2) * inner)
        
        left = t + h
        original_y = next_y
        
    return original_y
    
ans1 = euler_method(0, 2, 10, 1)

print("{:.5f}".format(round(ans1, 5)), "\n")



# ========== QUESTION 2 ==========

def f(x, y):
    
    return x - y**2

def runge_kutta(left, right, y_0, n):
    
    h = (right - left)/n
    
    for cur_iteration in range(0, n):
        
        t = left
        y = y_0
        
        k1 = h * f(left, y_0)
        k2 = h * f((left + h/2), (y_0 + k1/2))
        k3 = h * f((left + h/2), (y_0 + k2/2))
        k4 = h * f((left + h), (y_0 + k3))
        
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        
        y_n = y_0 + k
        
        y_0 = y_n
        left = left + h
        
    return y_n
    
ans2 = runge_kutta(0, 2, 1, 10)

print("{:.5f}".format(round(ans2, 5)), "\n")
    


# ========== QUESTION 3 ==========

def solve_sys_eqns(matrix3, n):
    
    # Gaussian Elimination
    
    for i in range(n):
        
        for j in range(i + 1, n):
            
            div = matrix3[j][i]/matrix3[i][i]
            
            for k in range(n + 1):
                
                matrix3[j][k] = matrix3[j][k] - div * matrix3[i][k]
    
    ans3[n-1] = matrix3[n-1][n]/matrix3[n-1][n-1]
    
    # Backward Substitution
    
    for i in range(n - 2, -1, -1):
        
        ans3[i] = matrix3[i][n]
        
        for j in range(i+1, n):
            
            ans3[i] = ans3[i] - matrix3[i][j] * ans3[j]
            
        ans3[i] = ans3[i]/matrix3[i][i]
    
    return ans3
    
matrix3 = numpy.zeros((3, 4))
    
ans3 = numpy.zeros(3)
    
matrix3[0][0] = 2
matrix3[0][1] = matrix3[2][0] = -1
matrix3[0][2] = matrix3[1][0] = matrix3[1][2] = 1
matrix3[0][3] = 6
matrix3[1][1] = 3 
matrix3[1][3] = 0
matrix3[2][1] = 5
matrix3[2][2] = 4
matrix3[2][3] = -3
    
ans3 = solve_sys_eqns(matrix3, 3)    
    
print(ans3, "\n")

# ========== QUESTION 4-A ==========

matrix4 = numpy.zeros((4, 4))

matrix4[0][0] = matrix4[0][1] = matrix4[1][1] = matrix4[1][3] = 1
matrix4[0][2] = 0
matrix4[0][3] = matrix4[2][0] = matrix4[3][2] = 3 
matrix4[1][0] = matrix4[2][3] = matrix4[3][1] = 2
matrix4[1][2] = matrix4[2][1] = matrix4[2][2] = matrix4[3][0] = matrix4[3][3] = -1 

ans4a = numpy.linalg.det(matrix4)
print("{:.5f}".format(round(ans4a, 5)), "\n")

# ========== QUESTION 4-B ==========

def l_matrix(matrix, n):
    
    ans4b = numpy.zeros((4, 4))
    
    for i in range(n):
        
        for j in range(i + 1, n):
            
            div = matrix[j][i]/matrix[i][i]
            
            for k in range(n):
                
                matrix[j][k] = matrix[j][k] - div * matrix[i][k]
            
            ans4b[j][i] = div
    
    for i in range(4):
        
        ans4b[i][i] = 1
            
    return ans4b
    
ans4b = l_matrix(matrix4, 4)

print(ans4b, "\n")
        
# ========== QUESTION 4-C ==========

def u_matrix(matrix, n):
    
    for i in range(n):
        
        for j in range(i + 1, n):
            
            div = matrix[j][i]/matrix[i][i]
            
            for k in range(n):
                
                matrix[j][k] = matrix[j][k] - div * matrix[i][k]
    
    return matrix
    
ans4c = u_matrix(matrix4, 4)

print(ans4c, "\n")



# ========== QUESTION 5 ==========
    
def diag_dom(matrix5, n):
    
    for i in range(n):
        
        sum_column = 0
        
        for j in range(n):
            
            if i != j:
                sum_column += matrix5[i][j]
        
        if sum_column > matrix5[i][i]:
            return False
    
    return True
    
m5 = numpy.zeros((5, 5))

m5[0][0] = m5[1][1] = 9
m5[0][1] = m5[2][0] = m5[4][3] = 0
m5[0][2] = 5
m5[0][3] = m5[1][3] = m5[2][3] = m5[3][1] = m5[3][4] = m5[4][1] = 2
m5[0][4] = m5[1][2] = m5[1][4] = m5[2][1] = 1
m5[1][0] = m5[2][4] = m5[3][2] = m5[4][0] = 3 
m5[2][2] = 7
m5[3][0] = m5[4][2] = 4
m5[3][3] = 12
m5[4][4] = 8
    
ans5 = diag_dom(m5, 5)
    
print(ans5, "\n")
    
# ========== QUESTION 6 ==========

def pos_def(matrix6, n):
    
    for i in range(n):
        for j in range(i + 1, n):
            
            if matrix6[i][j] != matrix6[j][i]:
                return False
    
    for k in range(n):
        
        temp = numpy.zeros((k, k))
        
        for i in range(k):
            for j in range(k):
                temp[i][j] = matrix6[i][j]
        
        if numpy.linalg.det(temp) <= 0:
            return False
    
    return True

m6 = numpy.zeros((3, 3))

m6[0][0] = m6[0][1] = m6[1][0] = m6[2][2] = 2
m6[0][2] = m6[2][0] = 1 
m6[1][1] = 3

ans6 = pos_def(m6, 3)

print(ans6)
