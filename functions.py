import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.special import lambertw

def calc_mu(mu_0,Ny):
  #returns array of mu over the whole space
  return mu_0 * np.ones(Ny+2)

def diagonal_A(mu, boundary_Condition,dy):
  diag_A = np.ones(len(mu))
  diag_A[1:-1] = mu[2:] + 2* mu[1:-1] + mu[:-2]
  diag_A[0] = boundary_Condition[0] * dy
  diag_A[-1] = boundary_Condition[1] * dy
  return diag_A

def diagonal_B(mu,dy):
  diag_B = np.zeros(len(mu)-1)
  diag_B = -1*(mu[:-1] + mu[1:] )
  diag_B[-1] = 1*dy
  return diag_B

def diagonal_C(mu,dy):
  diag_C = np.zeros(len(mu)-1)
  diag_C = -1*(mu[:-1] + mu[1:] )
  diag_C[0] = 1*dy
  return diag_C


def TDMAsolver(a, b, c, d):
    # copying the arrays so they can be overwritten
    ac = np.copy(a)
    bc = np.copy(b)
    cc = np.copy(c)
    dc = np.copy(d)

    xc = np.zeros(len(bc))

    for i in range(1, len(d)):
        mc = ac[i-1]/bc[i-1]
        bc[i] = bc[i] - mc*cc[i-1] 
        dc[i] = dc[i] - mc*dc[i-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for i in range(len(d)-2, -1, -1):
        xc[i] = (dc[i]-cc[i]*xc[i+1])/bc[i]
        
    return xc


def simulate(boundary_Condition, mu, pressure_dif, press_bound, vt):
    global dy
    
    # Calculating the diagonals
    diag_A = diagonal_A(mu, boundary_Condition, dy, vt) / dy
    diag_B = diagonal_B(mu,dy, vt) / dy
    diag_C = diagonal_C(mu,dy, vt) / dy

    # Filling the diagonals into the matrix for comparison
    matrix_A = 1/(rho) * sp.diags(diagonals=(diag_A, diag_B, diag_C), offsets=(0,-1,1))
    matrix_A = sp.dia_matrix(matrix_A)
    A = np.diag(diag_A, k=0) + np.diag(diag_B, k=-1) + np.diag(diag_C,k=1)

    # Calculating the pressure difference
    pressure_difference = pressure_dif * np.ones((Ny+2))
    pressure_difference[0] = 2*press_bound[0]
    pressure_difference[-1] = 2*press_bound[1]
    
    #solution = la.spsolve(matrix_A,pressure_difference)
    solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

    return solution

def simulate_wallfunctions(boundary_Condition, mu, pressure_dif, press_bound, vt, C, tau_w):
    global dy

    # Calculating the diagonals
    diag_A = diagonal_A(mu, boundary_Condition, dy, vt) / dy
    diag_B = diagonal_B(mu,dy, vt) / dy
    diag_C = diagonal_C(mu,dy, vt) / dy

    # Calculating the pressure difference
    pressure_difference = pressure_dif * np.ones((Ny+2))
    pressure_difference[0] = 2*press_bound[0]
    pressure_difference[-1] = 2*press_bound[1]

    # Adding the forced velocity
    diag_A[1] =  pressure_difference[1] / (C * tau_w)
    diag_A[-2] =  pressure_difference[-2] / (C * tau_w)
    diag_B[0] = 0
    diag_C[1] = 0
    diag_B[-2] = 0
    diag_C[-1] = 0

    # Filling the diagonals into the matrix for comparison
    matrix_A = 1/(rho) * sp.diags(diagonals=(diag_A, diag_B, diag_C), offsets=(0,-1,1))
    matrix_A = sp.dia_matrix(matrix_A)
    A = np.diag(diag_A, k=0) + np.diag(diag_B, k=-1) + np.diag(diag_C,k=1)
    
    solution = la.spsolve(matrix_A,pressure_difference)
    #solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

    return solution

