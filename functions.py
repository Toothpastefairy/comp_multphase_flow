import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

class Multiflow:

    def __init__(self,  Ny, y_end, rho, pressure_difference ,pressure_boundary, boundary_condition, mu_0):
        
        self.Ny = Ny
        self.y_end = y_end
        self.rho = rho
        self.pressure_difference = np.ones(self.Ny) * pressure_difference
        self.pressure_difference = np.append(self.pressure_difference, pressure_boundary[1])
        self.pressure_difference = np.append(pressure_boundary[0], self.pressure_difference)
        self.boundary_condition = boundary_condition
        self.mu_0 = mu_0

        self.dy = y_end / Ny
        self.y = np.linspace(-y_end / 2 - self.dy, y_end / 2 + self.dy, self.Ny + 2)
        self.y_wall = -np.abs(self.y) + self.y_end / 2

        return


    def calc_mu(self, mu_0):
        return mu_0 * np.ones(self.Ny + 2)


    def diagonal_A(self, mu):
        diag_A = np.ones(len(mu))
        diag_A[1:-1] = mu[2:] + 2 * mu[1:-1] + mu[:-2]
        diag_A[0] = self.boundary_condition[0] * self.dy
        diag_A[-1] = self.boundary_condition[1] * self.dy
        return diag_A


    def diagonal_B(self, mu):
        diag_B = np.zeros(len(mu)-1)
        diag_B = -1*(mu[:-1] + mu[1:] )
        diag_B[-1] = 1 * self.dy
        return diag_B


    def diagonal_C(self, mu):
        diag_C = np.zeros(len(mu)-1)
        diag_C = -1*(mu[:-1] + mu[1:] )
        diag_C[0] = 1 * self.dy
        return diag_C


    def simulate(self, mu):
        
        # Calculating the diagonals
        diag_A = self.diagonal_A(mu) / self.dy
        diag_B = self.diagonal_B(mu) / self.dy
        diag_C = self.diagonal_C(mu) / self.dy

        # Filling the diagonals into the matrix for comparison
        matrix_A = 1/(self.rho) * sp.diags(diagonals=(diag_A, diag_B, diag_C), offsets=(0,-1,1))
        matrix_A = sp.dia_matrix(matrix_A)

        # Calculating the pressure difference
        pressure_difference = self.pressure_difference
        
        # Solving the matrix equations
        solution = la.spsolve(matrix_A, pressure_difference)
        # solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

        return solution


    def simulate_wallfunctions(self, mu, C, tau_w):

        # Calculating the diagonals
        diag_A = self.diagonal_A(mu) / self.dy
        diag_B = self.diagonal_B(mu) / self.dy
        diag_C = self.diagonal_C(mu) / self.dy

        # Calculating the pressure difference
        pressure_difference = self.pressure_difference

        # Adding the forced velocity
        diag_A[1] =  pressure_difference[1] / (C * tau_w)
        diag_A[-2] =  pressure_difference[-2] / (C * tau_w)
        diag_B[0] = 0
        diag_C[1] = 0
        diag_B[-2] = 0
        diag_C[-1] = 0

        # Filling the diagonals into the matrix for comparison
        matrix_A = 1/(self.rho) * sp.diags(diagonals=(diag_A, diag_B, diag_C), offsets=(0,-1,1))
        matrix_A = sp.dia_matrix(matrix_A)
        
        # Solving the matrix equations
        solution = la.spsolve(matrix_A, pressure_difference)
        #solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

        return solution


    def mixing_length_func(self, velocity, argument_type):

        # Determining where steady flow
        delta99 = self.y_wall[np.max(velocity) * 0.99 < velocity]
        delta99 = delta99[0]
        # delta99 = y_end / 2
        
        # Known constants
        labda = 0.09
        kappa = 0.41
        A = 26

        # Calculate stresses in different ways
        tau_s = np.abs((velocity[1] - velocity[0]) / self.dy)
        tau = tau_s + self.pressure_difference * self.y_wall
        tau_plus = tau / tau_s
        
        y_plus = self.y_wall * np.sqrt(tau_s * self.rho) / self.mu_0
        
        # Away from the boundary
        lm = labda * delta99 * np.ones(len(velocity))

        # Calculate Von Driest damping types
        if argument_type == "Driest":
            argument = - y_plus / A
        elif argument_type == "Spalding":
            argument = -y_plus * np.sqrt(tau_plus) / A
        elif argument_type == "Jones":
            argument = -y_plus * tau_plus / A

        # Apply damping
        damping = (1 - np.exp(argument))

        # Close to the boundary, add Von Driest damping
        mask = self.y_wall/delta99 < labda/kappa
        lm[mask] = kappa * self.y_wall[mask]
        lm*=damping

        return lm


    def calc_mu_Prandtl(self, velocity, argument_type):
        # Returns array of mu over the whole space
        mixing_length = self.mixing_length_func(velocity, argument_type)
        
        # Determine the velocity gradient
        velocity_diff = (velocity[2:] - velocity[:-2]) / (self.y[2:] - self.y[:-2])
        velocity_diff = np.append(velocity_diff, velocity_diff[-1])
        velocity_diff = np.append(velocity_diff[0], velocity_diff)

        # Calculate the effective viscosity
        mu_eff = self.rho * mixing_length**2 * np.abs(velocity_diff)

        return self.mu_0 * np.ones(self.Ny + 2) + mu_eff
    


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