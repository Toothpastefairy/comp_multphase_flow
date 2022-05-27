import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

class MultiFlowAxis:
###############################################################################
# Initialisation                                                              #
###############################################################################


    def __init__(self, Nz, Nr, z_end, r_end, rho_fluid, mu_0):
        # basic Bich variables
        self.Nz = Nz
        self.Nr = Nr
        self.z_end = z_end
        self.r_end = r_end
        self.dz = z_end / Nz
        self.dr = r_end / Nr
        self.rho_fluid = rho_fluid
        self.mu_0 = mu_0
        self.make_boundary_conditions()

        ### Staggered Grid Positions ##########################################
        # Center points
        z = np.linspace( -self.dz / 2, z_end + self.dz/2, Nz + 2)
        r = np.linspace( -self.dr / 2, r_end + self.dr/2, Nr + 2)

        # Z-offset points
        z_z = z - self.dz / 2
        r_z = np.copy(r)
        ZZ , RZ = np.meshgrid(z_z, r_z)
        self.RZ = np.transpose(RZ)

        # R-offset points
        z_r = z
        r_r = r - self.dr / 2
        ZR , RR = np.meshgrid(z_r, r_r)
        self.RR = np.transpose(RR)

        return


    def make_boundary_conditions(self):

        # Central diagonal
        diag_i_j = np.zeros((self.Nz+2, self.Nr + 2))
        diag_i_j[:,0] = np.ones(self.Nz + 2)
        diag_i_j[:,-1] = -1 * np.ones(self.Nz + 2)
        self.boundary_i_j = self.make_diagonal(diag_i_j)

        # i+ diagonal (1 below central)
        diag_iplus_j = np.zeros_like(diag_i_j)
        diag_iplus_j[:,-1] = np.ones(self.Nz + 2)
        diag_iplus_j[-1,:] = np.ones(self.Nr + 2)
        self.boundary_iplus_j = self.make_diagonal(diag_iplus_j)[1:]
        
        # i- diagonal (1 above central)
        diag_imin_j = np.zeros_like(diag_i_j)
        diag_imin_j[:,0] = np.ones(self.Nz + 2)
        diag_imin_j[0,:] = np.ones(self.Nr + 2)
        self.boundary_imin_j = self.make_diagonal(diag_imin_j)[:-1]

        # j+ diagonal (Nr + 2 below central)
        diag_i_jplus = np.zeros_like(diag_i_j)
        diag_i_jplus[:,-1] = np.ones(self.Nz + 2)
        diag_i_jplus[-1,:] = np.ones(self.Nr + 2)
        self.boundary_i_jplus = self.make_diagonal(diag_i_jplus)[:-self.Nz-2]

        # j- diagonal (Nr +2 above central)
        diag_i_jmin = np.zeros_like(diag_i_j)
        diag_i_jmin[:,0] = np.ones(self.Nz + 2)
        diag_i_jmin[0,:] = np.ones(self.Nr + 2)
        self.boundary_i_jmin = self.make_diagonal(diag_i_jmin)[self.Nz+2:]

        pressure_boundary = np.zeros_like(diag_i_j)

        return

    
    def make_diagonal(self, matrix):
        return np.reshape(np.copy(matrix), newshape=(-1),  order="F")


###############################################################################
# Momentum equations in z-direction                                           #
###############################################################################

    def calc_convective_flux_z(self, velocity_z, velocity_r):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))
        
        matrix[1:-1,1:-1] = 0.25*(velocity_r[:-2,2:] + velocity_r[1:-1,2:] - velocity_r[:-2,1:-1] - velocity_r[1:-1,1:-1]) * self.dz + \
                            (velocity_z[1:-1,1:-1] + velocity_z[2:,1:-1]) * self.RZ[1:-1,2:] * self.dr + \
                            -(velocity_z[:-2,1:-1] + velocity_z[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_z[2:,1:-1] + velocity_z[1:-1,1:-1]) * self.RZ[1:-1,2:] * self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_z[:-2,1:-1] + velocity_z[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_r[:-2,2:] + velocity_r[1:-1,2:]) * self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_r[:-2,1:-1] + velocity_r[1:-1,1:-1]) * self.dz
        diag_i_jmin = self.make_diagonal(matrix)
        
        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_diffusive_flux_z(self, mu):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        matrix[1:-1,1:-1] = -mu[1:-1,1:-1] * ((self.RZ[1:-1,2:] + self.RZ[1:-1,1:-1]) * self.dr / self.dz + 2 * self.dz / self.dr)
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RZ[1:-1,2:] * self.dr / self.dz
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RZ[1:-1,1:-1] * self.dr / self.dz
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_i_jmin = self.make_diagonal(matrix)

        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_pressure_gradient_z(self, pressure):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] * self.dr * self.RR[1:-1,1:-1] - \
                                                  pressure[:-2,1:-1]  * self.dr * self.RR[:-2,1:-1])
        pressure_gradient_z = self.make_diagonal(matrix)

        return pressure_gradient_z


###############################################################################
# Momentum equations in r-direction                                           #
###############################################################################

    def calc_convective_flux_r(self, velocity_z, velocity_r):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        matrix[1:-1,1:-1] = 0.25*(velocity_r[1:-1,2:] - velocity_r[1:-1,:-2]) * self.dz + \
                            (velocity_z[2:,1:-1] + velocity_z[2:,:-2]) * self.RR[1:-1,2:] * self.dr + \
                            -(velocity_z[1:-1,:-2] + velocity_z[1:-1,1:-1]) * self.RR[1:-1,1:-1] * self.dr
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_z[2:,1:-1] + velocity_z[2:,:-2]) * self.RR[1:-1,2:] * self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_z[1:-1,:-2] + velocity_z[1:-1,1:-1]) * self.RR[1:-1,1:-1] * self.dr
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_r[1:-1,1:-1] + velocity_r[1:-1,2:]) * self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_r[1:-1,1:-1] + velocity_r[1:-1,:-2]) * self.dz
        diag_i_jmin = self.make_diagonal(matrix)
        
        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_diffusive_flux_r(self, mu):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))
        
        matrix[1:-1,1:-1] = -mu[1:-1,1:-1] * ((self.RR[1:-1,2:] + self.RR[1:-1,1:-1]) * self.dr / self.dz + 2 * self.dz / self.dr)
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RR[1:-1,2:] * self.dr / self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RR[1:-1,1:-1] * self.dr / self.dz
        diag_i_jmin = self.make_diagonal(matrix)

        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_pressure_gradient_r(self, pressure):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] * self.dz - pressure[1:-1,:-2] * self.dz)
        pressure_gradient_r = self.make_diagonal(matrix)

        return pressure_gradient_r


###############################################################################
# Solving Momentum equations                                                  #
###############################################################################

    def solve_velocity_z(self, velocity_z, velocity_r, mu, pressure):
        convective_z = self.calc_convective_flux_z(velocity_z, velocity_r)
        diffusive_z =  self.calc_diffusive_flux_z(mu)

        # Diagonals for matrix
        diag_i_j =       convective_z[0] + diffusive_z[0] + self.boundary_i_j
        diag_iplus_j =  (convective_z[1] + diffusive_z[1])[1:] + self.boundary_iplus_j
        diag_imin_j =   (convective_z[2] + diffusive_z[2])[:-1] + self.boundary_imin_j
        diag_i_jplus =  (convective_z[3] + diffusive_z[3])[:-self.Nz-2] + self.boundary_i_jplus
        diag_i_jmin =   (convective_z[4] + diffusive_z[4])[self.Nz+2:] + self.boundary_i_jmin

        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,1,-1,-self.Nz-2, self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_z(pressure)

        # Solve the matrix equation
        new_velocity_z = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_z = np.reshape(new_velocity_z, (self.Nz + 2 , self.Nr+2))
        return new_velocity_z


    def solve_velocity_r(self, velocity_z, velocity_r, mu, pressure):
        convective_z = self.calc_convective_flux_r(velocity_z, velocity_r)
        diffusive_z =  self.calc_diffusive_flux_r(mu)

        # Diagonals for matrix
        diag_i_j =       convective_z[0] + diffusive_z[0] + self.boundary_i_j
        diag_iplus_j =  (convective_z[1] + diffusive_z[1])[1:] + self.boundary_iplus_j
        diag_imin_j =   (convective_z[2] + diffusive_z[2])[:-1] + self.boundary_imin_j
        diag_i_jplus =  (convective_z[3] + diffusive_z[3])[:-self.Nz-2] + self.boundary_i_jplus
        diag_i_jmin =   (convective_z[4] + diffusive_z[4])[self.Nz+2:] + self.boundary_i_jmin

        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,1,-1,-self.Nz-2, self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_r(pressure)

        # Solve the matrix equation
        new_velocity_r = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_r = np.reshape(new_velocity_r, (self.Nz + 2 , self.Nr+2))
        return new_velocity_r


    def recalculate_pressure():
        return

