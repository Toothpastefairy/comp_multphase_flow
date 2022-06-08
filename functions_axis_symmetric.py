import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from matplotlib.colors import LogNorm

class MultiFlowAxis:
###############################################################################
# Initialisation                                                              #
###############################################################################


    def __init__(self, Nz, Nr, z_end, r_end, rho_fluid, mu_0, inlet):
        # basic Bich variables
        self.Nz = Nz
        self.Nr = Nr
        self.z_end = z_end
        self.r_end = r_end
        self.dz = z_end / Nz
        self.dr = r_end / Nr
        self.rho_fluid = rho_fluid
        self.mu_0 = mu_0
        self.inlet = inlet
        self.diagonals_corrector = {}
        

        ### Staggered Grid Positions ##########################################
        # Center points
        z = np.linspace( -self.dz / 2, z_end + self.dz/2, Nz + 2)
        r = np.linspace( -self.dr / 2, r_end + self.dr/2, Nr + 2)
        self.Z, self.R = np.meshgrid(z, r)

        # Z-offset points
        z_z = z - self.dz / 2
        r_z = np.copy(r)
        ZZ , self.RZ = np.meshgrid(z_z, r_z)

        # R-offset points
        z_r = z
        r_r = r - self.dr / 2
        ZR , self.RR = np.meshgrid(z_r, r_r)
        return


    def make_boundary_conditions(self):

        # Central diagonal z-direction
        diag_i_j = np.zeros((self.Nr+2, self.Nz+2))
        diag_i_j[:,0] = 1
        diag_i_j[0,:] = 1
        diag_i_j[-1,:] = 1
        diag_i_j[:,-1] = 1
        self.boundary_i_j_z = self.make_diagonal(diag_i_j)

        # Central diagonal r-direction
        diag_i_j = np.zeros((self.Nr+2, self.Nz + 2))
        diag_i_j[:,0] = 1
        diag_i_j[0,:] = 1
        diag_i_j[-1,:] = 1
        diag_i_j[:,-1] = 1
        self.boundary_i_j_r = self.make_diagonal(diag_i_j)

        # i+ diagonal (1 above central)
        diag_iplus_j = np.zeros_like(diag_i_j)
        self.boundary_iplus_j = self.make_diagonal(diag_iplus_j)
        
        # i- diagonal (1 below central)
        diag_imin_j = np.zeros_like(diag_i_j)
        diag_imin_j[1:-1,-1] = -1
        self.boundary_imin_j = self.make_diagonal(diag_imin_j)

        # j+ diagonal (Nz + 2 above central)
        diag_i_jplus = np.zeros_like(diag_i_j)
        diag_i_jplus[0,1:-1] = 1
        self.boundary_i_jplus = self.make_diagonal(diag_i_jplus)

        # j- diagonal z-direction (Nz +2 below central)
        diag_i_jmin = np.zeros_like(diag_i_j)
        diag_i_jmin[-1,1:-1] = -1
        self.boundary_i_jmin = self.make_diagonal(diag_i_jmin)

        # j- diagonal r-direction (Nz +2 below central)
        diag_i_jmin = np.zeros_like(diag_i_j)
        diag_i_jmin[-1,1:-1] = 1
        self.boundary_i_jmin_r = self.make_diagonal(diag_i_jmin)
        return

    
    def make_diagonal(self, matrix):
        return np.reshape(np.copy(matrix), newshape=(-1),  order="C")


###############################################################################
# Momentum equations in z-direction                                           #
###############################################################################

    def calc_diagonals_z(self, vel_old_z, vel_old_r, nu):
        u = vel_old_z
        v = vel_old_r
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        # all u_ij terms
        # Convective
        matrix[1:-1,1:-1] += -1* (
            + ( 1/4 * u[1:-1,2:] + 1/4 * u[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr +
            - (1/4 * u[1:-1,:-2] + 1/4 * u[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr +
            + (1/4 * v[2:,:-2] + 1/4 * v[2:,1:-1]) * self.dz +
            - (1/4 * v[1:-1,:-2] + 1/4 * v[1:-1,1:-1]) * self.dz
        )
        # Diffusive
        matrix[1:-1,1:-1] += (
            nu[1:-1,1:-1] *self.RZ[1:-1,1:-1] * self.dr / self.dz * -1 -
            nu[1:-1,:-2] * self.RZ[1:-1,:-2] * self.dr / self.dz +
            (nu[1:-1,:-2] + nu[1:-1,1:-1] + nu[2:,1:-1] + nu[2:,:-2]) / 4 * self.dz / self.dr * -1 -
            (nu[2:,:-2] + nu[2:,1:-1] + nu[1:-1,1:-1] + nu[1:-1,:-2]) / 4 * self.dz / self.dr
        )
        diag_i_j = self.make_diagonal(matrix)
        
        
        # all u_i_jplus terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] +=-1 * (
            + (1/4 * u[1:-1,:-2] + 1/4*u[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr
        )
        # Diffusive
        matrix[1:-1,1:-1] += nu[1:-1,1:-1] *self.RZ[1:-1,1:-1] * self.dr / self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        
        # all u_i_jmin terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1* (
            + (1/4 * u[1:-1,:-2] + 1/4 * u[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr * -1
        )
        # Diffusive
        matrix[1:-1,1:-1] += - nu[1:-1,:-2] * self.RZ[1:-1,:-2] * self.dr / self.dz * -1
        diag_i_jmin = self.make_diagonal(matrix)


        # all u_iplus_j terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1 * (
            + (1/4 * v[2:,:-2] + 1/4 * v[2:,1:-1]) * self.dz * -1
        )
        # Diffusive
        matrix[1:-1,1:-1] += (nu[1:-1,:-2] + nu[1:-1,1:-1] + nu[2:,1:-1] + nu[2:,:-2]) / 4 * self.dz / self.dr 
        diag_iplus_j = self.make_diagonal(matrix)

        # all u_imin_j terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1 * (
            (1/4 * v[1:-1,:-2] + 1/4 * v[1:-1,1:-1]) * self.dz * -1
        )
        # Diffusive
        matrix[1:-1,1:-1] += - (nu[2:,:-2] + nu[2:,1:-1] + nu[1:-1,1:-1] + nu[1:-1,:-2]) / 4 * self.dz / self.dr * -1
        diag_imin_j = self.make_diagonal(matrix)

        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_convective_flux_z(self, velocity_z, velocity_r):
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        
        matrix[1:-1,1:-1] = 0.25*((velocity_r[2:,:-2] + velocity_r[2:,1:-1] - velocity_r[1:-1,:-2] - velocity_r[1:-1,1:-1]) * self.dz + \
                            (velocity_z[1:-1,1:-1] + velocity_z[1:-1,2:]) * self.RZ[2:,1:-1] * self.dr + \
                            -(velocity_z[1:-1,:-2] + velocity_z[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr)
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_z[1:-1,2:] + velocity_z[1:-1,1:-1]) * self.RZ[2:,1:-1] * self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_z[1:-1,:-2] + velocity_z[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_r[2:,:-2] + velocity_r[2:,1:-1]) * self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_r[1:-1,:-2] + velocity_r[1:-1,1:-1]) * self.dz
        diag_i_jmin = self.make_diagonal(matrix)
        
        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_diffusive_flux_z(self, mu):
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        matrix[1:-1,1:-1] = -mu[1:-1,1:-1] * ((self.RZ[2:,1:-1] + self.RZ[1:-1,1:-1]) * self.dr / self.dz + 2 * self.dz / self.dr)
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RZ[2:,1:-1] * self.dr / self.dz
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RZ[1:-1,1:-1] * self.dr / self.dz
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_i_jmin = self.make_diagonal(matrix)

        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_pressure_gradient_z(self, pressure, vel_old_z, vel_old_r):
        u = vel_old_z
        v = vel_old_r
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] * self.dr * self.RR[1:-1,1:-1] - \
                                                  pressure[1:-1,:-2]  * self.dr * self.RR[1:-1,1:-1])

        matrix[1:-1,1:-1] += -1 * self.RZ[1:-1,1:-1] * (-1/4 * u[1:-1,:-2]**2 - 1/2 * u[1:-1,2:] * u[1:-1,1:-1] - 1/4 * u[1:-1,2:]**2 )
        matrix[1:-1,1:-1] += -1 * self.RZ[1:-1,1:-1] * (-1/4 * u[1:-1,1:-1]**2 - 1/2 * u[1:-1,1:-1] * u[1:-1,:-2] - 1/4 * u[1:-1,1:-1]**2 ) * -1

        matrix[:,0] = self.inlet[::-1]
    
        matrix[0,-1] = 1
        matrix[-1,-1] = 1
        pressure_gradient_z = self.make_diagonal(matrix)

        return pressure_gradient_z


###############################################################################
# Momentum equations in r-direction                                           #
###############################################################################

    def calc_diagonals_r(self, vel_old_z, vel_old_r, nu):
        u = vel_old_z
        v = vel_old_r
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        # all u_ij terms
        # Convective
        matrix[1:-1,1:-1] += -1 * ( 
            self.RR[1:-1,1:-1] * self.dr * (1/4 * u[1:-1,2:] + 1/4 * u[:-2,2:]) -
            self.RR[1:-1,1:-1] * self.dr * (1/4 * u[:-2,1:-1] + 1/4 * u[1:-1,1:-1]) +
            self.dz * (1/4 * v[1:-1,1:-1] + 1/4 * v[2:,1:-1]) - 
            self.dz * (1/4 * v[1:-1,1:-1] + 1/4 * v[:-2,1:-1])
        )
        # Diffusive
        matrix[1:-1,1:-1] += (
            (nu[1:-1,1:-1] + nu[1:-1,2:] + nu[:-2,1:-1] + nu[:-2,2:]) / 4 * self.RR[1:-1,1:-1] * self.dr / self.dz * -1 -
            (nu[1:-1,1:-1] + nu[:-2,1:-1] + nu[:-2,:-2] + nu[1:-1,:-2]) / 4 * self.RR[1:-1,1:-1] * self.dr / self.dz +
            nu[1:-1,1:-1] * self.dz / self.dr * -1 - 
            nu[:-2,1:-1] * self.dz / self.dr
        )
        diag_i_j = self.make_diagonal(matrix)
        
        
        # all u_i_jplus terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1 * (
            self.RR[1:-1,1:-1] * self.dr * (1/4 * u[1:-1,2:] + 1/4 * u[:-2,2:])
        )
        # Diffusive
        matrix[1:-1,1:-1] += (nu[1:-1,1:-1] + nu[1:-1,2:] + nu[:-2,1:-1] + nu[:-2,2:]) / 4 * self.RR[1:-1,1:-1] * self.dr / self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        
        # all u_i_jmin terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1 * (
            -self.RR[1:-1,1:-1] * self.dr * (1/4 * u[:-2,1:-1] + 1/4 * u[1:-1,1:-1])
        )
        # Diffusive
        matrix[1:-1,1:-1] += - (nu[1:-1,1:-1] + nu[:-2,1:-1] + nu[:-2,:-2] + nu[1:-1,:-2]) / 4 * self.RR[1:-1,1:-1] * self.dr / self.dz * -1
        diag_i_jmin = self.make_diagonal(matrix)


        # all u_iplus_j terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1 * (
            self.dz * (1/4 * v[1:-1,1:-1] + 1/4 * v[2:,1:-1])
        )
        # Diffusive
        matrix[1:-1,1:-1] += nu[1:-1,1:-1] * self.dz / self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        # all u_imin_j terms
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        # Convective
        matrix[1:-1,1:-1] += -1 * (
            - self.dz * (1/4 * v[1:-1,1:-1] + 1/4 * v[:-2,1:-1])
        )
        # Diffusive
        matrix[1:-1,1:-1] += nu[:-2,1:-1] * self.dz / self.dr
        diag_imin_j = self.make_diagonal(matrix)

        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin

    def calc_convective_flux_r(self, velocity_z, velocity_r):
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        matrix[1:-1,1:-1] = 0.25*((velocity_r[2:,1:-1] - velocity_r[:-2,1:-1]) * self.dz + \
                            (velocity_z[1:-1,2:] + velocity_z[:-2,2:]) * self.RR[2:,1:-1] * self.dr + \
                            -(velocity_z[:-2,1:-1] + velocity_z[1:-1,1:-1]) * self.RR[1:-1,1:-1] * self.dr)
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_z[1:-1,2:] + velocity_z[:-2,2:]) * self.RR[2:,1:-1] * self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_z[:-2,1:-1] + velocity_z[1:-1,1:-1]) * self.RR[1:-1,1:-1] * self.dr
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = 0.25*(velocity_r[1:-1,1:-1] + velocity_r[2:,1:-1]) * self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = -0.25*(velocity_r[1:-1,1:-1] + velocity_r[:-2,1:-1]) * self.dz
        diag_i_jmin = self.make_diagonal(matrix)
        
        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_diffusive_flux_r(self, mu):
        matrix = np.zeros((self.Nr+2, self.Nz+2))
        
        matrix[1:-1,1:-1] = -mu[1:-1,1:-1] * ((self.RR[2:,1:-1] + self.RR[1:-1,1:-1]) * self.dr / self.dz + 2 * self.dz / self.dr)
        diag_i_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_iplus_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.dz / self.dr
        diag_imin_j = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RR[2:,1:-1] * self.dr / self.dz
        diag_i_jplus = self.make_diagonal(matrix)

        matrix[1:-1,1:-1] = mu[1:-1,1:-1] * self.RR[1:-1,1:-1] * self.dr / self.dz
        diag_i_jmin = self.make_diagonal(matrix)

        return diag_i_j, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def calc_pressure_gradient_r(self, pressure, vel_old_z, vel_old_r):
        u = vel_old_z
        v = vel_old_r
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] - pressure[:-2,1:-1]) * self.dz
        matrix[1:-1,1:-1] += -1* self.dz * (-1/4 * v[1:-1,1:-1]**2 - 1/2 * v[1:-1,1:-1] * v[2:,1:-1] - 1/4 * v[2:,1:-1])
        matrix[1:-1,1:-1] += -1* self.dz * (-1/4 * v[1:-1,1:-1]**2 - 1/2 * v[1:-1,1:-1] * v[:-2,1:-1] - 1/4 * v[:-2,1:-1]) * -1

        matrix[0,-1] = 1
        matrix[-1,-1] = 1

        pressure_gradient_r = self.make_diagonal(matrix)

        return pressure_gradient_r


###############################################################################
# Solving Momentum equations                                                  #
###############################################################################

    def solve_velocity_z(self, velocity_z, velocity_r, mu, pressure):
        # convective_z = self.calc_convective_flux_z(velocity_z, velocity_r)
        # diffusive_z =  self.calc_diffusive_flux_z(mu)

        diags = self.calc_diagonals_z(velocity_z, velocity_r, mu / self.rho_fluid)

        # Diagonals for matrix
        diag_i_j =       diags[0] + self.boundary_i_j_z
        diag_iplus_j =  (diags[1] + self.boundary_iplus_j)[:-1]
        diag_imin_j =   (diags[2] + self.boundary_imin_j)[1:]
        diag_i_jplus =  (diags[3] + self.boundary_i_jplus)[:-(self.Nz+2)]
        diag_i_jmin =   (diags[4] + self.boundary_i_jmin)[self.Nz+2:]

        # Boulding matrix
        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,-1,1,-(self.Nz+2), self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()
        # plt.matshow(matrix_A.A, vmin=-.1, vmax=.1)#, norm=LogNorm(np.min(np.log10(matrix_A)), np.max(np.log10(matrix_A))))
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        # plt.show()
        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_z(pressure, velocity_z, velocity_r)
        

        # Solve the matrix equation
        new_velocity_z = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_z = np.reshape(new_velocity_z, (self.Nr+2, self.Nz+2))
        new_velocity_z = np.flipud(new_velocity_z)

        # plt.matshow(matrix_A.A, vmin=-2, vmax=2)
        # plt.colorbar()
        # plt.grid()
        # plt.show()
        return new_velocity_z


    def solve_velocity_r(self, velocity_z, velocity_r, mu, pressure):
        diags = self.calc_diagonals_r(velocity_z, velocity_r, mu / self.rho_fluid)

       # Diagonals for matrix
        diag_i_j =       diags[0] + self.boundary_i_j_r
        diag_iplus_j =  (diags[1] + self.boundary_iplus_j)[:-1]
        diag_imin_j =   (diags[2] + self.boundary_imin_j)[1:]
        diag_i_jplus =  (diags[3] + self.boundary_i_jplus)[:-(self.Nz+2)]
        diag_i_jmin =   (diags[4] + self.boundary_i_jmin_r)[self.Nz+2:]

        # Building matrix
        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,-1,1,-(self.Nz+2), self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_r(pressure, velocity_z, velocity_r)

        # Solve the matrix equation
        new_velocity_r = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_r  = np.reshape(new_velocity_r, (self.Nr+2, self.Nz+2))
        return new_velocity_r


    def pressure_correction(self, velocity_z, velocity_r, mu):
        u = velocity_z
        v = velocity_r
        # R-direction
        diags = self.calc_diagonals_r(velocity_z, velocity_r, mu / self.rho_fluid)

        # Diagonals for matrix
        diag_I_j =       diags[0] + self.boundary_i_j_z
        dummy_matrix = np.zeros((self.Nr+2, self.Nz+2))
        dummy_matrix[1:-1,1:-1] += -1 * self.RZ[1:-1,1:-1] * (-1/4 * u[1:-1,:-2]**2 - 1/2 * u[1:-1,2:] * u[1:-1,1:-1] - 1/4 * u[1:-1,2:]**2 )
        dummy_matrix[1:-1,1:-1] += -1 * self.RZ[1:-1,1:-1] * (-1/4 * u[1:-1,1:-1]**2 - 1/2 * u[1:-1,1:-1] * u[1:-1,:-2] - 1/4 * u[1:-1,1:-1]**2 ) * -1
        diag_I_j -= self.make_diagonal(dummy_matrix)

        diag_I_jplus =  (diags[3] + self.boundary_i_jplus)
        
        # Z-direction
        diags = self.calc_diagonals_z(velocity_z, velocity_r, mu / self.rho_fluid)

        # Diagonals for matrix
        diag_i_J =       diags[0] + self.boundary_i_j_z
        dummy_matrix = np.zeros((self.Nr+2, self.Nz+2))
        dummy_matrix[1:-1,1:-1] += -1* self.dz * (-1/4 * v[1:-1,1:-1]**2 - 1/2 * v[1:-1,1:-1] * v[2:,1:-1] - 1/4 * v[2:,1:-1])
        dummy_matrix[1:-1,1:-1] += -1* self.dz * (-1/4 * v[1:-1,1:-1]**2 - 1/2 * v[1:-1,1:-1] * v[:-2,1:-1] - 1/4 * v[:-2,1:-1]) * -1
        diag_i_J -= self.make_diagonal(dummy_matrix)

        diag_iplus_J =  (diags[1] + self.boundary_iplus_j)
        
        
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        # Convective and diffusive in (i,j) for z-staggered
        
        matrix[1:-1,1:-1] = (self.RR[2:,1:-1] * self.dr)**2
        
        diag = self.make_diagonal(matrix)
        diag_imin_j = np.zeros(len(diag))
        diag_imin_j[diag_i_J!=0] = diag[diag_i_J!=0] / diag_i_J[diag_i_J!=0]

        # Convective and diffusive in (i+1,j) for z-staggered
        matrix[1:-1,1:-1] = (self.RR[2:,1:-1] * self.dr)**2 
        diag = self.make_diagonal(matrix)
        diag_iplus_j = np.zeros(len(diag))
        diag_iplus_j[diag_iplus_J!=0] = diag[diag_iplus_J!=0] / diag_iplus_J[diag_iplus_J!=0] 

        # Convective and diffusive in (i,j) for r-staggered
        matrix[1:-1,1:-1] = (self.dz)**2
        diag = self.make_diagonal(matrix)
        diag_i_jmin = np.zeros(len(diag))
        diag_i_jmin[diag_I_j!=0] = diag[diag_I_j!=0] / diag_I_j[diag_I_j!=0]

        # Convective and diffusive in (i,j+1) for r-staggered
        matrix[1:-1,1:-1] = self.dz**2
        diag = self.make_diagonal(matrix)
        diag_i_jplus = np.zeros(len(diag))
        diag_i_jplus[diag_I_jplus!=0] = diag[diag_I_jplus!=0] / diag_I_jplus[diag_I_jplus!=0]

        # Diagonals for matrix
        diag_i_j = diag_imin_j + diag_iplus_j + diag_i_jmin + diag_i_jplus #+ self.boundary_i_j_z
        
        diag_i_j =      diag_i_j
        diag_iplus_j =  (diag_iplus_j)[:-1]
        diag_imin_j =   (diag_imin_j)[1:]
        diag_i_jplus =  (diag_i_jplus)[:-(self.Nz+2)]
        diag_i_jmin =   (diag_i_jmin)[self.Nz+2:]


        rhs = np.zeros_like(matrix)
        rhs[1:-1,1:-1] = velocity_z[1:-1,1:-1] * self.RZ[1:-1,1:-1] * self.dr - \
                         velocity_z[1:-1,2:] * self.RZ[1:-1,2:] * self.dr + \
                         velocity_r[1:-1,1:-1] * self.dz - velocity_r[2:,1:-1] * self.dz
        rhs = self.make_diagonal(rhs) 

        matrix_A = sp.diags(
            diagonals=(diag_i_j, -diag_imin_j, -diag_iplus_j, -diag_i_jmin, -diag_i_jplus),
            offsets=(0,-1,1,-(self.Nz+2), self.Nz+2)
        )
        # print(diag_imin_j)
        # print(diag_i_jmin)

        matrix_A = sp.dia_matrix(matrix_A).tocsr()
        plt.matshow(matrix_A.A, vmin=-2, vmax=2)
        plt.colorbar()
        plt.show()

        pressure_correction = la.spsolve(matrix_A, rhs)
        pressure_correction = np.reshape(pressure_correction, (self.Nr+2, self.Nz+2))
        pressure_correction[0,:] = pressure_correction[1,:]
        pressure_correction[-1,:] = pressure_correction[-2,:]
        pressure_correction[:,0] = pressure_correction[:,1]
        return pressure_correction

    def velocity_corrections(self, velocity_z, velocity_r, mu, 
                            pressure_correction, relaxation_factor,
                            velocity_z_star, velocity_r_star):
        # R-direction
        diags = self.calc_diagonals_r(velocity_z, velocity_r, mu / self.rho_fluid)

        # Diagonals for matrix
        diag_I_j =       diags[0] + self.boundary_i_j_z
        
        # Z-direction
        diags = self.calc_diagonals_z(velocity_z, velocity_r, mu / self.rho_fluid)

        # Diagonals for matrix
        diag_i_J =       diags[0] + self.boundary_i_j_z
        
        
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        # Convective and diffusive in (i,j) for z-staggered
        matrix[1:-1,1:-1] = (self.RR[2:,1:-1] * self.dr)
        diag = self.make_diagonal(matrix)
        d_i_J = np.zeros(len(diag))
        d_i_J[diag_i_J!=0] = diag[diag_i_J!=0] / diag_i_J[diag_i_J!=0]
        d_i_J = np.reshape(d_i_J, (self.Nr+2, self.Nz+2))

        # Convective and diffusive in (i,j) for r-staggered
        matrix[1:-1,1:-1] = (self.dz)
        diag = self.make_diagonal(matrix)
        d_I_j = np.zeros(len(diag))
        d_I_j[diag_I_j!=0] = diag[diag_I_j!=0] / diag_I_j[diag_I_j!=0]
        d_I_j = np.reshape(d_I_j, (self.Nr+2, self.Nz+2))

        # Correct velocities
        vel_new_z = np.copy(velocity_z)
        vel_new_r = np.copy(velocity_r)

        vel_new_z[1:-1,1:-1] = velocity_z_star[1:-1,1:-1] + d_i_J[1:-1,1:-1] * (pressure_correction[1:-1,:-2] - pressure_correction[1:-1,1:-1])
        vel_new_r[1:-1,1:-1] = velocity_r_star[1:-1,1:-1] + d_I_j[1:-1,1:-1] * (pressure_correction[:-2,1:-1] - pressure_correction[1:-1,1:-1])

        vel_new_z = relaxation_factor * vel_new_z + (1 - relaxation_factor) * velocity_z
        vel_new_r = relaxation_factor * vel_new_r + (1 - relaxation_factor) * velocity_r

        return vel_new_z, vel_new_r

