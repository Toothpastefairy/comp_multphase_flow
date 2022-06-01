import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

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


    def calc_pressure_gradient_z(self, pressure):
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        matrix[1:-1,1:] = 1 / self.rho_fluid * (pressure[1:-1,1:] * self.dr * self.RR[1:-1,1:] - \
                                                  pressure[1:-1,:-1]  * self.dr * self.RR[1:-1,:-1])
        matrix[:,0] = self.inlet[::-1]
    
        matrix[0,-1] = 1
        matrix[-1,-1] = 1
        pressure_gradient_z = self.make_diagonal(matrix)

        return pressure_gradient_z


###############################################################################
# Momentum equations in r-direction                                           #
###############################################################################

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


    def calc_pressure_gradient_r(self, pressure):
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] * self.dz - pressure[:-2,1:-1] * self.dz)
        pressure_gradient_r = self.make_diagonal(matrix)

        return pressure_gradient_r


###############################################################################
# Solving Momentum equations                                                  #
###############################################################################

    def solve_velocity_z(self, velocity_z, velocity_r, mu, pressure):
        convective_z = self.calc_convective_flux_z(velocity_z, velocity_r)
        diffusive_z =  self.calc_diffusive_flux_z(mu)

        # Diagonals for matrix
        diag_i_j =       convective_z[0] + diffusive_z[0] + self.boundary_i_j_z
        diag_iplus_j =  (convective_z[1] + diffusive_z[1] + self.boundary_iplus_j)[:-1]
        diag_imin_j =   (convective_z[2] + diffusive_z[2] + self.boundary_imin_j)[1:]
        diag_i_jplus =  (convective_z[3] + diffusive_z[3] + self.boundary_i_jplus)[:-(self.Nz+2)]
        diag_i_jmin =   (convective_z[4] + diffusive_z[4] + self.boundary_i_jmin)[self.Nz+2:]

        # save diagonals for corrector step
        self.diagonals_corrector["diag_i_J"] = convective_z[0] + diffusive_z[0] + self.boundary_i_j_z
        self.diagonals_corrector["diag_iplus_J"] = convective_z[1] + diffusive_z[1] + self.boundary_iplus_j

        # Boulding matrix
        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,-1,1,-(self.Nz+2), self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()
        # print(matrix_A.A)
        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_z(pressure)
        

        # Solve the matrix equation
        new_velocity_z = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_z = np.reshape(new_velocity_z, (self.Nr+2, self.Nz+2))

        # plt.matshow(matrix_A.A, vmin=-2, vmax=2)
        # plt.colorbar()
        # plt.grid()
        # plt.show()
        return new_velocity_z


    def solve_velocity_r(self, velocity_z, velocity_r, mu, pressure):
        convective_z = self.calc_convective_flux_r(velocity_z, velocity_r)
        diffusive_z =  self.calc_diffusive_flux_r(mu)

        # Diagonals for matrix
        diag_i_j =       convective_z[0] + diffusive_z[0] + self.boundary_i_j_z
        diag_iplus_j =  (convective_z[1] + diffusive_z[1] + self.boundary_iplus_j)[:-1]
        diag_imin_j =   (convective_z[2] + diffusive_z[2] + self.boundary_imin_j)[1:]
        diag_i_jplus =  (convective_z[3] + diffusive_z[3] + self.boundary_i_jplus)[:-(self.Nz+2)]
        diag_i_jmin =   (convective_z[4] + diffusive_z[4] + self.boundary_i_jmin_r)[self.Nz+2:]

        # save diagonals for corrector step
        self.diagonals_corrector["diag_I_j"] = convective_z[0] + diffusive_z[0] + self.boundary_i_j_z
        self.diagonals_corrector["diag_I_jplus"] = convective_z[3] + diffusive_z[3] + self.boundary_i_jplus

        # Building matrix
        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,-1,1,-(self.Nz+2), self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_r(pressure)

        # Solve the matrix equation
        new_velocity_r = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_r  = np.reshape(new_velocity_r, (self.Nr+2, self.Nz+2))
        return new_velocity_r


    def pressure_correction(self, velocity_z, velocity_r, mu):
        # R-direction
        convective_z = self.calc_convective_flux_r(velocity_z, velocity_r)
        diffusive_z =  self.calc_diffusive_flux_r(mu)

        # Diagonals for matrix
        diag_I_j =       convective_z[0] + diffusive_z[0] + self.boundary_i_j_z
        diag_I_jplus =  (convective_z[3] + diffusive_z[3] + self.boundary_i_jplus)
        
        # Z-direction
        convective_z = self.calc_convective_flux_z(velocity_z, velocity_r)
        diffusive_z =  self.calc_diffusive_flux_z(mu)

        # Diagonals for matrix
        diag_i_J =       convective_z[0] + diffusive_z[0] + self.boundary_i_j_z
        diag_iplus_J =  (convective_z[1] + diffusive_z[1] + self.boundary_iplus_j)
        
        
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        # Convective and diffusive in (i,j) for z-staggered
        
        matrix[1:-1,1:-1] = (self.RR[2:,1:-1] * self.dr)**2
        diag = self.make_diagonal(matrix)
        diag_imin_j = np.zeros(len(diag))
        # diag_imin_j[diag!=0] = diag[diag!=0] / self.diagonals_corrector["diag_i_J"][diag!=0]
        diag_imin_j[diag!=0] = diag[diag!=0] / diag_i_J[diag!=0]

        # Convective and diffusive in (i+1,j) for z-staggered
        matrix[1:-1,1:-1] = (self.RR[2:,1:-1] * self.dr)**2 
        diag = self.make_diagonal(matrix)
        diag_iplus_j = np.zeros(len(diag))
        # diag_iplus_j[diag!=0] = diag[diag!=0] / self.diagonals_corrector["diag_iplus_J"][diag!=0] 
        diag_iplus_j[diag!=0] = diag[diag!=0] / diag_iplus_J[diag!=0] 

        # Convective and diffusive in (i,j) for r-staggered
        matrix[1:-1,1:-1] = (self.dz)**2
        diag = self.make_diagonal(matrix)
        diag_i_jmin = np.zeros(len(diag))
        # diag_i_jmin[diag!=0] = diag[diag!=0] / self.diagonals_corrector["diag_I_j"][diag!=0]
        diag_i_jmin[diag!=0] = diag[diag!=0] / diag_I_j[diag!=0]

        # Convective and diffusive in (i,j+1) for r-staggered
        matrix[1:-1,1:-1] = self.dz**2
        diag = self.make_diagonal(matrix)
        diag_i_jplus = np.zeros(len(diag))
        # diag_i_jplus[diag!=0] = diag[diag!=0] / self.diagonals_corrector["diag_I_jplus"][diag!=0]
        diag_i_jplus[diag!=0] = diag[diag!=0] / diag_I_jplus[diag!=0]

        # Diagonals for matrix
        diag_i_j = diag_imin_j + diag_iplus_j + diag_i_jmin + diag_i_jplus
        
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
        print(diag_imin_j)
        print(diag_i_jmin)

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

    def velocity_corrections(self, velocity_z, velocity_r, mu, pressure_correction, relaxation_factor):
        matrix = np.zeros((self.Nr+2, self.Nz+2))

        a = np.zeros_like(matrix)
        # corrector for z-velocity
        # Convective and diffusive in (i,j) for z-staggered
        a[1:-1,1:-1] = 0.25*(velocity_r[2:,:-2] + velocity_r[2:,1:-1] - velocity_r[1:-1,:-2] - velocity_r[1:-1,1:-1]) * self.dz + \
                            (velocity_z[1:-1,1:-1] + velocity_z[1:-1,2:]) * self.RZ[2:,1:-1] * self.dr + \
                            -(velocity_z[1:-1,:-2] + velocity_z[1:-1,1:-1]) * self.RZ[1:-1,1:-1] * self.dr

        a[1:-1,1:-1] += -mu[1:-1,1:-1] * ((self.RZ[2:,1:-1] + self.RZ[1:-1,1:-1]) * self.dr / self.dz + 2 * self.dz / self.dr)
        
        velocity_z_new = np.copy(velocity_z)
        matrix[1:-1,1:-1] = (self.RR[2:,1:-1] * self.dr) / a[1:-1,1:-1]
        velocity_z_new[1:-1,1:-1] = velocity_z[1:-1,1:-1] + matrix[1:-1,1:-1] * \
                                (pressure_correction[1:-1,:-2] - pressure_correction[1:-1,1:-1])
        velocity_z_new = relaxation_factor * velocity_z_new + (1 - relaxation_factor) * velocity_z
        # velocity_z_new = relaxation_factor * velocity_z_new - (1 - relaxation_factor) * velocity_z

        # Convective and diffusive in (i,j) for r-staggered
        a[1:-1,1:-1] =  0.25*(velocity_r[2:,1:-1] - velocity_r[:-2,1:-1]) * self.dz + \
                            (velocity_z[1:-1,2:] + velocity_z[:-2,2:]) * self.RR[2:,1:-1] * self.dr + \
                            -(velocity_z[:-2,1:-1] + velocity_z[1:-1,1:-1]) * self.RR[1:-1,1:-1] * self.dr
        a[1:-1,1:-1] += -mu[1:-1,1:-1] * ((self.RR[2:,1:-1] + self.RR[1:-1,1:-1]) * self.dr / self.dz + 2 * self.dz / self.dr)

        velocity_r_new = np.copy(velocity_r)
        matrix[1:-1,1:-1] = (self.dz)**2 / a[1:-1,1:-1]
        velocity_r_new[1:-1,1:-1] = velocity_r[1:-1,1:-1] + matrix[1:-1,1:-1] * \
                                (pressure_correction[:-2,1:-1] - pressure_correction[1:-1,1:-1])
        velocity_r_new = relaxation_factor * velocity_r_new + (1 - relaxation_factor) * velocity_r
        # velocity_r_new = relaxation_factor * velocity_r_new - (1 - relaxation_factor) * velocity_r

        return velocity_z_new, velocity_r_new
        

