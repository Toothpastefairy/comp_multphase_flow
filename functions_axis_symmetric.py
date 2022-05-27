import numpy as np

class MultiFlowAxis:

    def __init__(self, Nz, Nr, z_end, r_end, rho_fluid, mu_0):
        self.Nz = Nz
        self.Nr = Nr
        self.z_end = z_end
        self.r_end = r_end
        self.dz = z_end / Nz
        self.dr = r_end / Nr
        self.rho_fluid = rho_fluid
        self.mu_0 = mu_0

        # Center
        z = np.linspace( -self.dz / 2, z_end + self.dz/2, Nz + 2)
        r = np.linspace( -self.dr / 2, r_end + self.dr/2, Nr + 2)

        # Z-offset
        z_z = z - self.dz / 2
        r_z = np.copy(r)
        ZZ , RZ = np.meshgrid(z_z, r_z)
        self.RZ = np.transpose(RZ)
        velocity_z = 2*np.ones((Nz + 2, Nr + 2)) # at z-offset points
        velocity_z[0,:] = 0
        velocity_z[-1,:] = 0
        velocity_z[:,0] = 0
        velocity_z[:,-1] = 0

        # R-offset
        z_r = z
        r_r = r - self.dr / 2
        ZR , RR = np.meshgrid(z_r, r_r)
        self.RR = np.transpose(RR)
        velocity_r = 3*np.ones((Nz + 2, Nr + 2)) # at r-offset points


        return

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


    def boundary_conditions(self):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        # Central diagonal
        matrix[1,:] = np.ones(self.Nz + 2)
        matrix[-1,:] = -1 * np.ones(self.Nz + 2)
        diag_i_j = self.make_diagonal(matrix)

        return diag_i_j#, diag_iplus_j, diag_imin_j, diag_i_jplus, diag_i_jmin


    def make_diagonal(self, matrix):
        return np.reshape(np.copy(matrix), newshape=(-1),  order="F")


    def calc_pressure_gradient_z(self, pressure):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] * self.dr * self.RR[1:-1,1:-1] - \
                                                  pressure[:-2,1:-1]  * self.dr * self.RR[:-2,1:-1])
        pressure_gradient_z = self.make_diagonal(matrix)

        return pressure_gradient_z


    def calc_pressure_gradient_r(self, pressure):
        matrix = np.zeros((self.Nz + 2 , self.Nr+2))

        matrix[1:-1,1:-1] = 1 / self.rho_fluid * (pressure[1:-1,1:-1] * self.dz - pressure[1:-1,:-2] * self.dz)
        pressure_gradient_r = self.make_diagonal(matrix)

        return pressure_gradient_r