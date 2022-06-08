import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

class Multiflow:
    """
    class containing all function to solve a 1D multiphase flow system using 
    one-way coupling and a eulerian approach (euler-euler)
    """

    def __init__(self,  Ny, y_end, rho, pressure_difference ,pressure_boundary, boundary_condition, mu_0, angle):
        
        self.Ny                     = Ny
        self.y_end                  = y_end
        self.rho                    = rho
        self.pressure_boundary      = pressure_boundary
        self.pressure_difference    = np.ones(self.Ny) * pressure_difference
        self.pressure_difference    = np.append(self.pressure_difference, pressure_boundary[1])
        self.pressure_difference    = np.append(pressure_boundary[0], self.pressure_difference)
        self.pressure_difference    /= self.rho
        self.boundary_condition     = boundary_condition
        self.mu_0                   = mu_0
        self.theta                  = angle

        self.dy     = y_end / Ny
        self.y      = np.linspace((-y_end - self.dy) / 2, (y_end + self.dy) / 2, self.Ny + 2)
        self.y_wall = -np.abs(self.y) + self.y_end / 2
        self.nu_0 = self.mu_0 / self.rho

        return

    
    def add_particles(self, rho_particle, particle_D, volume_fraction):
        """
        Include particle properties
        """
        self.rho_particle = rho_particle
        self.particle_D = particle_D
        self.volume_fraction = volume_fraction
        self.particle_mass = 3/4 * np.pi * (self.particle_D/2)**3 * self.rho_particle

        return


    def set_pressure_difference(self, pressure_difference, boundaries =None):
        """
        build right hand side for master matrix equation
        """
        if boundaries is None:
            boundaries = self.pressure_boundary
        
        self.pressure_difference    = np.ones(self.Ny) * pressure_difference
        self.pressure_difference    = np.append(self.pressure_difference, boundaries[1])
        self.pressure_difference    = np.append(boundaries[0], self.pressure_difference)
        return


    def calc_mu(self, mu_0):
        return mu_0 * np.ones(self.Ny + 2)


    def diagonal_A(self, mu):
        """
        Build central diagonal for master matrix
        """
        diag_A = np.ones(len(mu))
        diag_A[1:-1] = mu[2:] + 2 * mu[1:-1] + mu[:-2]
        diag_A[0] = self.boundary_condition[0] * self.dy
        diag_A[-1] = self.boundary_condition[1] * self.dy
        return diag_A


    def diagonal_B(self, mu):
        """
        Build diagonal below central for master matrix
        """
        diag_B = np.zeros(len(mu)-1)
        diag_B = -1*(mu[:-1] + mu[1:] )
        diag_B[-1] = 1 * self.dy
        return diag_B


    def diagonal_C(self, mu):
        """
        Build diagonal above central for master matrix
        """
        diag_C = np.zeros(len(mu)-1)
        diag_C = -1*(mu[:-1] + mu[1:] )
        diag_C[0] = 1 * self.dy
        return diag_C


    def simulate(self, mu):
        """
        Determine velocity profile by building matrix and solving it
        """
        # Calculating the diagonals
        diag_A = self.diagonal_A(mu) / self.dy
        diag_B = self.diagonal_B(mu) / self.dy
        diag_C = self.diagonal_C(mu) / self.dy

        # Filling the diagonals into the matrix for comparison
        matrix_A = 1/(self.rho) * sp.diags(diagonals=(diag_A, diag_B, diag_C), offsets=(0,-1,1))
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Calculating the pressure difference
        pressure_difference = self.pressure_difference
        
        # Solving the matrix equations
        solution = la.spsolve(matrix_A, pressure_difference)
        # solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

        return solution


    def simulate_wallfunctions(self, mu, C, tau_w):
        """
        Determine velocity profile by building matrix and solving it, but for wall functions specifically
        """
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
        matrix_A = sp.dia_matrix(matrix_A).tocsr()
        
        # Solving the matrix equations
        solution = la.spsolve(matrix_A, pressure_difference)
        #solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

        return solution


    def simulate_with_particles(self, mu, alpha):
        """
        Determine velocity profile by building matrix and solving it, but with particles in the system
        should be solved with a good initial guess
        """

        mu = mu * alpha
        
        # Calculating the diagonals
        diag_A = self.diagonal_A(mu) / self.dy
        diag_B = self.diagonal_B(mu) / self.dy
        diag_C = self.diagonal_C(mu) / self.dy

        # Filling the diagonals into the matrix for comparison
        matrix_A = 1/(self.rho) * sp.diags(diagonals=(diag_A, diag_B, diag_C), offsets=(0,-1,1))
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Calculating the pressure difference
        pressure_difference = self.pressure_difference
        
        # Solving the matrix equations
        solution = la.spsolve(matrix_A, pressure_difference)
        # solution = TDMAsolver(diag_B, diag_A, diag_C, pressure_difference)

        return solution


    def mixing_length_func(self, velocity, argument_type):
        """
        Determine the Prandtl mixing length of the model
        """

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

        # Close to the boundary, add Von Driest damping
        mask = self.y_wall/delta99 < labda/kappa
        lm[mask] = kappa * self.y_wall[mask]
        # Apply damping
        if argument_type != None:
            damping = (1 - np.exp(argument))
            lm*=damping

        return lm


    def calc_mu_Prandtl(self, velocity, argument_type):
        """
        Determine effective viscosity including prandtl mixing length modelling
        """
        # Returns array of mu over the whole space
        mixing_length = self.mixing_length_func(velocity, argument_type)
        
        # Determine the velocity gradient
        velocity_diff = (velocity[2:] - velocity[:-2]) / (self.y[2:] - self.y[:-2])
        velocity_diff = np.append(velocity_diff, velocity_diff[-1])
        velocity_diff = np.append(velocity_diff[0], velocity_diff)

        # Calculate the effective viscosity
        mu_eff = self.rho * mixing_length**2 * np.abs(velocity_diff)

        return self.mu_0 * np.ones(self.Ny + 2) + mu_eff


    def calc_alpha(self, velocity, mu):
        """
        Determine particle fraction in each control volume specifically
        """

        # le constants
        g = 9.81
        sigmad = 1
        alpha2 = np.zeros(self.Ny + 2)
        velocity_diff = np.abs(velocity[1:] - velocity[:-1]) / self.dy

        # nu
        nu_turbulent = (mu - self.mu_0) / self.rho
        mu_edges = (mu[1:] + mu[:-1]) / 2
        
        # Characteristic fluid time length
        T_fluid = self.y_end / np.mean(velocity)
        # Particle relaxation time
        T_part = self.rho_particle * self.particle_D**2 / (18 * mu_edges)

        # Define gamma
        particle_stokes = T_part * velocity_diff
        gamma = 1 / (1 + particle_stokes)
        gamma_center = (gamma[:-1] + gamma[1:]) / 2

        exponent_array = np.zeros(self.Ny+2)
        # Calculation first values
        exponent_array[0] = 0
        alpha2[int(len(alpha2)/2)] = 0.2
            
        nom2 = (self.rho - self.rho_particle) * g * np.sin(self.theta)
        nom3 = -self.rho_particle * (gamma_center[2] * (nu_turbulent[2]) * velocity_diff[2]) / (2 * self.dy)  

        denom1 = self.rho_particle * (gamma_center[0] / 2 * nu_turbulent[1] * np.abs(velocity[2] - velocity[1]) / self.dy)
        denom2 = 18 * self.nu_0 / self.particle_D**2 * self.rho * nu_turbulent[1] / sigmad

        exponent_array[1] = (nom2 + nom3 / (denom1 + denom2)) * self.dy

        # solve alpha iteratively
        for i in range(2, self.Ny - 1):

            nom2 = (self.rho_particle - self.rho) * g * np.sin(self.theta)
            nom3 = -self.rho_particle * (gamma_center[i] * (nu_turbulent[i+1]) * velocity_diff[i+1] -
                                         (gamma_center[i-2]) / 2 *nu_turbulent[i-1] * velocity_diff[i-1]) / (2 * self.dy)  

            denom1 = self.rho_particle * ((gamma[i-1] + gamma[i]) / 2 * nu_turbulent[i] * np.abs(velocity[i+1] - velocity[i]) / self.dy)
            denom2 = 18 * self.nu_0 / self.particle_D**2 * self.rho * nu_turbulent[i] / sigmad

            exponent_array[i] = (nom2 + nom3 / (denom1 + denom2)) * self.dy

        # Calculation last value
        i = self.Ny

        nom2 = (self.rho_particle - self.rho) * g * np.sin(self.theta)
        nom3 = -self.rho_particle * ( -(gamma_center[i-2]) / 2 *nu_turbulent[i-1] * velocity_diff[i-1]) / (2 * self.dy)  

        denom1 = self.rho_particle * ((gamma[i-1] + gamma[i]) / 2 * nu_turbulent[i] * np.abs(velocity[i+1] - velocity[i]) / self.dy)
        denom2 = 18 * self.nu_0 / self.particle_D**2 * self.rho * nu_turbulent[i] / sigmad

        exponent_array[-1] = (nom2 + nom3 / (denom1 + denom2)) * self.dy
        
        # Calculate alpha2 from exponents
        for i in range(int(len(exponent_array)/2), len(exponent_array)-1):
            alpha2[i+1] = alpha2[i] * np.exp(exponent_array[i])

        for i in range(1, int(len(exponent_array)/2)+1)[::-1]:
            alpha2[i-1] = alpha2[i] / np.exp(-exponent_array[i])

        
        # no particles on impossible wall
        alpha2[self.y_wall < self.particle_D] = 0
        
        # scale alpha2 to volume fraction
        factor = 1 / self.y_end * self.dy * np.trapz(alpha2) / self.volume_fraction
        alpha2 /= factor

        # make sure not biggger than 1
        if np.max(alpha2) > 1:
            alpha2 /= np.max(alpha2) / 0.9

        return alpha2


    def solve_particle_velocity(self, alpha_particles, velocity_plasma, mu):
        """
        Determine particle velocity profile using one-way coupling
        """
        # Define constants
        g = 9.81

        # Determine velocity gradient and alpha gradient
        velocity_diff = (velocity_plasma[1:] - velocity_plasma[:-1]) / self.dy
        alpha_edges = (alpha_particles[1:] + alpha_particles[:-1])/2

        # Find mu at the edges
        mu_edges = (mu[1:] + mu[:-1]) / 2

        # Define Gamma
        T_part = self.rho_particle * self.particle_D**2 / (18 * mu_edges)
        particle_stokes = T_part * velocity_diff
        gamma_edges = 1 / (1 + particle_stokes)

        # Define nu_t and nu
        nu_turbulent = (mu - self.mu_0) / self.rho
        nu_t = nu_turbulent - self.nu_0
        nu_t_edge = (nu_t[:-1] + nu_t[1:]) / 2

        reynolds_particle = np.zeros(self.Ny)
        f_particle = 1 + 0.15 * reynolds_particle**0.687

        error = 1
        epsilon = 1e-6
        velocity_particles_old = np.copy(velocity_plasma[1:-1])

        i = 0
        while error > epsilon:
            velocity_particles_new = velocity_plasma[1:-1] - (1 / (18 * alpha_particles[1:-1] * f_particle * self.nu_0 / self.particle_D**2)) * \
                                 (self.rho_particle * alpha_edges[:-1] * gamma_edges[:-1] * nu_t_edge[:-1] * velocity_diff[:-1]
                                 - self.rho_particle * alpha_edges[1:] * gamma_edges[1:] * nu_t_edge[1:] * velocity_diff[1:]
                                 + alpha_particles[1:-1] * (self.rho_particle - self.rho) * g * np.cos(self.theta))

            velocity_particles_new[alpha_particles[1:-1] == 0] = 0

            reynolds_particle = np.abs(velocity_particles_new - velocity_plasma[1:-1]) * self.particle_D / self.nu_0
            f_particle = 1 + 0.15 * reynolds_particle**(0.678)

            error = np.sum(np.abs(velocity_particles_new - velocity_particles_old)) / np.sum(velocity_particles_old)
            velocity_particles_old = np.copy(velocity_particles_new)

            i+=1
            print("iteration", i, "with error", error, end='\r')

            if i > 2000:
                print("Oh nyo, it's bwoken")
                break

        return velocity_particles_new



def TDMAsolver(a, b, c, d):
    """
    Uses TriDiagonal Matrix Algorithm to solve the matrix equation
    """
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

###############################################################################
# AxiSymmetric Try :(                                                         #
###############################################################################

class MultiFlowAxis:

    """
    class containing all function to solve a 3D axiSymmetrix multiphase flow system using 
    one-way coupling and a eulerian approach (euler-euler)
    """

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
        """
        Set matrix boundary conditions for z and r
        """

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
        """
        Turns 2D matrix into 1D vector for matrix equation
        """
        return np.reshape(np.copy(matrix), newshape=(-1),  order="C")


###############################################################################
# Momentum equations in z-direction                                           #
###############################################################################

    def calc_diagonals_z(self, vel_old_z, vel_old_r, nu):
        """
        Determine diagonals for z-momentum equation
        """
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


    def calc_pressure_gradient_z(self, pressure, vel_old_z, vel_old_r):
        """
        Determine right-hand side for z-momentum equation
        """
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
        """
        Determine diagonals for r-momentum equation
        """
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


    def calc_pressure_gradient_r(self, pressure, vel_old_z, vel_old_r):
        """
        Determine right-hand side for r-momentum equation
        """

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
        """
        Solve z-momentum equation
        """

        diags = self.calc_diagonals_z(velocity_z, velocity_r, mu / self.rho_fluid)

        # Diagonals for matrix
        diag_i_j =       diags[0] + self.boundary_i_j_z
        diag_iplus_j =  (diags[1] + self.boundary_iplus_j)[:-1]
        diag_imin_j =   (diags[2] + self.boundary_imin_j)[1:]
        diag_i_jplus =  (diags[3] + self.boundary_i_jplus)[:-(self.Nz+2)]
        diag_i_jmin =   (diags[4] + self.boundary_i_jmin)[self.Nz+2:]

        # Building matrix
        matrix_A = sp.diags(
            diagonals=(diag_i_j, diag_imin_j, diag_iplus_j, diag_i_jmin, diag_i_jplus),
            offsets=(0,-1,1,-(self.Nz+2), self.Nz+2)
        )
        matrix_A = sp.dia_matrix(matrix_A).tocsr()

        # Right hand side
        pressure_gradient_z = self.calc_pressure_gradient_z(pressure, velocity_z, velocity_r)
        

        # Solve the matrix equation
        new_velocity_z = la.spsolve(matrix_A, pressure_gradient_z)
        new_velocity_z = np.reshape(new_velocity_z, (self.Nr+2, self.Nz+2))
        new_velocity_z = np.flipud(new_velocity_z)

        return new_velocity_z


    def solve_velocity_r(self, velocity_z, velocity_r, mu, pressure):
        """
        Solving r-momentum equation
        """
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
        """
        Crying tears of blood (Solving pressure correction equation but oh nyo, it's bwoken)
        As per SIMPLE algorithm
        """
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

        """
        finding the corrections of velocity as per SIMPLE algorithm
        """
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


###############################################################################
# 1D blood flow (Should be run partially due to differnt methodologies)       #
###############################################################################

# Defining constants
rho_plasma = 1003  # kg/m3
mu_plasma = 1e-3  # kg/ms

rho_RBC = 1096  # kg/m3
diameter_RBC = 8e-6  # m
volume_fraction_RBC = 0.2

y_end = 5e-3  # m
Ny = 1000
angle = 0

boundary_condition = [1, 1] 
blood_flow_rate = 20e-3 # m/s
Q_des = 185*1e-6 #m3/s total bloodflow
pressure_difference = 1e-3
pressure_boundary = np.array([1, 1])*1e-3
system = Multiflow(Ny, y_end, rho_plasma, pressure_difference,
                      pressure_boundary, boundary_condition, mu_plasma, angle)
mu = system.calc_mu(mu_plasma)


def reloadsystem():
    system = Multiflow(Ny, y_end, rho_plasma, pressure_difference,
                      pressure_boundary, boundary_condition, mu_plasma, angle)
    return system



# Solve Laminar with forced blood flow rate

eps = 1e-3
error = 1
system.set_pressure_difference(pressure_difference)

velocity = system.simulate(mu)
Q = np.trapz(velocity, x=system.y)
factor = 0.95
switch = [0,0]

while error > eps:
    if switch[-2] != switch[-1]:
        factor*=0.9
    if Q<Q_des:
        system.pressure_difference += system.pressure_difference * factor

        velocity = system.simulate(mu)
        Q = np.trapz(velocity, x=system.y)
        error = np.abs((Q-Q_des)/Q_des)
        switch.append(0)
    else:
        system.pressure_difference -= system.pressure_difference * factor
        velocity = system.simulate(mu)
        Q = np.trapz(velocity, x=system.y)
        error = np.abs((Q-Q_des)/Q_des)
        switch.append(1)

    if len(switch) == 300:
        print("Oh nyo, it's bwoken")
        break

velocity_laminar = velocity
mu_laminar = mu

plt.plot(velocity, system.y)


# Add Prandtl mixing length

argument_type = None

velocity_new = velocity_laminar

# Then calculate the solution with Prandtl mixing length
eps = 0.01
error = 1
i = 0

# Looping till solution converges
while error > eps:
    velocity = velocity_new

    # Calculate new effective viscosity
    mu_Prandtl = system.calc_mu_Prandtl(velocity, argument_type)

    # Calculate ensuing velocity
    velocity_new = system.simulate(mu_Prandtl)

    error = np.sum(np.abs((velocity_new - velocity) / velocity))
    i+=1
    print("iteration", i, "with error", error, end='\r')

    if i > 2000:
        print("Oh nyo, it's bwoken")
        break

velocity_turbulent = velocity_new
mu_turbulent = mu_Prandtl

plt.plot(velocity_laminar, system.y, label="laminar")
plt.plot(velocity_turbulent, system.y, label="pml")
plt.title("velocity with prandtl mixing length vs laminar")
plt.xlabel('Velocity [m/s]')
plt.ylabel('Channel height [m]')
plt.legend()
plt.show()

# Check for log region
plt.plot(system.y + system.y_end / 2, velocity_laminar)
plt.plot(system.y + system.y_end / 2, velocity_turbulent)
plt.xscale("log")


# Add wall functions and wall damping using Driest method

argument_type = "Driest"

velocity_new = velocity_laminar

# Then calculate the solution with Prandtl mixing length
eps = 0.01
error = 1
i = 0

# Looping till solution converges
while error > eps:
    velocity = velocity_new

    # Calculate new effective viscosity
    mu_Prandtl = system.calc_mu_Prandtl(velocity, argument_type)

    # Calculate ensuing velocity
    velocity_new = system.simulate(mu_Prandtl)

    error = np.sum(np.abs((velocity_new - velocity) / velocity))
    i+=1
    print("iteration", i, "with error", error, end='\r')

    if i > 2000:
        print("Oh nyo, it's bwoken")
        break

velocity_Driest = velocity_new
mu_Driest = mu_Prandtl

plt.plot(velocity_laminar, system.y, label="laminar")
plt.plot(velocity_Driest, system.y, label="driest")
plt.plot(velocity_turbulent, system.y, label="pml")

plt.title("velocity with prandtl mixing length vs laminar")
plt.xlabel('Velocity [m/s]')
plt.ylabel('Channel height [m]')
plt.legend()
plt.show()

# Check for log region
plt.plot(system.y + system.y_end / 2, velocity_laminar)
plt.plot(system.y + system.y_end / 2, velocity_Driest)
plt.xscale("log")


# Wall Function Constants 
velocity_new = velocity_turbulent
E = 9.8
kappa = 0.41
ks = 10 * system.nu_0

i = 0
error = 1
epsilon = 1e-5
while error > epsilon:
    velocity = velocity_new

    # Calculate the new y_plus
    tau_wall = velocity[1] / system.y_wall[1]
    u_tau = np.sqrt(tau_wall / system.rho)
    nu_0 = (mu_turbulent / system.rho)
    y_plus = system.y_wall * u_tau / nu_0

    # Calculate new wall roughness
    ks_plus = u_tau * ks / nu_0[1]
    wall_roughness = 32.6 / ks_plus

    
    # Add new nu for at wall
    nu_plus = np.zeros(system.Ny+2)
    mask = (y_plus < 11.25) * (y_plus > 0)
    nu_plus[mask] = nu_0[mask] * (y_plus[mask] * kappa / np.log(wall_roughness * y_plus[mask]) - 1)
    nu = nu_0 + nu_plus
    mu = nu * system.rho

    # Calculate new velocity
    velocity_new = system.simulate(mu)

    # Error difference in old and new velocity
    error = np.sum(np.abs((velocity_new - velocity) / velocity))
    i+=1
    print("iteration", i, "with error", error, end='\r')

    if i > 200:
        print("Oh nyo, it's bwoken")
        break

print("iteration", i, "with error", error, end='\n')
plt.plot(velocity_new, system.y, label="wall func")
plt.plot(velocity_Driest, system.y, label="Driest")
plt.legend()
plt.show()



# Solve Multiphase

# Add particles to the system
system = reloadsystem()
system.add_particles(rho_RBC, diameter_RBC, volume_fraction_RBC)

# calculate initial alpha of the particles
alpha_particles = system.calc_alpha(velocity_turbulent, mu_turbulent)

# Calculate volumetric plasma concentration
alpha_fluid = 1 - alpha_particles
velocity_new = velocity_turbulent

# While loop to iterate of velocity
i = 0
error = 1
epsilon = 1e-3
while error > epsilon:
    velocity = velocity_new

    # Calculate new velocity
    velocity_new = system.simulate_with_particles(mu_turbulent, alpha_fluid)

    # Error difference in old and new velocity
    error = np.sum(np.abs((velocity_new - velocity) / velocity))
    alpha_particles = system.calc_alpha(velocity_turbulent, mu_turbulent)

    # Calculate volumetric plasma concentration
    alpha_fluid = 1 - alpha_particles
    i+=1

    if i > 200:
        print("Oh nyo, it's bwoken")
        break


velocity_plasma = velocity_new
velocity_plasma = system.simulate_with_particles(mu_turbulent, 1-alpha_particles)

velocity_particle = system.solve_particle_velocity(alpha_particles, velocity_plasma, mu_turbulent)

plt.plot(velocity_particle, system.y[1:-1],label="Particle")
plt.plot(velocity_plasma[1:-1], system.y[1:-1], label="Plasma")
plt.legend()
plt.show()

plt.plot(alpha_particles[alpha_particles != 0], system.y[alpha_particles != 0])
plt.show()



###############################################################################
# AxiSymmetric Try :( but the actual call to run code                         #
###############################################################################


def reload_system_axis():
    system_axis = MultiFlowAxis(Nz, Nr, z_end, r_end, rho_fluid, mu_fluid, inlet)
    system_axis.make_boundary_conditions()
    return system_axis

rho_fluid = 1000
z_end = 10
Nz = 50
r_end = 1
Nr = 30
mu_fluid = 1e-3 * np.ones((Nr+2, Nz+2))
pressure = np.zeros((Nr+2, Nz+2))

# 1D solution used for inlet
rho_plasma = 1003 #  kg/m3
mu_plasma = 1e-3 # kg/ms

rho_RBC = 1096 # kg/m3
diameter_RBC = 8e-6 # m
volume_fraction_RBC = 0.2

angle = 0 #np.pi / 2

boundary_condition = [-1, 1] 
blood_flow_rate = 20e-3 # m/s
Q_des = 185*1e-6 #m3/s total bloodflow
pressure_difference = 1e-2
pressure_boundary = np.array([1, 1])*1e-3
system = reloadsystem()

inlet = system.simulate(system.mu_0)

system_axis = reload_system_axis()



# Try to use SIMPLE (liars) algorithm

system_axis = reload_system_axis()
# SIMPLE ALGORITHM
relaxation_factor = 0.2
epsilon = 1e-3
error = 1
i=0

velocity_z = np.repeat([system_axis.inlet],system_axis.Nz + 2, axis=0) # at z-offset points
velocity_z = np.swapaxes(velocity_z, 0, 1)
velocity_r = np.zeros((system_axis.Nr+2, system_axis.Nz+2)) # at r-offset points

plt.figure(figsize=(6,3.5))
plt.pcolor(system_axis.Z, system_axis.R, velocity_z)#, vmin=-6, vmax=6)
plt.colorbar(label="velocity")
plt.xlabel("z-direction [mm]", fontsize=14)
plt.ylabel("r-direction [mm]", fontsize=14)
plt.show()

pressure = np.zeros((system_axis.Nr+2, system_axis.Nz+2)) + (system_axis.z_end - system_axis.Z) * pressure_difference


while error > epsilon:
    i+=1
    # 1. initial guess for velocities, pressure
    vel_old_z = velocity_z
    vel_old_r = velocity_r
    p_old = pressure

    # 2. solve the momentum equations using guesses
    velocity_z_star = system_axis.solve_velocity_z(vel_old_z, vel_old_r, mu_fluid, pressure)
    velocity_r_star = system_axis.solve_velocity_r(vel_old_z, vel_old_r, mu_fluid, pressure)

    # 3. solve pressure using velocity from momentum eq
    pressure_corrector = system_axis.pressure_correction(velocity_z_star, velocity_r_star, mu_fluid)
    
    # 4. correct pressure and velocities
    pressure = p_old + pressure_corrector * relaxation_factor
    velocity_z, velocity_r = system_axis.velocity_corrections(
        velocity_z, velocity_r, mu_fluid, pressure_corrector, relaxation_factor, velocity_z_star, velocity_r_star
    )

    #5. calculate if velocity has converged
    error = np.sum(np.abs((velocity_z - vel_old_z) / vel_old_z))
    print("iteration", i, "with error", error, end='\n')

    if i > 2000:
        print("Oh nyo, it's bwoken")
        break

