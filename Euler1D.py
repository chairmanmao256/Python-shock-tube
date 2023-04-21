'''
### Description

This implementation of 1D Euler equation solver is a translation of the MATLAB and C++
code written by the author in Junior and Senior. Hopefully more Riemann solvers and more
flux limiters will be added.

Wcy, 2022.11
'''
gamma = 1.4
zz = (gamma - 1.0) / (2.0 * gamma)
import numpy as np
import RiemannExact as RE

class Euler1D:
    def __init__(self, fname = "settings.txt"):
        '''
        ### Description

        We initialize the Euler 1d solver by a file. All the settings can be found in the code
        listed below. Comments about them will be added here later.
        '''

        file = open(fname, mode = "r")
        key_dict = {
                    # Problem definition
                    "LEFTSTATE": 1, 
                    "RIGHTSTATE": 2, 
                    "INTERFACE": 3,

                    # Numerical Schemes
                    "NCELLS": 4,
                    "RIEMANNSOLVER": 5,
                    "FLUXLIMITTER": 6,
                    "RCONSTRUCTION": 7}
        lines = file.readlines()
        iL = 0
        found_key = 0
        reconstruction_list = [0, 1]
        flux_limiter_list = [0, 1]
        Riemann_solver_list = [0, 1, 2, 3, 4]
        while iL<len(lines):
            line = lines[iL].split()
            if len(line) < 1:
                iL += 1
                continue
            
            if found_key == 0:
                if line[0] in key_dict:
                    found_key = key_dict[line[0]]
            elif found_key == 1:
                rho1, u1, p1 = lines[iL].split()
                self.rho1, self.u1, self.p1 = float(rho1), float(u1), float(p1)
                # print(f"Left side state:\nrho: {self.rho1}, u: {self.u1}, p: {self.p1}\n")
                found_key = 0
            elif found_key == 2:
                rho2, u2, p2 = lines[iL].split()
                self.rho2, self.u2, self.p2 = float(rho2), float(u2), float(p2)
                # print(f"Right side state:\nrho: {self.rho2}, u: {self.u2}, p: {self.p2}\n")
                found_key = 0
            elif found_key == 3:
                self.interface = float(lines[iL])
                # print(f"The x-coordinate of the interface is: \n{self.interface}\n")
                found_key = 0
            elif found_key == 4:
                self.ncells = int(lines[iL])
                # print(f"The number of cells is \n{self.ncells}\n")
                found_key = 0
            elif found_key == 5:
                self.RSopt = int(lines[iL])
                if self.RSopt == 0:
                    print("============ Selected RS: Exact ===========\n")
                elif self.RSopt == 1:
                    print("============ Selected RS: Roe =============\n")
                elif self.RSopt == 2:
                    print("============ Selected RS: AUSM ============\n")
                elif self.RSopt == 3:
                    print("============ Selected RS: HLLC ============\n")
                elif self.RSopt == 4:
                    print("============ Selected RS: HLL =============\n")
                if self.RSopt not in Riemann_solver_list:
                    raise Exception("The available Riemann solvers are:\n\
                                     0: Exact\n\
                                     1: Roe\n\
                                     2: AUSM\n\
                                     3: HLLC\n\
                                     4: HLL")
                found_key = 0
            elif found_key == 6:
                self.FLopt = int(lines[iL])
                if self.FLopt not in flux_limiter_list:
                    raise Exception("The available flux limiters are:\n\
                                     0: minmod\n\
                                     1: Van Leer")
                    raise Exception("Other Flux limitter options not supported yet!")
                found_key = 0
            elif found_key == 7:
                self.reconstruction = int(lines[iL])
                if self.reconstruction not in reconstruction_list:
                    raise Exception("The available reconstruction methods are:\n\
                                     0: Zero order\n\
                                     1: 2nd order TVD")

            iL += 1
        
        # store the solution on the cell centers
        self.sol = np.zeros((3, self.ncells), dtype = np.float64)
        self.sol_old = np.zeros((3, self.ncells), dtype = np.float64)

        # store the reconstructed values on the cell interface
        self.rec_l = np.zeros((3, self.ncells+1), dtype = np.float64)
        self.rec_r = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the limitted slope for centered TVD-MUSCL reconstruction
        self.D = np.zeros_like(self.sol)

        # store the difference between every cell interface
        self.del_U = np.zeros_like(self.rec_l)

        # store the averaged value on the interface 
        self.avg = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the flux on the cell interface
        self.flux = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the virtual cells to impose boundary conditions
        self.vLL = np.zeros((3,1), dtype = np.float64)
        self.vL = np.zeros((3,1), dtype = np.float64)
        self.vRR = np.zeros((3,1), dtype = np.float64)
        self.vR = np.zeros((3,1), dtype = np.float64)
    
        # calculate the mesh information
        self.dx = 1.0 / float(self.ncells)
        self.mesh = np.linspace(-0.0 + 0.5*self.dx, 1.0 - 0.5*self.dx, self.ncells)

        # set the initial condition
        for i in range(self.ncells):
            if self.mesh[i] <= self.interface:
                self.sol[0][i] = self.rho1
                self.sol[1][i] = self.u1
                self.sol[2][i] = self.p1
            else:
                self.sol[0][i] = self.rho2
                self.sol[1][i] = self.u2
                self.sol[2][i] = self.p2
        self.set_boundary()
        self.prim2Con(0)

    def set_boundary(self):
        self.vL = self.sol[:, 0].copy().reshape(-1,1)
        self.vLL = self.sol[:, 0].copy().reshape(-1,1)
        self.vR = self.sol[:, -1].copy().reshape(-1,1)
        self.vRR = self.sol[:, -1].copy().reshape(-1,1)

    def con2Prim(self, mode = 0):
        '''
        ### Descripition:

        Convert the conservative variables to primitive variables. 

        ### Input:

        `mode`: if `mode == 0`, then convert the cell center values (inlcuding virtual cells)
        elsewise, the reconstructed value will be converted
        '''
        if mode == 0:
            temp1 = self.sol[0, :].copy()
            temp2 = self.sol[1, :] / self.sol[0, :]
            temp3 = (1.4-1.0)*self.sol[2, :]-(1.4-1.0)/2.0*self.sol[1, :]*self.sol[1, :]/self.sol[0, :]

            self.sol[0, :] = temp1.copy()
            self.sol[1, :] = temp2.copy()
            self.sol[2, :] = temp3.copy()

            bc_list = [self.vLL, self.vL, self.vR, self.vRR]
            for bc in bc_list:
                temp1 = bc[0][0]
                temp2 = bc[1][0] / bc[0][0]
                temp3 = (1.4-1.0)*bc[2][0] - (1.4-1.0) / 2.0 * bc[1][0] * bc[1][0] / bc[0][0]
                bc[0][0], bc[1][0], bc[2][0] = temp1, temp2, temp3

        if mode == 1:
            temp1 = self.rec_l[0, :].copy()
            temp2 = self.rec_l[1, :] / self.rec_l[0, :]
            temp3 = (1.4-1.0)*self.rec_l[2, :]-(1.4-1.0)/2.0*self.rec_l[1, :]*self.rec_l[1, :]/self.rec_l[0, :]

            self.rec_l[0, :] = temp1.copy()
            self.rec_l[1, :] = temp2.copy()
            self.rec_l[2, :] = temp3.copy()

            temp1 = self.rec_r[0, :]
            temp2 = self.rec_r[1, :] / self.rec_r[0, :]
            temp3 = (1.4-1.0)*self.rec_r[2, :]-(1.4-1.0)/2.0*self.rec_r[1, :]*self.rec_r[1, :]/self.rec_r[0, :]

            self.rec_r[0, :] = temp1.copy()
            self.rec_r[1, :] = temp2.copy()
            self.rec_r[2, :] = temp3.copy()

    def prim2Con(self, mode = 0):

        if mode == 0:
            temp1 = self.sol[0, :].copy()
            temp2 = self.sol[0, :] * self.sol[1, :]
            temp3 = self.sol[0, :] * (0.5 * self.sol[1, :] * self.sol[1, :]) + 1.0 / (1.4 - 1.0) * self.sol[2, :]

            self.sol[0, :] = temp1.copy()
            self.sol[1, :] = temp2.copy()
            self.sol[2, :] = temp3.copy()

            bc_list = [self.vLL, self.vL, self.vR, self.vRR]
            for bc in bc_list:
                temp1 = bc[0][0]
                temp2 = bc[1][0] * bc[0][0]
                temp3 = bc[0][0] * (0.5 * bc[1][0] * bc[1][0]) + 1.0 / (1.4 - 1.0) * bc[2][0]
                bc[0][0], bc[1][0], bc[2][0] = temp1, temp2, temp3

        if mode == 1:
            temp1 = self.rec_l[0, :].copy()
            temp2 = self.rec_l[1, :] * self.rec_l[0, :]
            temp3 = self.rec_l[0, :] * (0.5 * self.rec_l[1, :] * self.rec_l[1, :]) + 1.0 / (1.4 - 1.0) * self.rec_l[2, :]

            self.rec_l[0, :] = temp1.copy()
            self.rec_l[1, :] = temp2.copy()
            self.rec_l[2, :] = temp3.copy()

            temp1 = self.rec_r[0, :].copy()
            temp2 = self.rec_r[1, :] * self.rec_r[0, :]
            temp3 = self.rec_r[0, :] * (0.5 * self.rec_r[1, :] * self.rec_r[1, :]) + 1.0 / (1.4 - 1.0) * self.rec_r[2, :]

            self.rec_r[0, :] = temp1.copy()
            self.rec_r[1, :] = temp2.copy()
            self.rec_r[2, :] = temp3.copy()

    def avg_roe(self):
        '''
        ### Description

        This function gives the roe-averaged value on the interface
        '''
        self.avg[0, :] = np.sqrt(self.rec_l[0, :] * self.rec_r[0, :]).copy()
        self.avg[1, :] = ((np.sqrt(self.rec_l[0, :]) * self.rec_l[1, :] + np.sqrt(self.rec_r[0, :]) * self.rec_r[1, :]) / \
                         (np.sqrt(self.rec_l[0, :])+np.sqrt(self.rec_r[0, :]))).copy()
        HL = (1.4 / (1.4-1.0) * self.rec_l[2, :] / self.rec_l[0, :]).copy()
        HR = (1.4 / (1.4-1.0) * self.rec_r[2, :] / self.rec_r[0, :]).copy()
        Hav = (((HR * np.sqrt(self.rec_r[0, :]) + HL * np.sqrt(self.rec_l[0, :]))) / (np.sqrt(self.rec_l[0, :])+np.sqrt(self.rec_r[0, :]))).copy()
        a2 = ((1.4-1.0)*(Hav - 0.5*self.avg[1, :]**2)).copy()
        self.avg[2, :] = (self.avg[0, :] * a2 / 1.4).copy()

    def reconstruction_0(self):
        '''
        ### Description:

        Conduct the reconstruction of order 0
        '''
        ncells = self.ncells
        self.rec_l[:, 0] = self.vL[:, 0].copy()
        self.rec_r[:, ncells] = self.vR[:, 0].copy()

        self.rec_l[:, 1:] = self.sol[:, :].copy()
        self.rec_r[:, 0:-1] = self.sol[:, :].copy()

        if np.min(self.rec_l[0, :]) <= 0 or np.min(self.rec_r[0, :]) <= 0:
            print("bounding rho...")
        
        for i in range(ncells+1):
            if self.rec_l[0, i] <= 0:
                self.rec_l[0,i] = 1e-3
            if self.rec_r[0, i] <= 0:
                self.rec_r[0,i] = 1e-3

    def reconstruction_MUSCL_TVD(self, omega = -1.0, omega_bar = 1.0):
        '''
        ### Description

        Use MUSCL reconstruction technique presented in the course materials
        '''
        if omega_bar < 1.0 or omega_bar > (3 - omega) / (1 - omega):
            raise Exception("Omega_bar should be larger than 1 and smaller than (3-omega)/(1-omega)!")
        
        L = np.hstack((self.vL, self.sol[:, :]))
        LL = np.hstack((self.vLL, self.vL, self.sol[:, 0:-1]))
        R = np.hstack((self.sol[:, :], self.vR))
        RR = np.hstack((self.sol[:, 1:], self.vR, self.vRR))
        del1 = L - LL
        del2 = R - L
        del3 = RR - R

        del1_bb = np.zeros_like(del1)
        del2_b = np.zeros_like(del1)
        del2_bb = np.zeros_like(del1)
        del3_b = np.zeros_like(del1)

        for j in range(3):
            del1_bb[j, :] = self.minmod(del1[j, :], omega_bar * del2[j, :])
            del2_b[j, :] = self.minmod(del2[j, :], omega_bar * del1[j, :])
            del2_bb[j, :] = self.minmod(del2[j, :], omega_bar * del3[j, :])
            del3_b[j, :] = self.minmod(del3[j, :], omega_bar * del2[j, :])

        self.rec_l = L + 0.25 * ((1 - omega)*del1_bb + (1 + omega)*del2_b)
        self.rec_r = R - 0.25 * ((1 - omega)*del3_b + (1 + omega)*del2_bb)

        if np.min(self.rec_l[0, :]) <= 0 or np.min(self.rec_r[0, :]) <= 0:
            raise Exception("negative density encountered!")

    def reconstruction_TVD(self, flag = "simple-reconstruct"):
        '''
        ### Description:

        Conduct the TVD reconstruction. Flux limiter is used and the spatial order is 2.
        '''
        ncells = self.ncells
        dx = self.dx
        # the left eigen vector matrix
        LM = np.zeros((3,3,ncells+1))
        # the right eigen vector matrix
        RM = np.zeros((3,3, ncells+1))

        Dr = np.zeros((3, ncells+1))
        Dl = np.zeros((3, ncells+1))
        temp_r = np.zeros((3, ncells+1))
        temp_l = np.zeros((3, ncells+1))

        # we now convert the cell-center values to conservative form
        # self.prim2Con(mode = 0)

        '''
        for every interface, we need the LL, L, R and RR value adjacent to 
        the interface, which is shown graphically below:

        | LL | L | R | RR |
                 ^
                  The interface we want to reconstruct 
        '''
        L = np.hstack((self.vL, self.sol[:, :]))
        LL = np.hstack((self.vLL, self.vL, self.sol[:, 0:-1]))
        R = np.hstack((self.sol[:, :], self.vR))
        RR = np.hstack((self.sol[:, 1:], self.vR, self.vRR))

        # do the following steps when the user specifies the eigen-version 
        # of reconstruction
        # calculate the left eigen vectors
        if flag == "eigen-reconstruct":
            rho = self.avg[0, :].copy()
            u = self.avg[1, :].copy()
            p = self.avg[2, :].copy()
            a = np.sqrt(1.4*p/rho)
            h = 1.4/(1.4-1.0)*p/rho
            
            LM[0, 0, :] = (-u**3 + a*u**2 + 2.0*h*u) / (2.0*(-a*u**2+2.0*a*h))
            LM[0, 1, :] = -(- u*u + 2.0*a*u + 2.0*h)/(2.0*(- a*u**2 + 2.0*a*h))
            LM[0, 2, :] = 1.0/(- u**2 + 2.0*h)
            LM[1, 0, :] = (2.0*(- u**2 + h))/(- u**2 + 2.0*h)
            LM[1, 1, :] = (2.0*u)/(- u**2 + 2.0*h)
            LM[1, 2, :] = -2.0/(- u**2 + 2.0*h)
            LM[2, 0, :] = (u**3 + a*u**2 - 2.0*h*u)/(2.0*(- a*u**2 + 2.0*a*h))
            LM[2, 1, :] = -(u**2 + 2.0*a*u - 2.0*h)/(2.0*(- a*u**2 + 2.0*a*h))
            LM[2, 2, :] = 1.0/(- u**2 + 2.0*h)

            RM[0, 0, :] = 1.0
            RM[0, 1, :] = 1.0
            RM[0, 2, :] = 1.0
            RM[1, 0, :] = u-a
            RM[1, 1, :] = u
            RM[1, 2, :] = u+a
            RM[2, 0, :] = h-u*a
            RM[2, 1, :] = 0.5*u*u
            RM[2, 2, :] = h+u*a

            for j in range(3):
                rdiff = (RR[0, :] - R[0, :]) * LM[j, 0, :] + (RR[1, :] - R[1, :]) * LM[j, 1, :] + \
                        (RR[2, :] - R[2, :]) * LM[j, 2, :]
                ldiff = (R[0, :] - L[0, :]) * LM[j, 0, :] + (R[1, :] - L[1, :]) * LM[j, 1, :] + \
                        (R[2, :] - L[2, :]) * LM[j, 2, :]
                Dr[j, :] = self.minmod(rdiff, ldiff)

                rdiff = (R[0, :] - L[0, :]) * LM[j, 0, :] + (R[1, :] - L[1, :]) * LM[j, 1, :] + \
                        (R[2, :] - L[2, :]) * LM[j, 2, :]
                ldiff = (L[0, :] - LL[0, :]) * LM[j, 0, :] + (L[1, :] - LL[1, :]) * LM[j, 1, :] + \
                        (L[2, :] - LL[2, :]) * LM[j, 2, :]
                Dl[j, :] = self.minmod(ldiff, rdiff)

                temp_l[j, :] = LM[j, 0, :] * L[0, :] + LM[j, 1, :] * L[1, :] + LM[j, 2, :] * L[2, :] + Dl[j, :] * 0.5
                temp_r[j, :] = LM[j, 0, :] * R[0, :] + LM[j, 1, :] * R[1, :] + LM[j, 2, :] * R[2, :] - Dr[j, :] * 0.5

            for j in range(3):
                self.rec_l[j, :] = RM[j, 0, :] * temp_l[0, :] + RM[j, 1, :] * temp_l[1, :] + RM[j, 2, :] * temp_l[2, :]
                self.rec_r[j, :] = RM[j, 0, :] * temp_r[0, :] + RM[j, 1, :] * temp_r[1, :] + RM[j, 2, :] * temp_r[2, :]

        # if we do not use the eigen version reconstruction...
        else:
            for j in range(3):
                rdiff = RR[j, :] - R[j, :]
                ldiff = R[j, :] - L[j, :]
                if self.FLopt == 0:
                    Dr[j, :] = self.minmod(rdiff, 1.5 * ldiff)
                elif self.FLopt == 1:
                    Dr[j, :] = self.vanalbada(rdiff, ldiff)

                rdiff = R[j, :] - L[j, :]
                ldiff = L[j, :] - LL[j, :]
                if self.FLopt == 0:
                    Dl[j, :] = self.minmod(1.5 * rdiff, ldiff)
                elif self.FLopt == 1:
                    Dl[j, :] = self.vanalbada(rdiff, ldiff)
            
            self.rec_l = L + 0.5 * Dl
            self.rec_r = R - 0.5 * Dr

        if np.min(self.rec_l[0, :]) <= 0 or np.min(self.rec_r[0, :]) <= 0:
            raise Exception("negative density encountered!")

    def reconstruction_TVD_centered(self):
        '''
        ### Description

        Reconstruct the conservative variables directly (characteristic reconstruction is not used).
        Centered formulation is used here for clarity:

        U_{j+1/2} = U_{j+1} - U_{j}
        D_{j} = B(\Delta U_{j-1/2}, \Delta U_{j+1/2})
        UL_{j+1/2} = U_{j} + 1/2 * D_{j}
        UR_{j-1/2} = U_{j} - 1/2 * D_{j}

        +-------+-------+-------+
        |       |       |       |
        |U_{j-1}| U_{j} |U_{j+1}|
        |       |       |       |
        +-------+-------+-------+

        '''
        # calculate D_j for every cell, including the virtual cell
        self.del_U = np.hstack((self.sol, self.vR)) - np.hstack((self.vL, self.sol))
        del_UL = self.vL - self.vLL
        del_UR = self.vRR - self.vR

        # choose the limiter
        match self.FLopt:
            case 0:
                self.D = self.B_minmod(self.del_U[:, 0:-1], self.del_U[:, 1:])
                DL = self.B_minmod(del_UL, self.del_U[:, 0].reshape(-1,1))
                DR = self.B_minmod(self.del_U[:, -1].reshape(-1,1), del_UR)
            case 1:
                self.D = self.B_VanLeer(self.del_U[:, 0:-1], self.del_U[:, 1:])
                DL = self.B_VanLeer(del_UL, self.del_U[:, 0].reshape(-1,1))
                DR = self.B_VanLeer(self.del_U[:, -1].reshape(-1,1), del_UR)

        # perform the reconstruction
        self.rec_l[:, 1:] = self.sol + 0.5 * self.D
        self.rec_l[:, 0] = (self.vL + 0.5 * DL).ravel()
        self.rec_r[:, 0:-1] = self.sol - 0.5 * self.D
        self.rec_r[:, -1] = (self.vR - 0.5 * DR).ravel()

    def roe_flux_alter(self):
        ncells = self.ncells
        # left state
        rL = self.rec_l[0, :].copy()
        uL = self.rec_l[1, :]/rL
        EL = self.rec_l[2, :]/rL
        pL = (1.4 - 1) * (self.rec_l[2, :] - rL * uL * uL / 2.0)
        aL = np.sqrt(1.4 * pL / rL)
        HL = (self.rec_l[2, :] + pL) / rL

        # right state
        rR = self.rec_r[0, :].copy()
        uR = self.rec_r[1, :]/rR
        ER = self.rec_r[2, :]/rR
        pR = (1.4 - 1) * (self.rec_r[2, :] - rR * uR * uR / 2.0)
        aR = np.sqrt(1.4 * pR / rR)
        HR = (self.rec_r[2, :] + pR) / rR

        # difference
        dr = rR-rL
        du = uR-uL
        dP = pR-pL

        # roe averages
        ratio = np.sqrt(rR / rL)
        r = ratio * rL
        u = (uL + ratio * uR) / (1 + ratio)
        H = (HL + ratio * HR) / (1 + ratio)
        a = np.sqrt((1.4 - 1)*(H - u * u / 2.0))

        # L*U (characteristics)
        dV = np.vstack(((dP - r*a*du) / (2*a*a), -(dP/(a*a)-dr), (dP + r*a*du) / (2*a*a)))
        
        # eigen values
        lamb = self.get_eig(u, a)

        # R * Lamb * dV
        K = np.zeros((3, ncells+1))

        for i in range(ncells+1):
            R = np.array([[1.0, 1.0, 1.0],
                          [u[i]-a[i], u[i], u[i] + a[i]],
                          [H[i] - u[i] * a[i], u[i]*u[i]/2.0, H[i] + u[i] * a[i]]])
            K[:, i] = R @ (lamb[:, i] * dV[:, i])

        FL = np.vstack((rL * uL, rL * uL * uL + pL, uL * (rL * EL + pL)))
        FR = np.vstack((rR * uR, rR * uR * uR + pR, uR * (rR * ER + pR)))

        self.flux = (FL + FR - K) / 2.0

    def HLL_family_flux(self, choice = 3):
        '''
        ### Description

        Using the HLLC flux calculation method presented in Toro's book
        '''
        ncells = self.ncells
        # left state
        rL = self.rec_l[0, :].copy()
        uL = self.rec_l[1, :]/rL
        EL = self.rec_l[2, :]/rL
        pL = (1.4 - 1) * (self.rec_l[2, :] - rL * uL * uL / 2.0)
        aL = np.sqrt(1.4 * pL / rL)

        # right state
        rR = self.rec_r[0, :].copy()
        uR = self.rec_r[1, :]/rR
        ER = self.rec_r[2, :]/rR
        pR = (1.4 - 1) * (self.rec_r[2, :] - rR * uR * uR / 2.0)
        aR = np.sqrt(1.4 * pR / rR)

        FL = np.vstack((rL* uL, rL*uL**2 + pL, (rL*EL + pL) * uL))
        FR = np.vstack((rR* uR, rR*uR**2 + pR, (rR*ER + pR) * uR))

        for i in range(ncells+1):
            ppv = max(0.0, 0.5 * (pL[i] + pR[i])+0.5*(uL[i]-uR[i])*(0.25*(rL[i]+rR[i])*(aL[i]+aR[i])))
            pmin = min(pL[i],pR[i])
            pmax = max(pL[i],pR[i])
            qmax = pmax/ pmin
            quser = 2.0

            if (qmax < quser) and (pmin<=ppv) and (ppv <= pmax):
                pM = ppv
            else:
                if ppv<pmin:
                    # use the two-expansion approximation, TRRS
                    pM = ((aL[i] + aR[i] - (gamma-1.0)/2.0*(uR[i] - uL[i])) / (aL[i]/(pL[i]**zz) + aR[i]/(pR[i]**zz))) ** (1.0 / zz)
                else:
                    # use the TSRS solution
                    gL = np.sqrt((2.0/(gamma+1)/rL[i])/((gamma-1.0)/(gamma+1.0)*pL[i] + ppv))
                    gR = np.sqrt((2.0/(gamma+1)/rR[i])/((gamma-1.0)/(gamma+1.0)*pR[i] + ppv))
                    pM = (gL * pL[i] + gR * pR[i] - (uR[i] - uL[i]))/(gL + gR)

            # estimate the wave speed based on the pM
            if pM > pL[i]:
                zL = np.sqrt(1+(gamma+1)/(2.0*gamma)*(pM/pL[i]-1.0))
            else:
                zL = 1.0

            if pM > pR[i]:
                zR = np.sqrt(1+(gamma+1)/(2.0*gamma)*(pM/pR[i]-1.0))
            else:
                zR = 1.0

            # calculate the wave speed
            SL = uL[i] - zL * aL[i]
            SR = uR[i] + zR * aR[i]
            SM = ((pR[i] - pL[i]) + rL[i] * uL[i] * (SL - uL[i]) - rR[i] * uR[i] * (SR - uR[i]))/\
                 (rL[i]*(SL - uL[i]) - rR[i]*(SR - uR[i]))
            
            # calculate the HLLC flux
            if choice == 3:
                if 0 <= SL:
                    self.flux[:, i] = FL[:, i]
                elif (SL <= 0) and (0 <= SM):
                    qsL = rL[i]*(SL-uL[i])/(SL-SM)*np.array([1.0, SM, EL[i] + (SM-uL[i])*(SM+pL[i]/(rL[i]*(SL-uL[i])))])
                    self.flux[:, i] = FL[:, i] + SL * (qsL - self.rec_l[:, i])
                elif (SM <= 0) and (0 <= SR):
                    qsR = rR[i]*(SR-uR[i])/(SR-SM)*np.array([1.0, SM, ER[i] + (SM-uR[i])*(SM+pR[i]/(rR[i]*(SR-uR[i])))])
                    self.flux[:, i] = FR[:, i] + SR * (qsR - self.rec_r[:, i])
                else:
                    self.flux[:, i] = FR[:, i]
            elif choice == 4: # calculate the HLL flux
                if 0 <= SL:
                    self.flux[:, i] = FL[:, i]
                elif (SL <= 0) and (0 <= SR):
                    self.flux[:, i] = (SR * FL[:, i] - SL * FR[:, i] + SL * SR * (self.rec_r[:, i] - self.rec_l[:, i])) / (SR - SL)
                else:
                    self.flux[:, i] = FR[:, i]

    def AUSM_flux(self):
        '''
        ### Description

        AUSM flux vector splitting scheme
        '''
        ncells = self.ncells
        # left state
        rL = self.rec_l[0, :].copy()
        uL = self.rec_l[1, :]/rL
        EL = self.rec_l[2, :]/rL
        pL = (1.4 - 1) * (self.rec_l[2, :] - rL * uL * uL / 2.0)
        aL = np.sqrt(1.4 * pL / rL)
        HL = (self.rec_l[2, :] + pL) / rL
        ML = uL / aL

        # right state
        rR = self.rec_r[0, :].copy()
        uR = self.rec_r[1, :]/rR
        ER = self.rec_r[2, :]/rR
        pR = (1.4 - 1) * (self.rec_r[2, :] - rR * uR * uR / 2.0)
        aR = np.sqrt(1.4 * pR / rR)
        HR = (self.rec_r[2, :] + pR) / rR
        MR = uR / aR

        # splitted Mach number
        Mp = np.zeros(ML.shape)
        Mn = np.zeros(ML.shape)
        # splitted pressure
        Pp = np.zeros(ML.shape)
        Pn = np.zeros(ML.shape)

        # the positive M is calculated from the left M
        for i in range(ncells + 1):
            if ML[i] <= -1:
                Mp[i] = 0.0 
                Pp[i] = 0.0
            elif ML[i] < 1:
                Mp[i] = (ML[i] + 1) * (ML[i] + 1) / 4.0
                Pp[i] = (ML[i] + 1) * (ML[i] + 1) / 4.0 * (2.0 - ML[i]) * pL[i]
            else:
                Mp[i] = ML[i]
                Pp[i] = pL[i]

        # the negative M is calculated from the right M
        for i in range(ncells + 1):
            if MR[i] <= -1:
                Mn[i] = MR[i]
                Pn[i] = pR[i]
            elif MR[i] < 1:
                Mn[i] = -(MR[i] - 1) * (MR[i] - 1) / 4.0
                Pn[i] = pR[i] * (1 - MR[i]) * (1 - MR[i]) * (2 + MR[i]) / 4.0
            else:
                Mn[i] = 0.0
                Pn[i] = 0.0

        # we will decide the flux based on the sign of Mn + Mp
        selectL = (np.sign(Mn + Mp) + 1) / 2.0 * (Mn + Mp)
        selectR = (1 - np.sign(Mn + Mp)) / 2.0 * (Mn + Mp)
        self.flux[0, :] = (selectL * rL * aL + selectR * rR * aR).copy()
        self.flux[1, :] = (selectL * rL * aL * uL + selectR * rR * aR * uR + Pn + Pp).copy()
        self.flux[2, :] = (selectL * rL * aL * HL + selectR * rR * aR * HR).copy()

    def exact_flux(self):
        '''
        ### Description:

        This function uses the exact Riemann solver `ToroExact` from
        https://github.com/tahandy/ToroExact to calculate the face flux.
        '''
        # left state
        rL = self.rec_l[0, :].copy()
        uL = self.rec_l[1, :]/rL
        EL = self.rec_l[2, :]/rL
        pL = (1.4 - 1) * (self.rec_l[2, :] - rL * uL * uL / 2.0)
        aL = np.sqrt(1.4 * pL / rL)
        HL = (self.rec_l[2, :] + pL) / rL

        # right state
        rR = self.rec_r[0, :].copy()
        uR = self.rec_r[1, :]/rR
        ER = self.rec_r[2, :]/rR
        pR = (1.4 - 1) * (self.rec_r[2, :] - rR * uR * uR / 2.0)
        aR = np.sqrt(1.4 * pR / rR)
        HR = (self.rec_r[2, :] + pR) / rR
        for i in range(self.ncells+1):
            state_l = np.array([rL[i], uL[i], pL[i]])
            state_r = np.array([rR[i], uR[i], pR[i]])
            # rp = RP.exactRP(1.4, state_l,state_r)
            re = RE.RiemannExact(state_l, state_r)
            success = re.solve()
            if not success:
                raise Exception(f"Exact Riemann solver fails at interface {i}!")
            w = re.sample(0.0)
            rho, p, u = w[0], w[2], w[1]
            rhoE = rho * u * u / 2.0 + p / (1.4 - 1)
            self.flux[0, i] = rho * u
            self.flux[1, i] = rho * u * u + p
            self.flux[2, i] = (rhoE + p) * u

    def time_advancement(self, cfl: float, oldset: int):
        '''
        ### Description:

        Perform the time advancement (order 1)
        
        '''
        r = self.sol[0, :]
        u = self.sol[1, :]/r
        E = self.sol[2, :]/r
        p = (1.4 - 1) * (self.sol[2, :] - r * u * u / 2.0)
        a = np.sqrt(1.4 * p / r)
        Smax = np.max(np.vstack((abs(u+a),abs(u-a), abs(u))))
        dt = cfl * self.dx / Smax
        if oldset > 0: # perform at the first Runge-Kutta step
            self.set_old()
        self.sol = self.sol_old - dt/self.dx * (self.flux[:, 1:] - self.flux[:, 0:-1])
        return dt

    def time_advancement_RK4(self, cfl: float) -> float:
        '''
        ### Description

        Use Runge-Kutta 4th order time advancement
        '''
        r = self.sol[0, :]
        u = self.sol[1, :]/r
        E = self.sol[2, :]/r
        p = (1.4 - 1) * (self.sol[2, :] - r * u * u / 2.0)
        a = np.sqrt(1.4 * p / r)
        Smax = np.max(np.vstack((abs(u+a),abs(u-a), abs(u))))
        dt = cfl * self.dx / Smax

        self.set_old()
        for j in range(4):
            dtt = dt / float(4 - j)
            # choose Reconstruction method
            if self.reconstruction == 1:
                # self.reconstruction_TVD()
                # self.reconstruction_MUSCL_TVD()
                self.reconstruction_TVD_centered()
            elif self.reconstruction == 0:
                self.reconstruction_0()

            # choose Riemann solver
            if self.RSopt == 0:
                self.exact_flux()
            elif self.RSopt == 1:
                self.roe_flux_alter()
            elif self.RSopt == 2:
                self.AUSM_flux()
            elif self.RSopt == 3:
                self.HLL_family_flux(choice = 3)
            elif self.RSopt == 4:
                self.HLL_family_flux(choice = 4)
            self.sol = self.sol_old - dtt/self.dx * (self.flux[:, 1:] - self.flux[:, 0:-1])
        
        return dt

    def time_advancement_RK3(self, cfl: float) -> float:
        '''
        ### Description

        Use Runge-Kutta 3rd order time advancement
        '''
        r = self.sol[0, :]
        u = self.sol[1, :]/r
        E = self.sol[2, :]/r
        p = (1.4 - 1) * (self.sol[2, :] - r * u * u / 2.0)
        a = np.sqrt(1.4 * p / r)
        Smax = np.max(np.vstack((abs(u+a),abs(u-a), abs(u))))
        dt = cfl * self.dx / Smax

        self.set_old()
        alpha1 = [1.0, 3.0/4.0, 1.0/3.0]
        alpha2 = [0.0, 1.0/4.0, 2.0/3.0]
        alpha3 = [1.0, 1.0/4.0, 2.0/3.0]

        for j in range(3):
            # choose Reconstruction method
            if self.reconstruction == 1:
                self.reconstruction_TVD_centered()
            elif self.reconstruction == 0:
                self.reconstruction_0()

            # choose Riemann solver
            if self.RSopt == 0:
                self.exact_flux()
            elif self.RSopt == 1:
                self.roe_flux_alter()
            elif self.RSopt == 2:
                self.AUSM_flux()
            elif self.RSopt == 3:
                self.HLL_family_flux(choice = 3)
            elif self.RSopt == 4:
                self.HLL_family_flux(choice = 4)
            self.sol = alpha1[j] * self.sol_old + alpha2[j] * self.sol - alpha3[j] * dt/self.dx * (self.flux[:, 1:] - self.flux[:, 0:-1])
        
        return dt

    def set_old(self):
        '''
        ### Description:

        Store the old value of sol
        '''
        self.sol_old = self.sol.copy()

    def get_eig(self, u: np.ndarray, a: np.ndarray):
        '''
        ### Description

        Calculate the eigen value of the Jacobian.
        '''
        lamb = np.zeros((3, self.ncells+1))
        eps = 0.3 * (a + abs(u))
        lamb[0, :] = (abs(u-a) - eps < 0).astype(np.int32) * ((abs(u-a))**2 + eps**2) / (2*eps) \
                    + (abs(u-a) -eps >= 0).astype(np.int32) * abs(u-a)
        lamb[1, :] = (abs(u) - eps < 0).astype(np.int32) * ((abs(u))**2 + eps**2) / (2*eps) \
                    + (abs(u) -eps >= 0).astype(np.int32) * abs(u)
        lamb[2, :] = (abs(u+a) - eps < 0).astype(np.int32) * ((abs(u+a))**2 + eps**2) / (2*eps) \
                    + (abs(u+a) -eps >= 0).astype(np.int32) * abs(u+a)

        return lamb

#  The flux limiters. Implemented in both centered form and non-centered form
    def B_minmod(self, a: np.ndarray, b: np.ndarray):
        '''
        ### Description:

        Centered-form flux limiter, minmod. The operation is performed element-wise.
        '''
        return 0.5 * (np.sign(a)+np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

    def B_VanLeer(self, a: np.ndarray, b: np.ndarray):
        '''
        ### Description

        Centered-form flux limiter, Van Leer. The operation is performed element-wise.
        '''
        return a * b * (np.sign(a) + np.sign(b)) / (np.abs(a) + np.abs(b) + 1e-6)

    def minmod(self, a: np.ndarray, b: np.ndarray):
        '''
        ### Description:

        minmod flux limiter
        '''
        flag1 = ((a*b) > 0).astype(np.int32)
        flag2 = (a > 0).astype(np.int32)
        flag3 = (a <= 0).astype(np.int32)
        val = np.min(np.vstack((abs(a), abs(b))), axis=0)
        return flag1 * (flag2 * np.ones(a.shape) - flag3 * np.ones(a.shape)) * val

    def vanalbada(self, a: np.ndarray, b: np.ndarray):
        eps2 = (0.3 * self.dx) ** 3
        va = 0.5 * (np.sign(a) + np.sign(b) + 1) * \
            ((b**2 + eps2)*a + (a**2 + eps2)*b) / (a**2 + b**2 + eps2)
        
        return va