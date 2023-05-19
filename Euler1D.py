'''
### Description

This implementation of 1D Euler equation solver is a translation of the MATLAB and C++
code written by the author in Junior and Senior. Hopefully more Riemann solvers and more
flux limiters will be added.

Wcy, 2022.11
'''
gamma = 1.4
ga1 = gamma - 1.0
zz = (gamma - 1.0) / (2.0 * gamma)
import numpy as np
import RiemannExact as RE

class Euler1D:
    def __init__(self, fname = "settings.txt", SO = False, FD = False):
        '''
        ### Description

        We initialize the Euler 1d solver by a file. All the settings can be found in the code
        listed below. Comments about them will be added here later.
        '''
        # determine in what mode we will run the program
        self.FD = FD

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
        reconstruction_list = [0, 1, 2]
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
                                     1: 2nd order TVD\n\
                                     2: 5th order WENO")

            iL += 1
        
        # store the solution on the cell centers
        self.sol = np.zeros((3, self.ncells), dtype = np.float64)
        self.sol_old = np.zeros((3, self.ncells), dtype = np.float64)

        # store the reconstructed values on the cell interface
        self.rec_l = np.zeros((3, self.ncells+1), dtype = np.float64)
        self.rec_r = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the interface flux for the finite difference method
        self.flux_neg = np.zeros((3, self.ncells+1), dtype = np.float64) # f^-
        self.flux_pos = np.zeros((3, self.ncells+1), dtype = np.float64) # f^+

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

        # store the eigen vectors for every interface
        self.Li = np.zeros((3,3,self.ncells+1),dtype=np.float64)
        self.Ri = np.zeros((3,3,self.ncells+1),dtype=np.float64)

        # store the eigen value for every interface
        self.eigi = np.zeros((3, self.ncells+1),dtype=np.float64)

        # store the WENO-5 polynomial coefficient
        self.weno_c = np.array([
            [11.0/6.0, -7.0/6.0, 1.0/3.0],
            [1.0/3.0, 5.0/6.0, -1.0/6.0],
            [-1.0/6.0, 5.0/6.0, 1.0/3.0],
            [1.0/3.0, -7.0/6.0, 11.0/6.0]
        ])
    
        # calculate the mesh information
        self.SO = SO
        if not SO:
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
        else:
            self.dx = 10.0 / float(self.ncells)
            self.mesh = np.linspace(-0.0 + 0.5*self.dx, 10.0 - 0.5*self.dx, self.ncells)

            # set the initial condition
            for i in range(self.ncells):
                if self.mesh[i] < 1.0:
                    self.sol[0][i] = 3.857
                    self.sol[1][i] = 2.629
                    self.sol[2][i] = 10.333
                else:
                    self.sol[0][i] = 1.0 + 0.2 * np.sin(5.0 * (self.mesh[i]-5.0))
                    self.sol[1][i] = 0.0
                    self.sol[2][i] = 1.0


        self.set_boundary()
        self.prim2Con(0)

    def set_boundary(self):
        self.vL = self.sol[:, 0].copy().reshape(-1,1)
        self.vLL = self.sol[:, 0].copy().reshape(-1,1)

        if not self.SO:
            self.vR = self.sol[:, -1].copy().reshape(-1,1)
            self.vRR = self.sol[:, -1].copy().reshape(-1,1)
        else:
            self.vR = 2.0 * self.sol[:, -1].copy().reshape(-1,1)-\
                      1.0 * self.sol[:, -2].copy().reshape(-1,1)
            self.vRR = 2.0 * self.vR - 1.0 * self.sol[:, -1].copy().reshape(-1,1)

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
        self.reconstruction_0()
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

        # roe averages
        ratio = np.sqrt(rR / rL)
        r = ratio * rL
        u = (uL + ratio * uR) / (1 + ratio)
        H = (HL + ratio * HR) / (1 + ratio)
        a = np.sqrt((1.4 - 1)*(H - u * u / 2.0))
        self.avg[0,:] = r.copy()
        self.avg[1,:] = r * u
        self.avg[2,:] = a**2 * r / (gamma * ga1) + 0.5 * r * u * u

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

    def calc_eigen(self):
        '''
        ### Description

        Calculate the eigen vectors for every interface
        '''

        # calculate the Roe's average for every interface
        self.avg_roe()
        
        for i in range(self.ncells+1):
            rho = self.avg[0,i]
            u = self.avg[1,i] / rho
            E = self.avg[2,i] / rho
            p = ga1 * (self.avg[2,i] - 0.5 * rho * u * u)
            a = np.sqrt(gamma * p / rho)
            H = (self.avg[2,i] + p) / rho

            self.Ri[:, :, i] = np.array([[1.0, u-a, H-u*a], 
                                        [1.0, u, 0.5*u*u], 
                                        [1.0, u+a, H+u*a]]).T
            self.Li[:, :, i] = np.array([[ga1/4.0*(u/a)**2+0.5*u/a, -ga1/a*u/(a*2)-1.0/(2.0*a), ga1/(2.0*a*a)],
                                         [1-ga1/2.0*(u/a)**2, ga1*u/(a**2), -ga1/a**2],
                                         [ga1/4.0*(u/a)**2-0.5*u/a, -ga1/a*u/(a*2)+1.0/(2.0*a), ga1/(2.0*a*a)]])
            
            self.eigi[:, i] = np.array([u-a, u, u+a])

    def nonlinear_weight(self, q: np.ndarray):
        '''
        ### Description

        Use difference method to calculate the non-linear weight. Based on 
        the note written by Shu. It is only for the WENO-5th order scheme

        q is the 5-dimensional vector containing the stencil i:
        ++---++---++---++---++---++
        ||i-2||i-1|| i ||i+1||i+2||
        ++---++---++---++---++---++
          q[0] q[1] q[2] q[3] q[4]

        The function should be called for every conservative variable.
        '''
        eps = 1e-8

        # calculate the smooth-indicator
        beta0 = 13.0/12.0*(q[2]-2.0*q[3]+q[4])**2 + \
                1.0/4.0*(3.0*q[2]-4.0*q[3]+q[4])**2
        beta1 = 13.0/12.0*(q[1]-2.0*q[2]+q[3])**2 + \
                1.0/4.0*(q[1]-q[3])**2
        beta2 = 13.0/12.0*(q[0]-2.0*q[1]+q[2])**2 + \
                1.0/4.0*(q[0]-4.0*q[1]+3.0*q[2])**2
        beta = np.array([beta0, beta1, beta2])

        # linear weights
        d = np.array([3.0/10.0, 6.0/10.0, 1.0/10.0])
        d_bar = np.array([1.0/10.0, 6.0/10.0, 3.0/10.0])

        # assemble nonlinear weights
        alpha = d / (beta + eps)**2
        alpha_bar = d_bar / (beta + eps)**2

        weight = alpha / np.sum(alpha)
        weight_bar = alpha_bar / np.sum(alpha_bar)
        
        return weight, weight_bar
    
    def polynomial_WENO5(self, q: np.ndarray):
        '''
        ### Description

        5-dimensional vector is fed in, with the central one the cell being reconstructed.
        '''
        qR = []
        qL = []
        for r in range(3):
            vr = q[2-r:5-r] @ self.weno_c[r, :]
            qR.append(vr)
            
            vr = q[2-r:5-r] @ self.weno_c[r+1, :]
            qL.append(vr)

        return np.array(qR), np.array(qL)

    def reconstruction_WENO5_cell(self, cellI: int):
        '''
        ### Description:

        Peform 5th order WENO reconstruction. Requires 2 virtual cells on the 
        boundary. The stencil is as follows:
        ++---++---++---++---++---++
        ||i-2||i-1|| i ||i+1||i+2||
        ++---++---++---++---++---++
        The stencil above is used to reconstruct the right, left state of cell i.
        '''
        # assemble the virtual cells and the internal cells
        U = np.hstack((self.vLL, self.vL, self.sol, self.vR, self.vRR))
        self.calc_eigen()
        for i in range(self.ncells):
            # i-1/2
            l1, l2, l3 = self.Li[0, :, i], self.Li[1, :, i], self.Li[2, :, i]
            r1, r2, r3 = self.Ri[0, :, i], self.Ri[1, :, i], self.Ri[2, :, i]

            v1 = np.array([l1 @ U[:, i], l1 @ U[:, i+1], l1 @ U[:, i+2], l1 @ U[:, i+3], l1 @ U[:, i+4]])
            v2 = np.array([l2 @ U[:, i], l2 @ U[:, i+1], l2 @ U[:, i+2], l2 @ U[:, i+3], l2 @ U[:, i+4]])
            v3 = np.array([l3 @ U[:, i], l3 @ U[:, i+1], l3 @ U[:, i+2], l3 @ U[:, i+3], l3 @ U[:, i+4]])

            _, weight_bar1 = self.nonlinear_weight(v1)
            _, weight_bar2 = self.nonlinear_weight(v2)
            _, weight_bar3 = self.nonlinear_weight(v3)

            vr1, _ = self.polynomial_WENO5(v1)
            vr2, _ = self.polynomial_WENO5(v2)
            vr3, _ = self.polynomial_WENO5(v3)

            v_rec1 = weight_bar1 @ vr1
            v_rec2 = weight_bar2 @ vr2
            v_rec3 = weight_bar3 @ vr3
            v = np.array([v_rec1, v_rec2, v_rec3])

            self.rec_r[0, i] = r1@v
            self.rec_r[1, i] = r2@v
            self.rec_r[2, i] = r3@v

            # i + 1/2
            l1, l2, l3 = self.Li[0, :, i+1], self.Li[1, :, i+1], self.Li[2, :, i+1]
            r1, r2, r3 = self.Ri[0, :, i+1], self.Ri[1, :, i+1], self.Ri[2, :, i+1]

            v1 = np.array([l1 @ U[:, i], l1 @ U[:, i+1], l1 @ U[:, i+2], l1 @ U[:, i+3], l1 @ U[:, i+4]])
            v2 = np.array([l2 @ U[:, i], l2 @ U[:, i+1], l2 @ U[:, i+2], l2 @ U[:, i+3], l2 @ U[:, i+4]])
            v3 = np.array([l3 @ U[:, i], l3 @ U[:, i+1], l3 @ U[:, i+2], l3 @ U[:, i+3], l3 @ U[:, i+4]])

            weight1, _ = self.nonlinear_weight(v1)
            weight2, _ = self.nonlinear_weight(v2)
            weight3, _ = self.nonlinear_weight(v3)

            _, vr1 = self.polynomial_WENO5(v1)
            _, vr2 = self.polynomial_WENO5(v2)
            _, vr3 = self.polynomial_WENO5(v3)

            v_rec1 = weight1 @ vr1
            v_rec2 = weight2 @ vr2
            v_rec3 = weight3 @ vr3
            v = np.array([v_rec1, v_rec2, v_rec3])

            self.rec_l[0, i+1]= r1@v
            self.rec_l[1, i+1] = r2@v
            self.rec_l[2, i+1] = r3@v

        # zero-order reconstruction for the boundary
        self.rec_l[:, 0] = self.vL.ravel().copy()
        self.rec_r[:, self.ncells] = self.vR.ravel().copy()

    def flux_(self, U:np.ndarray)->np.ndarray:
        '''
        ### Description

        Given the conservative variables, calculate the flux
        '''
        rho = U[0]
        u = U[1] / rho
        E = U[2] / rho
        p = ga1 * (U[2] - 0.5 * rho * u * u)

        f = np.array([rho * u, rho*u*u + p, (rho * E + p)*u])
        return f

    def FD_flux_WENO5(self):
        '''
        ### Description
        Use 5-th order WENO reconstruction to calculate the numerical 
        flux directly. It is a finite-difference based method

        Requires 2 virtual cells on the boundary. The stencil is as follows:
        ++---++---++---++---++---++
        ||i-2||i-1|| i ||i+1||i+2||
        ++---++---++---++---++---++
        '''
        # assemble the virtual cells and the internal cells
        U = np.hstack((self.vLL, self.vL, self.sol, self.vR, self.vRR))
        self.calc_eigen()
        for i in range(self.ncells):
            # i-1/2
            l1, l2, l3 = self.Li[0, :, i], self.Li[1, :, i], self.Li[2, :, i]

            v1 = np.array([l1 @ self.flux_(U[:, i]), l1 @ self.flux_(U[:, i+1]), l1 @ self.flux_(U[:, i+2]), l1 @ self.flux_(U[:, i+3]), l1 @ self.flux_(U[:, i+4])])
            v2 = np.array([l2 @ self.flux_(U[:, i]), l2 @ self.flux_(U[:, i+1]), l2 @ self.flux_(U[:, i+2]), l2 @ self.flux_(U[:, i+3]), l2 @ self.flux_(U[:, i+4])])
            v3 = np.array([l3 @ self.flux_(U[:, i]), l3 @ self.flux_(U[:, i+1]), l3 @ self.flux_(U[:, i+2]), l3 @ self.flux_(U[:, i+3]), l3 @ self.flux_(U[:, i+4])])

            _, weight_bar1 = self.nonlinear_weight(v1)
            _, weight_bar2 = self.nonlinear_weight(v2)
            _, weight_bar3 = self.nonlinear_weight(v3)

            vr1, _ = self.polynomial_WENO5(v1)
            vr2, _ = self.polynomial_WENO5(v2)
            vr3, _ = self.polynomial_WENO5(v3)

            v_rec1 = weight_bar1 @ vr1
            v_rec2 = weight_bar2 @ vr2
            v_rec3 = weight_bar3 @ vr3
            v = np.array([v_rec1, v_rec2, v_rec3])

            self.flux_neg[:, i] = v.copy()

            # i+1/2
            l1, l2, l3 = self.Li[0, :, i+1], self.Li[1, :, i+1], self.Li[2, :, i+1]

            v1 = np.array([l1 @ self.flux_(U[:, i]), l1 @ self.flux_(U[:, i+1]), l1 @ self.flux_(U[:, i+2]), l1 @ self.flux_(U[:, i+3]), l1 @ self.flux_(U[:, i+4])])
            v2 = np.array([l2 @ self.flux_(U[:, i]), l2 @ self.flux_(U[:, i+1]), l2 @ self.flux_(U[:, i+2]), l2 @ self.flux_(U[:, i+3]), l2 @ self.flux_(U[:, i+4])])
            v3 = np.array([l3 @ self.flux_(U[:, i]), l3 @ self.flux_(U[:, i+1]), l3 @ self.flux_(U[:, i+2]), l3 @ self.flux_(U[:, i+3]), l3 @ self.flux_(U[:, i+4])])

            weight1, _ = self.nonlinear_weight(v1)
            weight2, _ = self.nonlinear_weight(v2)
            weight3, _ = self.nonlinear_weight(v3)

            _, vr1 = self.polynomial_WENO5(v1)
            _, vr2 = self.polynomial_WENO5(v2)
            _, vr3 = self.polynomial_WENO5(v3)

            v_rec1 = weight1 @ vr1
            v_rec2 = weight2 @ vr2
            v_rec3 = weight3 @ vr3
            v = np.array([v_rec1, v_rec2, v_rec3])

            self.flux_pos[:, i+1] = v.copy()

         # zero-order reconstruction for the boundary
        self.flux_neg[:, self.ncells] = self.Li[:,:,self.ncells]@(self.flux_(self.vR).ravel())
        self.flux_pos[:, 0] = self.Li[:,:,0]@(self.flux_(self.vL).ravel())

        # compute the numerical flux at the interface
        for i in range(self.ncells+1):
            V = 0.5 * (np.eye(3) + np.sign(np.diag(self.eigi[:, i]))) @ self.flux_pos[:, i] +\
                0.5 * (np.eye(3) - np.sign(np.diag(self.eigi[:, i]))) @ self.flux_neg[:, i]
            
            self.flux[:, i] = self.Ri[:,:,i] @ V

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
            if not self.FD:
                # choose Reconstruction method
                if self.reconstruction == 1:
                    self.reconstruction_TVD_centered()
                elif self.reconstruction == 0:
                    self.reconstruction_0()
                elif self.reconstruction == 2:
                    self.reconstruction_WENO5_cell(0)

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
            else:
                self.FD_flux_WENO5()
            
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