import numpy as np

class RiemannExact:
    def __init__(self, WL: np.ndarray, WR: np.ndarray, gamma = 1.4e0):
        '''
        `WL`, `WR`: Array that stores the primitive variables on the left side and the right side, 
        the sequence should be: rho, u, p
        '''
        # left and right data
        self.rhoL, self.uL, self.pL = WL[0], WL[1], WL[2]
        self.rhoR, self.uR, self.pR = WR[0], WR[1], WR[2]
        self.gamma = gamma

        # sound speed
        self.aL, self.aR = np.sqrt(gamma * self.pL/self.rhoL),\
                           np.sqrt(gamma * self.pR/self.rhoR)
        
        # stared variables in the central region
        self.pStar = 0.0
        self.uStar = 0.0

        # now check if the problem can be sovled, if not, throw out an error
        if 2.e0 / (gamma - 1.e0) * (self.aL + self.aR) <= (self.uR - self.uL):
            raise Exception("Initial state will generate vaccum. Problem can not be solved!")

        # frequently used constants
        self.c1 = (gamma + 1.e0) / 2.e0
        self.c2 = (gamma - 1.e0) / 2.e0
        self.c3 = self.c1 / gamma
        self.c4 = self.c2 / gamma


    def fK(self, p, rhoK, uK, pK, aK)->tuple:
        '''
        Note that this function returns both the function value and
        the derivative
        '''
        # gamma = self.gamma

        if p > pK: # shock wave
            AK = 1.e0 / self.c1 / rhoK
            BK = self.c2/self.c1*pK

            F = (p - pK)*np.sqrt(AK/(p+BK))
            dF = np.sqrt(AK/(p+BK)) * (1.e0 - (p - pK) / (2.e0 * (p+BK)))

            return F, dF
        else: # expansion wave
            F = aK / self.c2 * ((p / pK)**(self.c4) - 1.e0)
            dF = 1.e0 / (rhoK * aK) * (p / pK)**(-self.c3)

            return F, dF
        
    def initial_guess(self):
        # we use 2 rarefraction wave approximation
        apv = 0.25 * (self.rhoL + self.rhoR) * (self.aL + self.aR)
        ppv = 0.5 * (self.pL + self.pR) + 0.5 * (self.uL - self.uR) * apv
        ppv = max(0.0, ppv)
        pmin = min(self.pL, self.pR)
        pmax = max(self.pL, self.pR)
        q = pmax / pmin

        if (q <= 2.0) and (pmin <= ppv) and (ppv <= pmax):
            p0 = ppv
        else:
            if(ppv < pmin):
                p0 = ((self.aL + self.aR - self.c2 * (self.uR - self.uL)) / \
                      (self.aL / self.pL ** self.c4 + self.aR / self.pR ** self.c4))**(1.0 / self.c4)
            else: 
                AL = 1.e0 / self.c1 / self.rhoL
                BL = self.c2/self.c1*self.pL
                AR = 1.e0 / self.c1 / self.rhoR
                BR = self.c2/self.c1*self.pR
                gL = ((AL) / (BL + ppv)) ** 0.5
                gR = ((AR) / (BR + ppv)) ** 0.5
                delta_u = self.uR - self.uL
                p0 = (gL * self.pL + gR * self.pR - delta_u) / (gL + gR)


        return p0
    
    def solve(self, max_iter = 20, tol_res = 1e-6)->bool:
        pOld = self.initial_guess()
        pNew = 0.0
        delta_u = self.uR - self.uL
        success = False

        for i in range(max_iter):
            FL, dFL = self.fK(pOld, self.rhoL, self.uL, self.pL, self.aL)
            FR, dFR = self.fK(pOld, self.rhoR, self.uR, self.pR, self.aR)

            pNew = pOld - (FL + FR + delta_u) / (dFL + dFR)
            res = abs(pNew - pOld) / (0.5 * (pNew + pOld))
            if res <= tol_res:
                success = True
                break

            if(pNew < 0.0):
                pNew = tol_res
            pOld = pNew

        if not success:
            print("Warning: tolerance is not met after max_iter")
        
        self.pStar = pNew
        self.uStar = 0.5 * (self.uL + self.uR + FR - FL)

        return success
    
    def sample(self, x2t):
        if x2t <= self.uStar: # aquire the state at the left side of contact discontinuity

            if self.pStar > self.pL: # shock wave on the left side
                # calculate the shock wave velocity:
                SL = self.uL - self.aL * np.sqrt(self.c3 * self.pStar / self.pL + self.c4)
                
                if x2t < SL: # left uniform region, in front of the shock
                    W = np.array([self.rhoL, self.uL, self.pL])
                else: # contact region, behind the shock
                    self.rhoStarL = self.rhoL * (self.pStar / self.pL + self.c2/self.c1) /\
                                    (self.c2/self.c1 * self.pStar / self.pL + 1.0)
                    W = np.array([self.rhoStarL, self.uStar, self.pStar])

            else: # expansion wave on the left side
                self.rhoStarL = self.rhoL * (self.pStar/self.pL) ** (1.0 / self.gamma)
                aStarL = self.aL * (self.pStar/self.pL) ** self.c4
                SHL = self.uL - self.aL
                STL = self.uStar - aStarL

                if x2t > STL: # left of the contact discontinuity
                    W = np.array([self.rhoStarL, self.uStar, self.pStar])
                elif x2t > SHL: # in the expansion fan
                    rho = self.rhoL * (1.0 / self.c1 + self.c2 / self.c1 / self.aL * (self.uL - x2t)) ** (1.0/self.c2)
                    u = 1.0/self.c1 * (self.aL + self.c2*self.uL + x2t)
                    p = self.pL*(1.0/self.c1 + self.c2/self.c1/self.aL*(self.uL - x2t))**(1.0/self.c4)
                    W = np.array([rho,u,p])
                else: # in the left uniform region
                    W = np.array([self.rhoL, self.uL, self.pL])

        else: # aquire the state at the right side of the contact discontinuity

            if self.pStar > self.pR: # shock wave on the right side
                # calculate the shock wave velocity
                SR = self.uR+self.aR * np.sqrt(self.c3*self.pStar/self.pR + self.c4)
                
                if x2t > SR: # right side uniform region
                    W = np.array([self.rhoR, self.uR, self.pR])
                else: # right side of the contact discontinuity
                    rhoStarR = self.rhoR * (self.pStar/self.pR + self.c2/self.c1)/\
                               (self.c2/self.c1*self.pStar/self.pR + 1.0)
                    W = np.array([rhoStarR, self.uStar, self.pStar])

            else: # expansion wave on the right side
                rhoStarR = self.rhoR * (self.pStar / self.pR) ** (1.0 / self.gamma)
                aStarR = self.aR * (self.pStar / self.pR) ** (self.c4)
                SHR = self.uR + self.aR
                STR = self.uStar + aStarR

                if x2t < STR: # right side of the contact discontinuity
                    W = np.array([rhoStarR, self.uStar, self.pStar])
                elif x2t < SHR: # inside the expansion fan
                    rho = self.rhoR*(1.0/self.c1 - self.c2/self.c1/self.aR*(self.uR - x2t))**(1.0/self.c2)
                    u = 1.0/self.c1 * (-self.aR + self.c2*self.uR + x2t)
                    p = self.pR*(1.0/self.c1 - self.c2/self.c1/self.aR*(self.uR - x2t))**(1.0/self.c4)
                    W = np.array([rho, u, p])
                else: # right side uniform region
                    W = np.array([self.rhoR, self.uR, self.pR])

        return W

