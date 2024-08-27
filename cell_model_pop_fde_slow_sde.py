import numpy as np
from scipy import signal, optimize
from scipy.special import factorial, gamma
import random as rdm


class Cell_Population:
    def __init__(self, T_final, delta_t=0.15, alpha_mem=0.7, sigma=0.2):
        self.delta_t = delta_t
        self.treat = True
        self.alpha_mem = alpha_mem
        self.T = T_final
        self.t0 = 0
        self.sigma = sigma

        # parameters for Fischer et al. framework
        self.beta_ps = 0.2
        self.beta_pr = -0.21
        self.beta_ts = -0.21
        self.beta_tr = 0.2
        self.alpha = 0.1
        self.delta = 0.1

    # calculating necessary observation rate, not used in RL training
    def time_pause(self, Qa=0.3, Qb=0.6):
        lamb = (self.beta_pr - self.beta_ps - self.delta) / (self.beta_pr - self.beta_ps)
        return np.log(Qa*(lamb - Qb) / (Qb*(lamb - Qa))) / (self.beta_pr - self.beta_ps - self.delta)

    def time_treat(self, Qa=0.3, Qb=0.6):
        a = self.beta_ts - self.beta_tr
        b_ = self.beta_tr - self.beta_ts - self.alpha
        return (np.log(1 - (2*a*Qb + b_) / np.sqrt(-(4*a*self.alpha - b_**2))) - np.log(1 + (2*a*Qb + b_) / np.sqrt(-(4*a*self.alpha - b_**2))) - np.log(1 - (2*a*Qa + b_) / np.sqrt(-(4*a*self.alpha - b_**2))) + np.log(1 + (2*a*Qa + b_) / np.sqrt(-(4*a*self.alpha - b_**2)))) / np.sqrt(-(4*a*self.alpha - b_**2))


    def dNdt(self, y, b):
        if b > 0:
            rate = (self.beta_ts - self.alpha)*y[0]
        else:
            rate = self.beta_ps*y[0] + self.delta*y[1]
        return rate

    def dRdt(self, y, b):
        if b > 0:
            rate = self.alpha*y[0] + self.beta_tr*y[1]
        else:
            rate = (self.beta_pr - self.delta)*y[1]
        return rate

    def odes(self, t, y, b):
        # packing coupled ODEs together to be numerically integrated
        N_ode = self.dNdt(y, b)
        R_ode = self.dRdt(y, b)

        dSpeciesdt = np.array([N_ode,R_ode]).T
        return dSpeciesdt


    # initialize simulation

    def initialize(self, param=0, h=2**-6, rand=False, mu=1, mu_tol=1.0e-6):
        if rand:
            N0 = rdm.randint(100,1000)
        else:
            N0 = 1000
        self.S0 = N0
        R0 = 0
        y0 = np.array([[N0, R0]]).T
        self.init_conditions = y0
        self.t_start = 0
        self.h = h
        T = self.T
        t0 = self.t0
        alpha = self.alpha_mem
        f_fun = self.odes


        # Check order of the FDE
        alpha = np.array(alpha).flatten()
        if any(alpha <= 0):
            i_alpha_neg = np.where(alpha <= 0)[0][0]
            raise ValueError(f'The orders ALPHA of the FDEs must be all positive. The value ALPHA({i_alpha_neg}) = {alpha[i_alpha_neg]:.6f} cannot be accepted.')

        # Check the step-size of the method
        if h <= 0:
            raise ValueError(f'The step-size H for the method must be positive. The value H = {h:.6e} cannot be accepted.')

        # Check compatibility size of the problem with number of fractional orders
        alpha_length = len(alpha)
        problem_size = y0.shape[0]
        if alpha_length > 1:
            if problem_size != alpha_length:
                raise ValueError(f'Size {problem_size} of the problem as obtained from initial conditions not compatible with the number {alpha_length} of fractional orders for multi-order systems.')
        else:
            alpha = alpha * np.ones(problem_size)
            alpha_length = problem_size

        # Storage of initial conditions
        ic = {'t0': t0, 'y0': y0, 'm_alpha': np.ceil(alpha).astype(int)}
        ic['m_alpha_factorial'] = np.array([factorial(j) for i in range(alpha_length) for j in range(ic['m_alpha'][i])]).reshape(alpha_length, -1)

        # Storage of information on the problem
        Probl = {
            'ic': ic,
            'f_fun': f_fun,
            'problem_size': problem_size,
            'param': param,
            'alpha': alpha,
            'alpha_length': alpha_length
        }
        self.Probl = Probl

        # Check number of initial conditions
        if y0.shape[1] < max(ic['m_alpha']):
            raise ValueError(f'A not sufficient number of assigned initial conditions. Order ALPHA = {max(alpha):.6f} requires {max(ic["m_alpha"])} initial conditions.')

        # Check compatibility size of the problem with size of the vector field
        f_temp = self.f_vectorfield(t0, y0[:, 0], Probl)
        if Probl['problem_size'] != f_temp.shape[0]:
            raise ValueError(f'Size {Probl["problem_size"]} of the problem as obtained from initial conditions not compatible with the size {f_temp.shape[0]} of the output of the vector field F_FUN.')


        # Number of points in which to evaluate weights and solution
        N = int(np.ceil((T - t0) / h))
        self.N = N

        # Preallocation of some variables
        y = np.zeros((Probl['problem_size'], N + 1))
        y[:, 0] = y0[:, 0]
        self.y = y
        fy = np.zeros((Probl['problem_size'], N + 1))
        fy[:, 0] = f_temp
        self.fy = fy
        zn_pred = np.zeros((Probl['problem_size'], N + 1))
        self.zn_pred = zn_pred
        zn_corr = np.zeros((Probl['problem_size'], N + 1)) if mu > 0 else 0
        self.zn_corr = zn_corr

        # Evaluation of coefficients of the PECE method
        nvett = np.arange(N + 2)
        bn = np.zeros((Probl['alpha_length'], N + 1))
        an = np.zeros_like(bn)
        a0 = np.zeros_like(bn)
        for i_alpha in range(Probl['alpha_length']):
            find_alpha = np.where(alpha[i_alpha] == alpha[:i_alpha])[0]
            if find_alpha.size > 0:
                bn[i_alpha, :] = bn[find_alpha[0], :]
                an[i_alpha, :] = an[find_alpha[0], :]
                a0[i_alpha, :] = a0[find_alpha[0], :]
            else:
                nalpha = nvett ** alpha[i_alpha]
                nalpha1 = nalpha * nvett
                bn[i_alpha, :] = nalpha[1:] - nalpha[:-1]
                an[i_alpha, :] = np.concatenate(([1], nalpha1[:-2] - 2 * nalpha1[1:-1] + nalpha1[2:]))
                a0[i_alpha, :] = np.concatenate(([0], nalpha1[:-2] - nalpha[1:-1] * (nvett[1:-1] - alpha[i_alpha] - 1)))


        METH = {
            'bn': bn, 'an': an, 'a0': a0,
            'halpha1': h ** alpha / gamma(alpha + 1),
            'halpha2': h ** alpha / gamma(alpha + 2),
            'mu': mu, 'mu_tol': mu_tol
        }
        self.meth = METH

        # Initializing solution and process of computation
        t = t0 + np.arange(N + 1) * h
        self.t = t

        self.beg = 1


    # simulatation implementation

    def simulate_population(self, b, delta_t=None, plot=True):
        self.Probl['param'] = b

        if delta_t is not None:
            self.delta_t = delta_t
        self.t_stop = self.t_start + self.delta_t

        # Check to ensure simulation call is less than total simulation length
        if self.t_stop > self.T:
            raise ValueError(f'Simulation call will exceed total time allotted and thus can not be accepted.')


        t, y = self.FDE_PI12_PC(self.t_stop) # solving for values at next timestep using fractional integration

        # upacking
        N = y[0,:]
        R = y[1,:]

        self.t_start = t[-1]
        S = N + R

        # get reward
        p_init = self.S0
        p_final = S[-1]

        growth_rate = (np.log(p_final) - np.log(p_init)) / self.delta_t
        cost = growth_rate
        self.S0 = p_final

        if plot:
            return t, S, N, R
        else:
            return t, S, N, R, cost


    ##################################
    # fuctions for numerical integration of nonlocal dynamics of fractional differential equations
    # see "Numerical solution of fractional differential equations: A survey and a software tutorial" by Roberto Garrappa for a good reference for theory and simulation of FDE systems

    def FDE_PI12_PC(self, T_stop):

        # Main process of computation by means of the FFT algorithm

        N = int(np.ceil((T_stop - self.t0) / self.h))

        self.y, self.fy = self.Triangolo(self.beg, N, self.t, self.y, self.fy, self.zn_pred, self.zn_corr, self.N, self.meth, self.Probl)
        self.beg = N

        # Evaluation solution in T when T is not in the mesh
        if T_stop < self.t[N]:
            c = (T_stop - self.t[N-1]) / self.h
            self.t[N] = T_stop
            self.y[:, N] = (1 - c) * self.y[:, N-1] + c * self.y[:, N]

        return self.t[:N + 1], self.y[:, :N + 1]


    def Triangolo(self, nxi, nxf, t, y, fy, zn_pred, zn_corr, N, METH, Probl):
        # trunc_wind = 10000 # for changing environments, introducing truncations increases error significantly

        for n in range(nxi, min(N, nxf) + 1):

            # Evaluation of the predictor
            Phi = np.zeros(Probl['problem_size'])
            # if nxi == 1: # Case of the first triangle
            #     j_beg = 0
            # else: # Case of any triangle but not the first
            #     j_beg = max(0,N-trunc_wind)
            j_beg = 0

            for j in range(j_beg, n):
                Phi += METH['bn'][:Probl['alpha_length'], n - j-1] * fy[:, j]

            St = self.StartingTerm(t[n], Probl['ic'])
            # if no corrector application, add noise now
            if METH['mu'] == 0:
                noise = np.sqrt(2*self.sigma) * np.sqrt(self.h) * np.random.normal(size=2)

                y_pred = St + METH['halpha1'] * (zn_pred[:, n] + Phi) + noise
                # check to see if values are physical
                for i,k in enumerate(y_pred):
                    if k < 0:
                        y_pred[i] = 0
            else:
                y_pred = St + METH['halpha1'] * (zn_pred[:, n] + Phi)
            f_pred = self.f_vectorfield(t[n], y_pred, Probl)


            # Evaluation of the corrector

            if METH['mu'] == 0:
                y[:, n] = y_pred
                fy[:, n] = f_pred
            else:
                # if nxi == 1: # Case of the first triangle
                #     j_beg = 1
                # else: # Case of any triangle but not the first
                #     j_beg = max(1,N-trunc_wind)
                j_beg = 1
                Phi = np.zeros(Probl['problem_size'])
                for j in range(j_beg, n):
                    Phi += METH['an'][:Probl['alpha_length'], n - j] * fy[:, j]

                Phi_n = St + METH['halpha2'] * (METH['a0'][:Probl['alpha_length'], n] * fy[:, 0] + zn_corr[:, n] + Phi)
                yn0 = y_pred
                fn0 = f_pred
                stop = False
                mu_it = 0

                while not stop:
                    yn1 = Phi_n + METH['halpha2'] * fn0
                    mu_it += 1
                    if METH['mu'] == float('inf'):
                        stop = np.linalg.norm(yn1 - yn0, np.inf) < METH['mu_tol']
                        if mu_it > 100 and not stop:
                            print(f"Warning: It has been requested to run corrector iterations until convergence but the process does not converge to the tolerance {METH['mu_tol']} in 100 iterations")
                            stop = True
                    else:
                        stop = mu_it == METH['mu']

                    fn1 = self.f_vectorfield(t[n], yn1, Probl)
                    yn0 = yn1
                    fn0 = fn1

                # add noise
                noise = np.sqrt(2*self.sigma) * np.sqrt(self.h) * np.random.normal(size=2)
                # yn1 = yn1 + noise # arithmetic brownian motion
                yn1 = yn1 * (1 + noise) # geometric brownian motion
                # check to see if values are physical
                for i,k in enumerate(yn1):
                    if k < 0:
                        yn1[i] = 0
                fn1 = self.f_vectorfield(t[n], yn1, Probl)

                y[:, n] = yn1
                fy[:, n] = fn1
        return y, fy

    def f_vectorfield(self, t, y, Probl):
        if Probl['param'] is None:
            return Probl['f_fun'](t, y)
        else:
            return Probl['f_fun'](t, y, Probl['param'])

    def StartingTerm(self, t, ic):
        ys = np.zeros(ic['y0'].shape[0])
        for k in range(1, max(ic['m_alpha']) + 1):
            if len(ic['m_alpha']) == 1:
                ys += (t - ic['t0'])**(k-1) / ic['m_alpha_factorial'][k-1] * ic['y0'][:, k-1]
            else:
                i_alpha = np.where(k <= ic['m_alpha'])[0]
                ys[i_alpha] += (t - ic['t0'])**(k-1) * ic['y0'][i_alpha, k-1] / ic['m_alpha_factorial'][i_alpha, k-1]
        return ys
    ################ end ################