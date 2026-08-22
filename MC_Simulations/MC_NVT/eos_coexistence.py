import numpy as np
from scipy.optimize import fsolve
from johnson_lj_eos import jhonson_lj_eos


def compute_eos_psat(T):

    gamma = 3.0
    x = np.array([
            0.8623085097507421, 2.976218765822098, -8.402230115796038,
            0.1054136629203555, -0.8564583828174598, 1.582759470107601,
            0.7639421948305453, 1.753173414312048, 2.798291772190376e+03,
            -4.8394220260857657e-02, 0.9963265197721935,
            -3.698000291272493e+01, 2.084012299434647e+01,
            8.305402124717285e+01, -9.574799715203068e+02,
            -1.477746229234994e+02, 6.398607852471505e+01,
            1.603993673294834e+01, 6.805916615864377e+01,
            -2.791293578795945e+03, -6.245128304568454,
            -8.116836104958410e+03, 1.488735559561229e+01,
            -1.059346754655084e+04, -1.131607632802822e+02,
            -8.867771540418822e+03, -3.986982844450543e+01,
            -4.689270299917261e+03, 2.593535277438717e+02,
            -2.694523589434903e+03, -7.218487631550215e+02,
            1.721802063863269e+02
        ])
    eos = jhonson_lj_eos(gamma, x)

    # system of equations
    def equations(vars):
        rho_v, rho_l = vars

        P_v, _, _, mu_v = eos.lj_eos(T, rho_v)
        P_l, _, _, mu_l = eos.lj_eos(T, rho_l)

        return [
            mu_v - mu_l,   # equal chemical potential
            P_v - P_l      # equal pressure
        ]

    # initial guess (important!)
    guess = [0.05, 0.7]

    rho_v, rho_l = fsolve(equations, guess)

    # compute coexistence pressure
    P_sat, _, _, _ = eos.lj_eos(T, rho_v)

    return P_sat, rho_v, rho_l