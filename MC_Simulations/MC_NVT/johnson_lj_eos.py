import numpy as np

class jhonson_lj_eos():

    def __init__(self, gamma, x):
        self.gamma = gamma
        self.x = x

    def constants(self, T):
        # a_i
        self.a1 = self.x[0]*T + self.x[1]*np.sqrt(T) + self.x[2] + self.x[3]/T + self.x[4]/T**2
        self.a2 = self.x[5]*T + self.x[6] + self.x[7]/T + self.x[8]/T**2
        self.a3 = self.x[9]*T + self.x[10] + self.x[11]/T
        self.a4 = self.x[12]
        self.a5 = self.x[13]/T + self.x[14]/T**2
        self.a6 = self.x[15]/T
        self.a7 = self.x[16]/T + self.x[17]/T**2
        self.a8 = self.x[18]/T**2

        # b_i
        self.b1 = self.x[19]/T**2 + self.x[20]/T**3
        self.b2 = self.x[21]/T**2 + self.x[22]/T**4
        self.b3 = self.x[23]/T**2 + self.x[24]/T**3
        self.b4 = self.x[25]/T**2 + self.x[26]/T**4
        self.b5 = self.x[27]/T**2 + self.x[28]/T**3
        self.b6 = self.x[29]/T**2 + self.x[30]/T**3 + self.x[31]/T**4

    def lj_eos(self, T, rho):
        
        self.constants(T)
        exp_term = np.exp(-self.gamma * rho**2)

        P = (
            rho*T
            + self.a1*rho**2 + self.a2*rho**3 + self.a3*rho**4 + self.a4*rho**5
            + self.a5*rho**6 + self.a6*rho**7 + self.a7*rho**8 + self.a8*rho**9
            + exp_term * (
                self.b1*rho**3 + self.b2*rho**5 + self.b3*rho**7 +
                self.b4*rho**9 + self.b5*rho**11 + self.b6*rho**13
            )
        )

        Z = P / (rho * T)

        # Helmholtz free energy
        fexp = np.exp(-self.gamma * rho**2)
        g1 = (1 - fexp) / (2 * self.gamma)
        g2 = -(fexp*rho**2 - 2*g1) / (2 * self.gamma)
        g3 = -(fexp*rho**4 - 4*g2) / (2 * self.gamma)
        g4 = -(fexp*rho**6 - 6*g3) / (2 * self.gamma)
        g5 = -(fexp*rho**8 - 8*g4) / (2 * self.gamma)
        g6 = -(fexp*rho**10 - 10*g5) / (2 * self.gamma)

        aex = (
            self.a1*rho +self. a2*rho**2/2 + self.a3*rho**3/3 + self.a4*rho**4/4 +
            self.a5*rho**5/5 + self.a6*rho**6/6 + self.a7*rho**7/7 + self.a8*rho**8/8
            + self.b1*g1 +self.b2*g2 + self.b3*g3 + self.b4*g4 + self.b5*g5 + self.b6*g6
        )

        mu_ex = aex + P/rho - T
        beta_mu_total_eos = np.log(rho) + mu_ex

        return [P, Z, mu_ex/T, beta_mu_total_eos]
    
    def calculate(self):
        red_temp = 1.12
        red_rhos = np.arange(0.02, 0.22, 0.02)
        final = []
        for red_rho in red_rhos:
            y = self.lj_eos(red_temp, red_rho)
            final.append(((red_temp, red_rho), y))
        return final

def main():
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

    results = eos.calculate()

    for (T, rho), (P, Z, beta_mu_ex, beta_mu_total_eos) in results:
        print(f"T={T:.2f}, rho={rho:.2f} | P={P:.5f}, Z={Z:.5f}, beta_mu_ex={beta_mu_ex:.5f}, beta_mu_total={beta_mu_total_eos:.5f}")

if __name__ == "__main__":
    main()