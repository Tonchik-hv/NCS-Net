import numpy as np
from scipy.integrate import odeint
import scipy.interpolate
from scipy.special import erf, lambertw, airy, jv

def gaussian_pulse(t, x, y, z, a0=1.0, duration=10*np.pi, position=0.0, phase=0.0, polarization=1.0):
    zeta = t - z
    envelop_func = np.exp(-(zeta + position)**2/2/duration**2)
    ex = a0 * ((zeta + position)/duration**2) * envelop_func * np.sin(zeta + position + phase) - a0 * envelop_func * np.cos(zeta + position + phase)
    ey = polarization * (a0 * ((zeta + position)/duration**2) * envelop_func * np.cos(zeta + position + phase) + a0 * envelop_func * np.sin(zeta + position + phase))
    ez = 0.0 * zeta
    bx = -ey
    by = ex
    bz = 0.0 * zeta
    return np.array([ex, ey, ez, bx, by, bz], dtype=np.float64)

def f(state, t, a0=1.0, duration=10*np.pi):
    x, y, z, px, py, pz = state
    gamma = np.sqrt(1+px**2+py**2+pz**2)
    der_x, der_y, der_z = px/gamma, py/gamma, pz/gamma
    ex, ey, ez, bx, by, bz = gaussian_pulse(t,x,y,z, a0, duration)
    der_px = -ex - (py/gamma*bz - pz/gamma*by)
    der_py = -ey - (pz/gamma*bx - px/gamma*bz)
    der_pz = -ez - (px/gamma*by - py/gamma*bx)
    return np.array([der_x, der_y, der_z, der_px, der_py, der_pz], dtype=np.float64)

def calculate_trajectory(state0, t, a0=1.0, duration=10*np.pi):
    states = odeint(f, state0, t, (a0, duration,))
    return np.array([states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]], dtype=np.float64)

def calculate_N_photons_with_freqs(states, t, theta=np.pi, phi=0):
    alpha = 1/137
    x, y, z, ux, uy, uz = states
    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    
    denom = gamma - np.cos(phi)*np.sin(theta)*ux - np.sin(phi)*np.sin(theta)*uy - uz*np.cos(theta)
    f_phi = (np.sin(phi)*ux - np.cos(phi)*uy)/denom
    f_theta = (np.sin(theta)*uz - np.cos(phi)*np.cos(theta)*ux + np.sin(phi)*np.cos(theta)*uy)/denom
    t_ret = t - x*np.sin(theta)*np.cos(phi) - y*np.sin(theta)*np.sin(phi) - z*np.cos(theta)
    
    interp_fun_phi = scipy.interpolate.interp1d(t_ret, f_phi, fill_value="extrapolate")
    interp_fun_theta = scipy.interpolate.interp1d(t_ret, f_theta, fill_value="extrapolate")
    interp_t_ret = np.linspace(t_ret[0], t_ret[-1], t_ret.shape[0])
    dt_ret = interp_t_ret[1] - interp_t_ret[0]
    if dt_ret == 0: dt_ret = 1e-9
    
    interp_freqs = np.fft.fftfreq(interp_t_ret.shape[0], d=dt_ret/2/np.pi)
    interp_f_phi = interp_fun_phi(interp_t_ret)
    interp_f_theta = interp_fun_theta(interp_t_ret)
    
    interp_fourier_complex_phi = np.fft.fft(interp_f_phi)*dt_ret
    interp_fourier_complex_theta = np.fft.fft(interp_f_theta)*dt_ret
    
    N_photons = alpha*abs(interp_freqs)*(abs(interp_fourier_complex_phi)**2 + abs(interp_fourier_complex_theta)**2)/4/(np.pi**2)
    idxs = np.argwhere(interp_freqs >= 0).flatten()
    return N_photons[idxs], interp_freqs[idxs]


class D2ECalculator:

    def __init__(self, a0: float, T: float, gamma: float):
        self.a0 = float(a0)
        self.T = float(T)
        self.gamma = float(gamma)
        self.p_minus = self.gamma + np.sqrt(self.gamma**2-1)
        self.alpha = 1 / 137.0

    def F(self, phi, chi):
        return (chi - 1.0) * phi + self.a0**2 * chi * np.sqrt(np.pi) / 2.0 * erf(phi)

    def d1F(self, phi, chi):
        return (chi - 1.0) + self.a0**2 * chi * np.exp(-phi**2)

    def d2F(self, phi, chi):
        return -2.0 * self.a0**2 * chi * phi * np.exp(-phi**2)

    def d3F(self, phi, chi):
        return self.a0**2 * chi * (4.0 * phi**2 - 2.0) * np.exp(-phi**2)

    def d4F(self, phi, chi):
        return -4.0 * self.a0**2 * chi * phi * (2.0 * phi**2 - 3.0) * np.exp(-phi**2)

    def q(self, phi, chi):
        return 1j * self.F(phi, chi) - phi**2 / 2.0 / self.T

    def d1q(self, phi, chi):
        return 1j * self.d1F(phi, chi) - phi / self.T

    def d2q(self, phi, chi):
        return 1j * self.d2F(phi, chi) - 1 / self.T

    def d3q(self, phi, chi):
        return 1j * self.d3F(phi, chi)

    def d4q(self, phi, chi):
        return 1j * self.d4F(phi, chi)

    def phi0(self, chi, chi0):
        chi = np.asarray(chi, dtype=float)
        chi1 = 1.0 / (1.0 + self.a0**2)
        phi0 = np.zeros_like(chi, dtype=np.complex128)

        ratio = self.a0**2 * chi / (1.0 - chi)

        m1 = chi <= chi1
        if np.any(m1):
            phi0[m1] = 1j*np.sqrt(np.abs(np.log(ratio[m1])))

        m2 = (chi > chi1) & (chi < chi0)
        if np.any(m2):
            phi0[m2] = np.sqrt(np.log(ratio[m2]))

        m3 = chi >= chi0
        if np.any(m3):
            chi3 = chi[m3]
            bar_phi = np.sqrt(0.5*lambertw(-2*self.T**2*self.a0**4+0j))
            tilde_phi0 = np.sqrt(np.log(self.a0**2 * chi3 / (1.0 - chi3 + self.a0**2*chi3*np.exp(-bar_phi**2-(1-chi3)**2))))
            d_tilde_phi0 = - self.d1q(tilde_phi0, chi3) / self.d2q(tilde_phi0, chi3)   
            phi0[m3] = tilde_phi0 + d_tilde_phi0
    
        return phi0

    def Aplus(self, chi, chi0):
        chi = np.asarray(chi, dtype=float)
        phi0 = self.phi0(chi, chi0)
        Aplus = np.zeros_like(chi, dtype=np.float64)

        m1 = chi <= chi0
        if np.any(m1):
            chi1 = chi[m1]
            x0 = 1.5 * self.T * 1j * self.F(phi0[m1], chi1)
            x2 = 1.5 * self.T * np.abs(self.d2F(phi0[m1], chi1))
            y = (np.abs(x0)**(2/3)*np.exp(2j*np.angle(x0))).real
            y0 = -1.173713222709128
            Ai, _, Bi, _ = airy(y)
            # AiSm = np.where(y < y0, np.sqrt(Ai**2 + Bi**2), Ai)
            AiSm = Ai.copy()
            mask = (y < y0)
            AiSm[mask] = np.hypot(Ai[mask], Bi[mask])
            g_phi = np.exp(-phi0[m1]**2/2.0).real
            Aplus[m1] = 2.0*np.pi*np.sqrt(3.0)*np.sqrt(np.abs(x0)**(1.0/3.0)/x2)*g_phi*AiSm
            
        m2 = chi > chi0
        if np.any(m2):
            chi2 = chi[m2]
            q = self.q(phi0[m2], chi2)
            d2q = self.d2q(phi0[m2], chi2)
            d3q = self.d3q(phi0[m2], chi2)
            d4q = self.d4q(phi0[m2], chi2)

            Sigma = 1.0+(d4q/8.0/d2q**2-5.0*d3q**2/24.0/d2q**3)/self.T
            Aplus[m2] = np.sqrt(8.0*np.pi/(self.T*np.abs(d2q)))*np.abs(Sigma)*np.exp(self.T*np.real(q))
        
        return Aplus

    def d2E(self, chi, chi0):
        Aplus = self.Aplus(chi, chi0)
        d2E = self.alpha * self.a0**2 * self.T**2 * chi**2 * self.p_minus**2 / (8.0 * np.pi**2) * Aplus**2
        return d2E


def peak_position(a0, tau, gamma):
    omega_L, c = 1, 1
    alpha = 1/137
    x1 = 1.018 * alpha
    u_minus = gamma + np.sqrt(gamma**2 - 1)
    ksi = 1 / (1 + a0**2)
    eps = x1 * (1 - ksi)**(1/3)/(c*tau)**(2/3)
    pos = omega_L * u_minus**2 * ksi * (1 + eps)
    return pos


class D2ECalculator_angles:

    def __init__(self, a0: float, T: float, gamma: float):
        self.a0 = float(a0)
        self.T = float(T)
        self.gamma = float(gamma)
        self.p_minus = self.gamma + np.sqrt(self.gamma**2-1)
        self.alpha = 1 / 137.0

    def _broadcast(self, *args):
        return np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in args))

    def k(self, theta):
        theta = np.asarray(theta, dtype=float)
        pz = - np.sqrt(self.gamma**2-1)
        pn = np.cos(theta) * pz
        return self.p_minus * (self.gamma - pn) 
    
    def b(self, theta):
        theta = np.asarray(theta, dtype=float)
        return np.sin(theta / 2.0) ** 2
    
    def alpha_bar(self, chi, theta):
        chi, theta = self._broadcast(chi, theta)
        return self.a0*np.sin(theta)*chi*self.p_minus
        
    def F(self, phi, chi, theta):
        chi, theta = self._broadcast(chi, theta)
        k, b = self.k(theta), self.b(theta)
        return (chi * k - 1.0) * phi + self.a0**2 * chi * b * np.sqrt(np.pi) / 2.0 * erf(phi)

    def d1F(self, phi, chi, theta):
        chi, theta = self._broadcast(chi, theta)
        k, b = self.k(theta), self.b(theta)
        return (chi * k - 1.0) + self.a0**2 * chi * b * np.exp(-phi**2)

    def d2F(self, phi, chi, theta):
        chi, theta = self._broadcast(chi, theta)
        b = self.b(theta)
        return -2.0 * self.a0**2 * chi * b * phi * np.exp(-phi**2)

    def d3F(self, phi, chi, theta):
        chi, theta = self._broadcast(chi, theta)
        b = self.b(theta)
        return self.a0**2 * chi * b * (4.0 * phi**2 - 2.0) * np.exp(-phi**2)

    def d4F(self, phi, chi, theta):
        chi, theta = self._broadcast(chi, theta)
        b = self.b(theta)
        return -4.0 * self.a0**2 * chi * b * phi * (2.0 * phi**2 - 3.0) * np.exp(-phi**2)

    def q(self, phi, chi, theta):
        return 1j * self.F(phi, chi, theta) - phi**2 / 2.0 / self.T

    def d1q(self, phi, chi, theta):
        return 1j * self.d1F(phi, chi, theta) - phi / self.T

    def d2q(self, phi, chi, theta):
        return 1j * self.d2F(phi, chi, theta) - 1 / self.T

    def d3q(self, phi, chi, theta):
        return 1j * self.d3F(phi, chi, theta)

    def d4q(self, phi, chi, theta):
        return 1j * self.d4F(phi, chi, theta)

    def phi0(self, chi, theta, chi0):
        chi, theta, chi0 = self._broadcast(chi, theta, chi0)

        k, b = self.k(theta), self.b(theta)

        chi1 = 1.0 / (k + self.a0**2*b)
        ratio = self.a0**2 * chi * b / (1.0 - chi*k)

        phi0 = np.empty(chi.shape, dtype=np.complex128)

        m1 = chi <= chi1
        m2 = (chi > chi1) & (chi < chi0)
        m3 = chi >= chi0

        if np.any(m1):
            phi0[m1] = 1j*np.sqrt(np.abs(np.log(ratio[m1])))

        if np.any(m2):
            phi0[m2] = np.sqrt(np.log(ratio[m2]))

        if np.any(m3):
            chi_m = chi[m3]
            theta_m = theta[m3]
            k_m = k[m3]
            b_m = b[m3]

            bar_phi = np.sqrt(0.5 * lambertw(-2.0 * (self.T * self.a0 ** 2 * b_m / k_m)**2 + 0j))
            
            denom = (1.0 - chi_m * k_m + self.a0**2 * chi_m * b_m * np.exp(-bar_phi**2 - (1.0 - k_m * chi_m)**2))

            tilde_phi0 = np.sqrt(np.log(self.a0**2 * chi_m * b_m / denom))
            d_tilde_phi0 = -self.d1q(tilde_phi0, chi_m, theta_m) / self.d2q(tilde_phi0, chi_m, theta_m)
            
            phi0[m3] = 1*tilde_phi0 + 1*d_tilde_phi0
    
        return phi0

    def I_1(self, chi, theta, chi0):
        chi, theta, chi0 = self._broadcast(chi, theta, chi0)
        
        phi0 = self.phi0(chi, theta, chi0)
        I_1 = np.zeros_like(chi, dtype=np.float64)

        m1 = chi <= chi0
        if np.any(m1):
            chi1 = chi[m1]
            theta1 = theta[m1]
            phi1 = phi0[m1]

            x0 = 1.5 * self.T * 1j * self.F(phi1, chi1, theta1)
            x2 = 1.5 * self.T * np.abs(self.d2F(phi1, chi1, theta1))
            y = (np.abs(x0)**(2.0 / 3.0) * np.exp(2j * np.angle(x0))).real

            y0 = -1.173713222709128
            Ai, _, Bi, _ = airy(y)
            AiSm = Ai.copy()

            mask = y < y0
            AiSm[mask] = np.hypot(Ai[mask], Bi[mask])

            I_1[m1] = (2.0 * np.pi * np.sqrt(3.0) * np.sqrt(np.abs(x0)**(1.0 / 3.0) / x2) * AiSm)**2
            
        m2 = chi > chi0
        if np.any(m2):
            chi2 = chi[m2]
            theta2 = theta[m2]
            phi2 = phi0[m2]

            F2 = self.F(phi2, chi2, theta2)
            d2q2 = self.d2q(phi2, chi2, theta2)
            d3q2 = self.d3q(phi2, chi2, theta2)
            d4q2 = self.d4q(phi2, chi2, theta2)

            Sigma = 1.0 + (d4q2 / (8.0 * d2q2**2) - 5.0 * d3q2**2 / (24.0 * d2q2**3)) / self.T
            Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=1e10, neginf=-1e10)
            # exp(T*Re(iF)) легко переполняет float64; без ограничения дальше pref*I_1 даёт inf и ∫∫ становится nan
            expo = np.asarray(self.T * np.real(1j * F2), dtype=np.float64)
            expo = np.clip(expo, -700.0, 700.0)
            pref_m2 = np.sqrt(8.0 * np.pi / (self.T * np.abs(d2q2))) * np.abs(Sigma)
            pref_m2 = np.nan_to_num(pref_m2, nan=0.0, posinf=1e200, neginf=0.0)
            I_1[m2] = np.clip((pref_m2 * np.exp(expo)) ** 2, 0.0, 1e300)

        return np.nan_to_num(I_1, nan=0.0, posinf=0.0, neginf=0.0)

    def d2E(self, chi, theta, chi0):
        chi, theta, chi0 = self._broadcast(chi, theta, chi0)

        I_1 = self.I_1(chi, theta, chi0)
        a_b = self.alpha_bar(chi, theta)
        phi0 = self.phi0(chi, theta, chi0)

        g_phi0 = np.exp(-(phi0**2).real / 2)
        z = a_b * np.abs(g_phi0)
        
        pref = -np.abs(jv(1, z)) ** 2 + np.abs(g_phi0) ** 2 * self.a0 ** 2 / 2 * (np.abs(jv(0, z)) ** 2 + np.abs(jv(2, z)) ** 2 - 2 * np.abs(jv(1, z)) ** 2)
        pref = np.nan_to_num(pref, nan=0.0, posinf=1e200, neginf=-1e200)
        d2E = self.alpha * self.T**2 * chi**2 * self.p_minus**2 / (4.0 * np.pi**2) * pref * I_1
        return np.nan_to_num(d2E, nan=0.0, posinf=0.0, neginf=0.0)
        # return I_1