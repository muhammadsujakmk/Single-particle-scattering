import numpy as np
from scipy import special
import mpmath as mp

def Dn(z,n):
    jn = special.spherical_jn(n,z)
    jnd = special.spherical_jn(n,z, derivative=True)
    
    psi = z * jn
    psid = jn + z * jnd
    return psid/psi 

def ricabes1(v,z):
    jn = np.sqrt(0.5*np.pi/z) * mp.besselj(v+0.5,z, 0)
    jnl = np.sqrt(0.5*np.pi/z)* mp.besselj(v-0.5, z,0)
    
    psi1 = z * jn
    psid = z * jnl - v * jn # derive from analytic
    psi2 = psid + psi1 * v/z

    return psi1, psi2

def ricabes3(v,z):
    jn = np.sqrt(0.5*np.pi/z)*mp.besselj(v+0.5,z, 0)
    yn = np.sqrt(0.5*np.pi/z)*mp.bessely(v+0.5,z, 0)
    hn = jn + 1j*yn
    
    jnl = np.sqrt(0.5*np.pi/z)*mp.besselj(v-0.5,z, 0)
    ynl = np.sqrt(0.5*np.pi/z)*mp.bessely(v-0.5,z, 0)
    hnl = jnl + 1j*ynl

    zeta1 = z * hn
    zetad = z * hnl - v * hn
    zeta2 = zetad + zeta1 * v/z

    return zeta1, zeta2

def coef_ab(x,m):
    res_a1 = (m**2-1)/3
    res_a11 = 1-(x**2/10)+(4 * m**2 + 5) * x**4/1400
    
    res_D3 = (8 * m**4 - 385 * m**2 + 350) * x**4 / 1400
    res_D4 = 2j * ( m**2 - 1 ) * x**3/3 * (1 - x**2/10)
    D = m**2 + 2 + (1-(7 * m**2)/10) * x**2 - res_D3 + res_D4
    a1 = 2j*res_a1*res_a11/D

    res_b1 = 1+(2 * m**2 - 5) * x**2/70
    res_b11 = 1-(2 * m**2 - 5) * x**2/30
    b1 = 1j * x**2 * (m**2 - 1)/45 * res_b1/res_b11

    res_a2 = 1 - x**2/14
    res_a22 = 2 * m**2 + 3 - (2 * m**2 -7) * x**2/14
    a2 = 1j * x**2 * (m**2-1)/15 * res_a2/res_a22

    return a1,b1,a2


