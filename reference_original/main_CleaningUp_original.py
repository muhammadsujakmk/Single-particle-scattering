import numpy as np
import matplotlib.pyplot as plt
import recmethod as rc
import matimport as mat

## Unit 
nm = 1e-9


def main():
    n_med = 1                           # Refractive index of medium
    wvl = np.linspace(400,1000,601)*nm    # wavelength number
    r = 100*nm                        # Sphere radius
    Q = calculate_cross_section(wvl,n_med,r)
    #plot_results(wvl,Q, r,unit=[nm,"nm"])
    plot_Modes(wvl,Q, r,unit=[nm,"nm"])

def N_det(x):
    for i in range(x.size):
        if 0.02<=x[i]<=8:
            return int(x[i]+4 * x[i]**(1/3)+1)
        elif 8<x[i]<4200:
            return int(x[i]+4.05 * x[i]**(1/3)+2)
        elif 4200<=x[i]<=20000:
            return int(x[i]+4 * x[i]**(1/3)+2)

def electric_magnetic_modes(k0,j,a,b):
    sig_ext_E =(2*np.pi/(k0)**2) * (2*j+1)*a.real
    sig_ext_M =(2*np.pi/(k0)**2) * (2*j+1)*b.real
    sig_sca_E =(2*np.pi/(k0)**2) * (2*j+1)*abs(a)**2
    sig_sca_M =(2*np.pi/(k0)**2) * (2*j+1)*abs(b)**2
    sig_abs_E = sig_ext_E - sig_sca_E
    sig_abs_M = sig_ext_M - sig_sca_M
    
    return sig_ext_E,sig_ext_M, sig_sca_E,sig_sca_M, sig_abs_E, sig_abs_M

def calculate_cross_section(wvl,n_med,r_in):
    Q = np.zeros((wvl.size,21))
    for i in range(wvl.size):
        mat_core = mat.mat_cal(wvl[i]/nm,"\Si_Green.txt")
        k0 = 2*n_med*np.pi/wvl[i]
        x1 = k0*r_in
        m1 = mat_core / n_med
        N = 4#N_det(x1) 
        for j in range(1,N):
            psi_n, psi_n1 = rc.ricabes1(j, x1)
            zeta_n, zeta_n1 = rc.ricabes3(j, x1)

            D = rc.Dn(m1*x1, j)
            anum = (D/m1 + j/x1) * psi_n - psi_n1
            aden = (D/m1 + j/x1) * zeta_n - zeta_n1
            a = anum/aden
            
            bnum = (m1*D + j/x1) * psi_n - psi_n1
            bden = (m1*D + j/x1) * zeta_n - zeta_n1
            b = bnum/bden

            sig_ext =(2*np.pi/(k0)**2) * (2*j+1)*(a+b).real
            sig_sca =(2*np.pi/(k0)**2) * (2*j+1)*(abs(a)**2+abs(b)**2)
            sig_abs = sig_ext - sig_sca
            Q[i,0] += sig_ext
            Q[i,1] += sig_sca
            Q[i,2] += sig_abs

            if j==1: #Dipole
                ext_E, ext_M, sca_E, sca_M, abs_E, abs_M = electric_magnetic_modes(k0,j,a,b)
                Q[i,3] += ext_E
                Q[i,4] += ext_M
                Q[i,5] += sca_E
                Q[i,6] += sca_M
                Q[i,7] += abs_E
                Q[i,8] += abs_M
            elif j==2: #Quadrupole 
                ext_E, ext_M, sca_E, sca_M, abs_E, abs_M = electric_magnetic_modes(k0,j,a,b)
                Q[i,9] += ext_E
                Q[i,10] += ext_M
                Q[i,11] += sca_E
                Q[i,12] += sca_M
                Q[i,13] += abs_E
                Q[i,14] += abs_M
            elif j==3: #Octupole
                ext_E, ext_M, sca_E, sca_M, abs_E, abs_M = electric_magnetic_modes(k0,j,a,b)
                Q[i,15] += ext_E
                Q[i,16] += ext_M
                Q[i,17] += sca_E
                Q[i,18] += sca_M
                Q[i,19] += abs_E
                Q[i,20] += abs_M

    return Q

def plot_results(wvl,Q, r,unit=[1e-9,"nm"]):
    area_sph = np.pi*r**2 #Geometrical cross section
    
    plt.xlabel(f'Wavelength ({unit[1]})')
    plt.ylabel('Efficiency, $Q$')
    plt.plot(wvl/unit[0],Q[:,0] / area_sph, label="$Q_{ext}$")
    plt.plot(wvl/unit[0],Q[:,1] / area_sph, label="$Q_{sca}$")
    plt.plot(wvl/unit[0],Q[:,2] / area_sph, label="$Q_{abs}$")
    plt.legend(frameon=False, fontsize=14) 
    plt.show()

def plot_Modes(wvl,Q, r,unit=[1e-9,"nm"]):
    area_sph = np.pi*r**2 #Geometrical cross section
    
    plt.xlabel(f'Wavelength ({unit[1]})')
    plt.ylabel('Efficiency, $Q$')
    plt.plot(wvl/unit[0],Q[:,1] / area_sph, label="$Q_{sca}$")
    plt.plot(wvl/unit[0],Q[:,5] / area_sph, label="$Q_{ED}$")
    plt.plot(wvl/unit[0],Q[:,6] / area_sph, label="$Q_{MD}$")
    plt.plot(wvl/unit[0],Q[:,11] / area_sph, label="$Q_{EQ}$")
    plt.plot(wvl/unit[0],Q[:,12] / area_sph, label="$Q_{MQ}$")
    plt.plot(wvl/unit[0],Q[:,17] / area_sph, label="$Q_{EO}$")
    plt.plot(wvl/unit[0],Q[:,18] / area_sph, label="$Q_{MO}$")
    plt.legend(frameon=False, fontsize=14) 
    plt.show()

main()



