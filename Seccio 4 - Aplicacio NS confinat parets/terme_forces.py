import numpy as np 
import numpy.polynomial.chebyshev as np_cheb
from utils import *

# --------------------------
# Terme forçant descrit a la seccio 4.1.2
# --------------------------
def F_gravity(Nx, Ny, Lx, theta):
    """
    Input:
        - Nx, Ny (numero d'harmonics)
        - Lx (domini)
        - Theta (angle del vector gravetat, en radiants, respecte la vertical)

    Output:
        - terme espectral per la força externa. 
    """

    N = 4 * ((Ny + 4))  
    kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)
    
    points_x = np.linspace(0, Lx, Nx, endpoint = False)
    points_y, weights = np_cheb.chebgauss(N)

    #Cas gravetat:
    Fx_gravity = np.array([[np.cos(theta) for _ in range(len(kx))] for y in points_y])
    Fx_gravity_fft = np.fft.fft(Fx_gravity, axis = 1) / len(kx)

    Fy_gravity = np.array([[np.sin(theta) for _ in range(len(kx))] for y in points_y])
    Fy_gravity_fft = np.fft.fft(Fy_gravity, axis = 1) / len(kx)

    #Cas un terme forçant inversemblant:
    #X_points, Y_points = np.meshgrid(points_x, points_y)
    #Theta_points = np.arctan2(Y_points, X_points - np.pi)
    
    #Fx_gravity = - np.sin(Theta_points)
    #Fx_gravity_fft = np.fft.fft(Fx_gravity, axis = 1) / Nx

    #Fy_gravity = np.cos(Theta_points)
    #Fy_gravity_fft = np.fft.fft(Fy_gravity, axis = 1) / Nx

    F_spectral = []
    for i in range(Ny):
        #Entrades matriu F_ij = <phi_ij, F>
        R = [] 
        for j, k in enumerate(kx):         
            if k == 0:
                points, weights = np_cheb.chebgauss(N)
                f_points  = Fx_gravity_fft[:,j] * P_j(points,i) + Fy_gravity_fft[:,j] * 0

            else: 
                points, weights = np_cheb.chebgauss(N)
                f_points = -1j * Fy_gravity_fft[:,j] * dg_j_b(points,i) + Fy_gravity_fft[:,j] * k * g_j(points,i)
            
            if np.abs(np.sum(f_points * weights)) < 10**(-7): 
                S =  0 
        
            else: 
                S = np.sum(f_points * weights)  

            R.append(S)

        F_spectral.append(R)

    F_spectral = np.array(F_spectral)
    F_spectral[abs(F_spectral < 10**(-10))] = 0 

    return F_spectral 