import numpy as np 
from scipy.integrate import solve_ivp
import time
from matrius import *
from terme_no_lineal import * 
from terme_forces import *

def Iniciar_programa(alpha0, Nx, Ny, Lx, Re, T, n_steps):
    """
    Inicia el programa, computa les matrius i termes de força i fa correr l'integrador numeric.
    Input:
        - alpha0 (vector de dades inicial)
        - Nx, Ny, Lx, Re (harmonics, longitud domini i Re)
        - T (temps final)
        - Numero de frames

    Output:
        -  vector solucio alpha. 
    """

    Ny_integration = 2 * (Ny + 4)

    matrius_lineal = Carregar_Matrius_Terme_lineal(Nx, Ny, Lx)
    matrius_terme_no_lineal = Carregar_Matrius_Terme_no_lineal(Ny, Ny_integration)

    Bk = np.array(matrius_lineal['B_k'])
    Ak_inv = np.array(matrius_lineal['Ainv_k'])
    A_bar = matrius_lineal['A_bar']
    
    Terme_forces = F_gravity(Nx, Ny, 2 * np.pi, np.pi/2)

    def ode_system(t, alpha_flat):
        
        start_time = time.time()
        
        alpha = alpha_flat.reshape(Ny, Nx) 

        Terme_lineal = np.stack([1/Re * A_bar[i] @ alpha[:, i] for i in range(Nx)], axis=1)

        F_non_lineal_padded = F_term_padded(alpha, Lx, matrius_terme_no_lineal)
        Terme_no_lineal = np.stack([Ak_inv[i] @ F_non_lineal_padded[:, i] for i in range(Nx)], axis=1)
        #Terme_no_lineal[np.abs(Terme_no_lineal) < 10**(-8)] = 0 (matriu de filtratge, per a Re alts.)
        
        F_value_return = Terme_lineal + Terme_no_lineal + Terme_forces

        elapsed_time = time.time() - start_time
        print(f"ODE System Call: t = {t:.5f}, Time = {elapsed_time:.5f} sec")

        return (F_value_return).flatten() 

    alpha = alpha0
    t_span = (0, T)
    
    solution = solve_ivp(ode_system, t_span, alpha.flatten(), method='RK45', t_eval=np.linspace(0, T, n_steps), rtol = 1e-8 , atol = 1e-11)

    alpha_solution_history_reshaped = np.array([alpha_i.reshape(Ny, Nx) for alpha_i in solution.y.T])

    return alpha_solution_history_reshaped