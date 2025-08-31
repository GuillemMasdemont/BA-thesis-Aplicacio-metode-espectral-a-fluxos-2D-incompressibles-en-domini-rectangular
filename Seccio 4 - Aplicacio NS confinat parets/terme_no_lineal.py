import numpy as np 

def F_term_padded(alpha, Lx, matrius_terme_no_lineal): 
    """
    Calcul del terme no lineal amb regla 3/2. 
    Input:
        - quoeficients alpha (matriu)
        - Lx (domini)
        - matrius_terme_no_lineal (matriu de transformacio)

    Output:
        - terme no lineal (a l'espai de fourier)
    """
    matriu_int_primera_coordenada_k_nul = matrius_terme_no_lineal['matrius_integracio']['primera_coordenada_k_nul']
    matriu_int_primera_coordenada_k_no_nul = matrius_terme_no_lineal['matrius_integracio']['primera_coordenada_k_no_nul']
    matriu_int_segona_coordenada  = matrius_terme_no_lineal['matrius_integracio']['segona_coordenada']
    D = matrius_terme_no_lineal['matriu_diferenciacio']
    
    Ny, Nx = alpha.shape
    Ny_integration = 2 * (Ny + 4)

    kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)  

    w = np.full(Ny_integration, np.pi / Ny_integration)  
    w[0] = w[-1] = np.pi / (2*Ny_integration)

    _, _, u, v = alpha_to_u_efficient(alpha, Ny_integration, Lx, matrius_terme_no_lineal)
    
    u_hat = np.fft.fft(u, axis=1) 
    v_hat = np.fft.fft(v, axis=1) 

    v_x_hat = 1j * kx * v_hat
    u_y_hat = np.fft.fft(D @ u, axis = 1)    
    
    non_lin_x_fft = nonlinear_term_uv_hat_x(v_hat, v_x_hat, Ny_integration, Nx) - nonlinear_term_uv_hat_x(v_hat, u_y_hat, Ny_integration, Nx)  
    non_lin_y_fft = - nonlinear_term_uv_hat_x(u_hat, v_x_hat, Ny_integration, Nx) + nonlinear_term_uv_hat_x(u_hat, u_y_hat, Ny_integration, Nx)  

    f_term_u = np.zeros((Ny, Nx), dtype=complex)
    f_term_u[:, 0] = matriu_int_primera_coordenada_k_nul @ (w * non_lin_x_fft[:, 0])
    f_term_u[:, 1:] = matriu_int_primera_coordenada_k_no_nul @ (w[:, np.newaxis] * non_lin_x_fft[:, 1:])

    f_term_v = matriu_int_segona_coordenada @ (w[:, np.newaxis] * (non_lin_y_fft * kx))

    return f_term_u + f_term_v


#3/2 - rule definition: 

def nonlinear_term_uv_hat_x(u_hat, v_hat, Ny, Nx):
    """
    Calcul del terme no lineal amb regla 3/2. 
    Input:
        - matrius de velocitat u_hat, v_hat (a l'espai de fourier)
        - Nx_plot, Ny_plot (harmonics)

    Output:
        - terme no lineal (a l'espai de fourier)
    """
    Mx, My = (3 * Nx) // 2, (3 * Ny) // 2  # New padded resolution

    u_hat_pad = np.zeros((Ny, Mx), dtype=complex)  
    v_hat_pad = np.zeros((Ny, Mx), dtype=complex)  

    u_hat_pad[:, :Nx//2] = u_hat[:, :Nx//2]  
    u_hat_pad[:, -Nx//2:] = u_hat[:, -Nx//2:]

    v_hat_pad[:, :Nx//2] = v_hat[:, :Nx//2]  
    v_hat_pad[:, -Nx//2:] = v_hat[:, -Nx//2:]

    u_pad = np.fft.ifft(u_hat_pad, axis = 1).real * (Mx / Nx) 
    v_pad = np.fft.ifft(v_hat_pad, axis = 1).real * (Mx / Nx) 

    nonlinear_term_pad = np.fft.fft(u_pad * v_pad, axis = 1) * (1 / Mx) 


    nonlinear_term = np.zeros((Ny, Nx), dtype=complex)
    nonlinear_term[:, :Nx//2] = nonlinear_term_pad[:, :Nx//2]  
    nonlinear_term[:, -Nx//2:] = nonlinear_term_pad[:, -Nx//2:]

    return nonlinear_term



#Convert alpha to u efficiently:  

def alpha_to_u_efficient(alpha, Ny_integration, Lx, matrius_terme_no_lineal): 
    """
    Transforma la solucio de l'espai espectral a l'espai inicial de forma eficient 
    Input:
        - alpha (matriu de quoeficients)
        - Nx_plot, Ny_plot (harmonics)
        - Lx (domini) 
        - matrius_terme_no_lineal (matrius de trasnformacio)

    Output:
        - retorna el mallat de punts
    """    
    T_alpha_u_k_nul = matrius_terme_no_lineal['matrius_conversio']['alpha_u_k_nul']
    T_alpha_u_k_no_nul = matrius_terme_no_lineal['matrius_conversio']['alpha_u_k_no_nul']
    T_alpha_v = matrius_terme_no_lineal['matrius_conversio']['alpha_v']

    _ , Nx = alpha.shape

    points_x = np.linspace(0, Lx, Nx, endpoint = False)
    points_y = np.cos(np.pi * np.arange(Ny_integration) / (Ny_integration-1))

    kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)

    X,Y = np.meshgrid(points_x, points_y)

    u_hat = np.zeros((Ny_integration , Nx), dtype = complex)
    v_hat = np.zeros((Ny_integration , Nx), dtype = complex)

    u_hat[:, 0] = T_alpha_u_k_nul @ alpha[:,0]
    u_hat[:, 1:] = T_alpha_u_k_no_nul @ alpha[:, 1:]

    alpha_v = alpha * kx
    v_hat = T_alpha_v @ alpha_v

    u = Nx * np.fft.ifft(u_hat, axis = 1).real + points_y[:, np.newaxis] * np.ones((Ny_integration ,Nx), dtype = complex) #New term from moving walls. 
    v = Nx * np.fft.ifft(v_hat, axis = 1).real

    return X, Y, u, v 


def Matrix_filter(matrix, tol):
    #Matriu de filtratge espectral
    F = matrix.copy()
    F[np.abs(F) < tol] = 0
    return F



