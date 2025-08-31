#Funcions de dibuix
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
import matplotlib.colors as colors
from utils import *


params = {
    'legend.fontsize': 15,
    'legend.loc': 'best',
    'figure.figsize': (14, 5),
    'lines.markerfacecolor': 'none',
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20, 
    'ytick.labelsize': 20, 
    'grid.alpha': 0.6,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}'
}

plt.rcParams.update(params)





def reconstruct_velocity_field(alpha, Nx, Ny, Lx):
    """
    Calcula u i v a partir d'alpha en un mallat seguint Fourier–Txev

    Input:
        - alpha (matriu)
        - Nx, Ny (nombre de harmonics en x i y)
        - Lx (longitud, 2*np.pi)

    Output:
        - X, Y (matrius de punts)
        - u, v (solució en cada punt de la graella)
    """


    x = np.linspace(0, Lx, Nx)
    y = np.cos(np.pi * np.arange(Ny) / (Ny - 1))
    X, Y = np.meshgrid(x, y)
    u = np.zeros_like(X, dtype=complex) + Y #Y es el perfil base, sol. particular, tal i com es descriu al treball 
    v = np.zeros_like(X, dtype=complex)


    Nj, Nk = alpha.shape  
    kx = np.fft.fftfreq(Nk, d=(Lx / Nk)) * (2 * np.pi)
    for j in range(Nj):  
        h_j_k0 = h_j(Y, j) 
        df_j_k = df_j(Y, j)
        f_j_k = f_j(Y, j)

        for k in range(Nk): 
            #Aplica la transformació: alpha * funció base (matriu de la graella).
            if np.isclose(kx[k], 0):  
                u += alpha[j, k] * h_j_k0 
            else: 
                exp_term = np.exp(1j * kx[k] * X)
                u += alpha[j, k] * 1j * df_j_k  * exp_term
                v += alpha[j, k] * kx[k] * f_j_k * exp_term
                
    return X, Y, u.real, v.real 



def plot_velocity(alpha, Nx_plot, Ny_plot): 
    """
    Dibuixa el perfil de velocitat donat la matriu alpha i un mallat (Nx_plot, Ny_plot)
    """
    X, Y, u_final, v_final = reconstruct_velocity_field(alpha, Nx_plot, Ny_plot, 2 * np.pi)
    velocity_magnitude_0 = np.sqrt(u_final**2 + v_final**2)  
    plt.figure(figsize = (12,4))
    plt.quiver(X, Y, u_final, v_final, velocity_magnitude_0, scale_units = 'xy', cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Condició Inicial')
    plt.colorbar()
    plt.show()


def reconstruct_last_velocity_field_efficient_equidistant(alpha, Nx_plot, Ny_plot, Lx): 
    """
    Dibuixa el perfil de velocitat donat la matriu alpha i un mallat amb punts equidistant (Nx_plot, Ny_plot) 
    """
    Ny, Nx = alpha.shape 

    points_y = np.linspace(-1, 1, Ny_plot)
    points_x = np.linspace(0, Lx, Nx_plot, endpoint = False)

    #Transformacio espectral 
    Phi_u_other_col = np.zeros((Ny_plot, Ny), dtype=complex)
    for j in range(Ny):
        Phi_u_other_col[:,j] = 1j * df_j(points_y, j)

    Phi_u_first_col = np.zeros((Ny_plot, Ny), dtype=complex)
    for j in range(Ny):
        Phi_u_first_col[:,j] = h_j(points_y, j)

    Phi_Y_v = np.zeros((Ny_plot, Ny), dtype=complex)
    for j in range(Ny):
        Phi_Y_v[:,j] = f_j(points_y, j)


    Phi_X = np.zeros((Nx, Nx_plot), dtype = complex)
    kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)

    for j, k in enumerate(kx): 
        Phi_X[j, :] = np.exp(1j * k * points_x)

    X,Y = np.meshgrid(points_x, points_y)

    u_hat = np.zeros((Ny_plot , Nx), dtype = complex)
    v_hat = np.zeros((Ny_plot , Nx), dtype = complex)

    u_hat[:, 0] = Phi_u_first_col @ alpha[:,0]
    u_hat[:, 1:] = Phi_u_other_col @ alpha[:, 1:]

    alpha_v = alpha * kx
    v_hat = Phi_Y_v @ alpha_v

    u = u_hat @ Phi_X + points_y[:, np.newaxis] * np.ones((Ny_plot,Nx_plot), dtype = complex) #Terme corresponent al moviment de les parets
    v = v_hat @ Phi_X

    return X, Y, u.real, v.real 



def plot_velocity_comparison(alpha_0, alpha_1): 
    """
    Dibuixa la comparació de dues solucions alpha.
    Input:
        - alpha0 (matriu)
        - alpha1 (matriu)

    Output:
        - gràfiques dels perfils per alpha0 i alpha1. 
    """

    #Copia per no malmetre les dades. 
    alpha_1 = np.array(alpha_1, copy=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    X, Y, u_final, v_final = reconstruct_velocity_field(alpha_0, 32, 32, 2 * np.pi)
    
    velocity_magnitude_0 = np.sqrt(u_final**2 + v_final**2)  
    quiver1 = axes[0].quiver(X, Y, u_final, v_final, velocity_magnitude_0, scale_units = 'xy', scale = 10, cmap='viridis')
    
    fig.colorbar(quiver1, ax=axes[0], label="mòdul velocitat")
    
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Condició Inicial")

    X, Y, u_final_2, v_final_2 = reconstruct_velocity_field(alpha_1, 32, 32, 2 * np.pi)

    velocity_magnitude_1 = np.sqrt(u_final_2**2 + v_final_2**2)  
    quiver2 = axes[1].quiver(X, Y, u_final_2, v_final_2, velocity_magnitude_1 ,scale_units = 'xy', scale = 10, cmap = 'viridis')
    
    fig.colorbar(quiver2, ax=axes[1], label="mòdul velocitat")
    
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Condició Final")

    plt.tight_layout()  
    plt.show()


def reconstruct_velocity_field_efficient_equidistant(alpha_solution_history_reshaped, Nx_plot, Ny_plot, Lx): 
    """
    Transforma el vector de l'evolucio de coeficients alpha en el vector l'evolucio de posicions. 
    Input:
        - alpha_solutions_history_reshaped (vector de coeficients alpha a cada temps).
        - Nx_plot, Ny_plot, Lx (resolucio)

    Output:
        - vector l'evolucio de posicions per a cada instant de temps. 
    """
    Ny, Nx = alpha_solution_history_reshaped[0].shape 

    points_y = np.linspace(-1, 1, Ny_plot)
    points_x = np.linspace(0, Lx, Nx_plot, endpoint = False)

    Phi_u_other_col = np.zeros((Ny_plot, Ny), dtype=complex)
    for j in range(Ny):
        Phi_u_other_col[:,j] = 1j * df_j(points_y, j)

    Phi_u_first_col = np.zeros((Ny_plot, Ny), dtype=complex)
    for j in range(Ny):
        Phi_u_first_col[:,j] = h_j(points_y, j)

    Phi_Y_v = np.zeros((Ny_plot, Ny), dtype=complex)
    for j in range(Ny):
        Phi_Y_v[:,j] = f_j(points_y, j)


    Phi_X = np.zeros((Nx, Nx_plot), dtype = complex)
    kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)

    for j, k in enumerate(kx): 
        Phi_X[j, :] = np.exp(1j * k * points_x)

    X,Y = np.meshgrid(points_x, points_y)

    u_list = []
    v_list = []
    for i in range(len(alpha_solution_history_reshaped)): 

        alpha = alpha_solution_history_reshaped[i]

        u_hat = np.zeros((Ny_plot , Nx), dtype = complex)
        v_hat = np.zeros((Ny_plot , Nx), dtype = complex)

        u_hat[:, 0] = Phi_u_first_col @ alpha[:,0]
        u_hat[:, 1:] = Phi_u_other_col @ alpha[:, 1:]

        alpha_v = alpha * kx
        v_hat = Phi_Y_v @ alpha_v

        u = u_hat @ Phi_X + points_y[:, np.newaxis] * np.ones((Ny_plot,Nx_plot), dtype = complex) #New term from moving walls. 
        v = v_hat @ Phi_X

        u_list.append(u.real)
        v_list.append(v.real)

    return X, Y, u_list, v_list 


def Animacio_grafic_vectors_vorticitat(alpha_solution_history_reshaped, Nx_plot, Ny_plot, Lx, fps_animacio): 
    """
    Input:
        - alpha_solutions_history_reshaped (vector de coeficients alpha a cada temps).
        - Nx_plot, Ny_plot, Lx (resolucio)

    Output:
        - Animacio de la velocitat i vorticitat donat per un perfil alpha_solution_history_reshaped. 
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Navier–Stokes 2D, velocitat i vorticitat.', fontsize=14, fontweight='bold')

    X,Y, u_list, v_list = reconstruct_velocity_field_efficient_equidistant(alpha_solution_history_reshaped, Nx_plot, Ny_plot, Lx)

    u_plot = u_list[0]
    v_plot = v_list[0]
    velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)

    quiv = axes[0].quiver(X, Y, u_plot, v_plot, velocity_magnitude, cmap='viridis')

    D_chev, _ = chebyshev_diff_matrix(Ny_plot-1)
    kx = np.fft.fftfreq(Nx_plot, d=(Lx / Nx_plot)) * (2 * np.pi)  

    vx_plot = np.fft.ifft(1j * kx * np.fft.fft(v_plot, axis=1), axis=1).real
    uy_plot = D_chev @ u_plot 
    w_plot = (vx_plot - uy_plot).real

    pcm = axes[1].pcolormesh(X, Y, w_plot, cmap='coolwarm', norm = colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20), shading='auto')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Vorticity Field (w)')
    axes[1].set_xlim([0 - Lx * 0.05, Lx * 1.05])
    axes[1].set_ylim([-1 * 1.05, 1 * 1.05])
    cbar = fig.colorbar(pcm, ax=axes[1])

    def update(frame):

        u_plot = u_list[frame]
        v_plot = v_list[frame]

        vx_plot = np.fft.ifft(1j * kx * np.fft.fft(v_plot, axis=1), axis=1).real
        uy_plot = D_chev @ u_plot 
        w_plot = (vx_plot - uy_plot).real

        velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)
        quiv.set_UVC(u_plot, v_plot)
        quiv.set_array(velocity_magnitude.ravel()) 

        pcm.set_array(w_plot.ravel())

        return quiv, pcm

    ani = animation.FuncAnimation(fig, update, frames = len(u_list), blit=True)
    video_path = "Video.gif"  
    writer = PillowWriter(fps = fps_animacio)
    ani.save(video_path, writer=writer)
    print("Guardat correctament")


def Animacio_grafic_magnitud_vorticitat(alpha_solution_history_reshaped, Nx_plot, Ny_plot, Lx, fps_animacio): 
    """
    Input:
        - alpha_solutions_history_reshaped (vector de coeficients alpha a cada temps).
        - Nx_plot, Ny_plot, Lx (resolucio)

    Output:
        - Animacio del modul de la velocitat i vorticitat donat per un perfil alpha_solution_history_reshaped. 
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
    fig.suptitle('2D Periòdic en x, Dirichlet en y - Navier-Stokes: Mòdul de la Velocitat i Vorticitat', fontsize=14, fontweight='bold')

    X,Y, u_list, v_list = reconstruct_velocity_field_efficient_equidistant(alpha_solution_history_reshaped, Nx_plot, Ny_plot, Lx)

    u_plot = u_list[0]
    v_plot = v_list[0]
    velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)

    vel_levels = np.linspace(0, np.max(velocity_magnitude), 50)
    vel_norm = colors.BoundaryNorm(vel_levels, ncolors=256)

    vel_pcm = axes[0].pcolormesh(X, Y, velocity_magnitude, cmap='viridis', norm=vel_norm, shading='auto')

    axes[0].set_title("Magnitud de velocitats")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    D_chev, _ = chebyshev_diff_matrix(Ny_plot-1)
    kx = np.fft.fftfreq(Nx_plot, d=(Lx / Nx_plot)) * (2 * np.pi)

    vx_plot = np.fft.ifft(1j * kx * np.fft.fft(v_plot, axis=1), axis=1).real
    uy_plot = D_chev @ u_plot 
    w_plot = (vx_plot - uy_plot).real

    vort_norm = colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    vort_pcm = axes[1].pcolormesh(X, Y, w_plot, cmap='coolwarm', norm=vort_norm, shading='auto')
    axes[1].set_title("Camp de vorticitat")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_xlim([0 - Lx * 0.05, Lx * 1.05])
    axes[1].set_ylim([-1 * 1.05, 1 * 1.05])

    # animation update function
    def update(frame):
        u_plot = u_list[frame]
        v_plot = v_list[frame]

        velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)
        vel_pcm.set_array(velocity_magnitude.ravel())

        vx_plot = np.fft.ifft(1j * kx * np.fft.fft(v_plot, axis=1), axis=1).real
        uy_plot = D_chev @ u_plot
        w_plot = (vx_plot - uy_plot).real
        vort_pcm.set_array(w_plot.ravel())

        return vel_pcm, vort_pcm

    ani = animation.FuncAnimation(fig, update, frames=len(u_list), blit=True)
    video_path = "Animacio.gif"
    writer = PillowWriter(fps=fps_animacio)
    ani.save(video_path, writer=writer)
    print("Animacio guardada correctament")