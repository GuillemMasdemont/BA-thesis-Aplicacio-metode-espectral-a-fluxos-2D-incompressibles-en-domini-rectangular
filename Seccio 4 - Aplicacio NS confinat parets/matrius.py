
from utils import a_ij, b_ij, h_j, df_j, f_j, g_j, dg_j, P_j
import numpy as np 
import os

# --------------------------
# Crida les matrius, del terme lineal, no lineal i de diferenciacio. Veure l'esquema de la Figura 10 a la memoria. 
# --------------------------
def Carregar_Matrius_Terme_lineal(Nx, Ny, Lx): 
    kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)  

    fitxer = f'Fitxers_interns/Terme_lineal/Matrius_Cas_Nx_{Nx}_Ny_{Ny}'
    os.makedirs(fitxer, exist_ok=True)

    fitxer_matriu_A = f'{fitxer}/Matrius_A_k_Nx_{Nx}_Ny{Ny}.npy'
    fitxer_matriu_B = f'{fitxer}/Matrius_B_k_Nx_{Nx}_Ny{Ny}.npy'
    fitxer_matriu_A_inversa = f'{fitxer}/Matrius_A_inverses_k_Nx_{Nx}Ny_{Ny}.npy'
    fitxer_matriu_A_bar = f'{fitxer}/Matrius_A_bar_k_Nx_{Nx}Ny_{Ny}.npy'

    matrius_lineal = {}
    if os.path.exists(fitxer_matriu_A) and os.path.exists(fitxer_matriu_B) and os.path.exists(fitxer_matriu_A_inversa) and os.path.exists(fitxer_matriu_A_bar) :
        print("Matrius del terme lineal carregades.")
        matrius_lineal['A_k'] = np.load(fitxer_matriu_A, allow_pickle=True)
        matrius_lineal['B_k'] = np.load(fitxer_matriu_B, allow_pickle=True)
        matrius_lineal['Ainv_k'] = np.load(fitxer_matriu_A_inversa, allow_pickle=True)
        matrius_lineal['A_bar'] = np.load(fitxer_matriu_A_bar, allow_pickle=True)

    else:
        print("Calculant matrius del terme lineal...")

        kx = np.fft.fftfreq(Nx, d=(Lx / Nx)) * (2 * np.pi)

        matrius_lineal['A_k'] = np.array([[[a_ij(i, j, k) for j in range(Ny)] for i in range(Ny)] for k in kx])
        matrius_lineal['B_k'] = np.array([[[b_ij(i, j, k) for j in range(Ny)] for i in range(Ny)] for k in kx])
        matrius_lineal['Ainv_k'] = np.array([np.linalg.inv(matrius_lineal['A_k'][k]) for k in range(len(kx))])
        matrius_lineal['A_bar'] = matrius_lineal['Ainv_k'] @ matrius_lineal['B_k']

        np.save(fitxer_matriu_A, matrius_lineal['A_k'])
        np.save(fitxer_matriu_B, matrius_lineal['B_k'])
        np.save(fitxer_matriu_A_inversa, matrius_lineal['Ainv_k'])
        np.save(fitxer_matriu_A_bar, matrius_lineal['A_bar'])

        print("Matrius del terme lineal carregades.")
    
    return matrius_lineal


def Carregar_Matrius_Conversio_Alpha_u(Ny_alpha, Ny): 
    fitxer = f'Fitxers_interns/Terme_no_lineal/Matrius_de_Conversio_Cas_Ny_{Ny}'
    os.makedirs(fitxer, exist_ok=True)

    fitxer_matriu_conversio_alpha_u_k_nul = f'{fitxer}/Matriu_conversio_alpha_u_k_nul_Ny_{Ny}.npy'
    fitxer_matriu_conversio_alpha_u_k_no_nul = f'{fitxer}/Matriu_conversio_alpha_u_k_no_nul_Ny_{Ny}.npy'
    fitxer_matriu_conversio_alpha_v = f'{fitxer}/Matriu_conversio_alpha_v_Ny_{Ny}.npy'

    if os.path.exists(fitxer_matriu_conversio_alpha_u_k_nul) and os.path.exists(fitxer_matriu_conversio_alpha_u_k_no_nul) and os.path.exists(fitxer_matriu_conversio_alpha_v):
        print("Matrius conversio alpha a u carregades.")
        matriu_conversio_alpha_u_k_nul = np.load(fitxer_matriu_conversio_alpha_u_k_nul, allow_pickle=True)
        matriu_conversio_alpha_u_k_no_nul = np.load(fitxer_matriu_conversio_alpha_u_k_no_nul, allow_pickle=True)
        matriu_conversio_alpha_v = np.load(fitxer_matriu_conversio_alpha_v, allow_pickle=True)

    else: 
        print("Calculant matrius conversio alpha a u.")
        
        punts_y = np.cos(np.pi * np.arange(Ny) / (Ny-1))

        matriu_conversio_alpha_u_k_nul = np.zeros((Ny, Ny_alpha), dtype=complex)
        for j in range(Ny_alpha):
            matriu_conversio_alpha_u_k_nul[:,j] = h_j(punts_y, j)

        matriu_conversio_alpha_u_k_no_nul = np.zeros((Ny, Ny_alpha), dtype=complex)
        for j in range(Ny_alpha):
            matriu_conversio_alpha_u_k_no_nul[:,j] = 1j * df_j(punts_y, j)

        matriu_conversio_alpha_v = np.zeros((Ny, Ny_alpha), dtype=complex)
        for j in range(Ny_alpha):
            matriu_conversio_alpha_v[:,j] = f_j(punts_y, j)

        np.save(fitxer_matriu_conversio_alpha_u_k_nul, matriu_conversio_alpha_u_k_nul)
        np.save(fitxer_matriu_conversio_alpha_u_k_no_nul, matriu_conversio_alpha_u_k_no_nul)
        np.save(fitxer_matriu_conversio_alpha_v, matriu_conversio_alpha_v)

        print("Matrius conversio alpha a u carregades.")

    matrius_conversio = {}
    matrius_conversio['alpha_u_k_nul'] = matriu_conversio_alpha_u_k_nul
    matrius_conversio['alpha_u_k_no_nul'] = matriu_conversio_alpha_u_k_no_nul
    matrius_conversio['alpha_v'] = matriu_conversio_alpha_v

    return  matrius_conversio

def Integration_matrix(Ny): 
    fitxer = f'Fitxers_interns/Terme_no_lineal/Matrius_de_integracio_Cas_Ny_{Ny}'
    os.makedirs(fitxer, exist_ok=True)

    fitxer_integracio_primera_coordenada_k_nul = f'{fitxer}/Matriu_integracio_primera_coordenada_k_nul_Ny_{Ny}.npy'
    fitxer_integracio_primera_coordenada_k_no_nul = f'{fitxer}/Matriu_integracio_primera_coordenada_k_no_nul_Ny_{Ny}.npy'
    fitxer_integracio_segona_coordenada = f'{fitxer}/Matriu_integracio_segona_coordenada_Ny_{Ny}.npy'

    if os.path.exists(fitxer_integracio_primera_coordenada_k_nul) and os.path.exists(fitxer_integracio_primera_coordenada_k_no_nul) and os.path.exists(fitxer_integracio_segona_coordenada):
        print("Matrius integracio alpha a u carregades.")
        integracio_primera_coordenada_k_nul = np.load(fitxer_integracio_primera_coordenada_k_nul, allow_pickle=True)
        integracio_primera_coordenada_k_no_nul = np.load(fitxer_integracio_primera_coordenada_k_no_nul, allow_pickle=True)
        integracio_segona_coordenada = np.load(fitxer_integracio_segona_coordenada, allow_pickle=True)

    else: 
        print("Calculant matrius conversio alpha a u.")
        Ny_integration = 2 * (Ny + 4)
        punts_y = np.cos(np.pi * np.arange(Ny_integration) / (Ny_integration-1))

        integracio_primera_coordenada_k_nul = np.array([P_j(punts_y, j) for j in range(0, Ny)])
        integracio_primera_coordenada_k_no_nul = np.array([-1j * dg_j(punts_y, j) for j in range(0, Ny)])
        integracio_segona_coordenada = np.array([g_j(punts_y, i) for i in range(0, Ny)])

        np.save(fitxer_integracio_primera_coordenada_k_nul, integracio_primera_coordenada_k_nul)
        np.save(fitxer_integracio_primera_coordenada_k_no_nul, integracio_primera_coordenada_k_no_nul)
        np.save(fitxer_integracio_segona_coordenada, integracio_segona_coordenada)
        
        print("Matrius integracio alpha a u carregades.")

    matrius_integracio = {}
    matrius_integracio['primera_coordenada_k_nul'] = integracio_primera_coordenada_k_nul
    matrius_integracio['primera_coordenada_k_no_nul'] = integracio_primera_coordenada_k_no_nul
    matrius_integracio['segona_coordenada'] = integracio_segona_coordenada

    return matrius_integracio

def Matriu_diferenciacio_Txebitxev(Ny): 
    def chebyshev_diff_matrix(N):
        if N == 0:
            return np.array([[0]]), np.array([0])

        x = np.cos(np.pi * np.arange(N+1) / N)
        
        c = np.ones(N+1)
        c[0] = c[N] = 2

        D = np.zeros((N+1, N+1))

        for i in range(N+1):
            for j in range(N+1):
                if i != j:
                    D[i, j] = (-1)**(i+j) * c[i] / (c[j] * (x[i] - x[j]))
                else:
                    if i == 0:
                        D[i, j] = (2*N**2 + 1) / 6
                    elif i == N:
                        D[i, j] = -(2*N**2 + 1) / 6
                    else:
                        D[i, j] = -x[i] / (2 * (1 - x[i]**2))
        
        return D, x

    Ny_integration = 2 * (Ny + 4)
    
    print('Matriu derivacio Txebitxev carregada.')

    return chebyshev_diff_matrix(Ny_integration-1)[0] 


def Carregar_Matrius_Terme_no_lineal(Ny, Ny_integration): 
    matrius_terme_no_lineal = {
        'matrius_conversio': Carregar_Matrius_Conversio_Alpha_u(Ny, Ny_integration),
        'matrius_integracio': Integration_matrix(Ny),
        'matriu_diferenciacio': Matriu_diferenciacio_Txebitxev(Ny)
    }

    return matrius_terme_no_lineal