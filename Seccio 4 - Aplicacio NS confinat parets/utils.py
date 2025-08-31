import numpy as np 
import numpy.polynomial.chebyshev as np_cheb


# --------------------------
# UTILS : Definicio funcions base, test i calcul de matrius 
# --------------------------

#Cas k = 0.  
#Definicions seguint Appendix C2. 
def h_j(y, j):
    Tj = np_cheb.Chebyshev.basis(j) 
    return (1 - y**2) * Tj(y) 

def dh_j(y,j):
    Tj = np_cheb.Chebyshev.basis(j) 
    dTj = Tj.deriv()
    return -2 * y * Tj(y) + (1 - y**2) * dTj(y)

def ddh_j(y, j):
    Tj = np_cheb.Chebyshev.basis(j)
    dTj = Tj.deriv()  
    ddTj = dTj.deriv()  
    
    term1 = -2* Tj(y)
    term2 = -4 * y * dTj(y)
    term3 = (1 - y**2) * ddTj(y)
    
    return term1 + term2 + term3

def P_j(y,j):
    j = j+1
    Tjp1 = np_cheb.Chebyshev.basis(j+1)
    Tjl1 = np_cheb.Chebyshev.basis(j-1)

    return (Tjl1(y) - Tjp1(y))/ (2*j)

#Cas k > 0.  
#Definicions seguint Appendix C2. 
def f_j(y, j):
    Tj = np_cheb.Chebyshev.basis(j) 
    return (1 - y**2)**2 * Tj(y) 

def df_j(y,j):
    Tj = np_cheb.Chebyshev.basis(j) 
    dTj = Tj.deriv()
    return -4 * y * (1 - y**2) * Tj(y) + (1 - y**2)**2 * dTj(y)

def ddf_j(y, j):
    Tj = np_cheb.Chebyshev.basis(j)
    dTj = Tj.deriv()  
    ddTj = dTj.deriv()  
    
    term1 = -4 * (1 - 3 * y**2) * Tj(y)
    term2 = -8 * y * (1 - y**2) * dTj(y)
    term3 = (1 - y**2)**2 * ddTj(y)
    
    return term1 + term2 + term3

def dddf_j(y, j):
    Tj = np_cheb.Chebyshev.basis(j)
    dTj = Tj.deriv()  
    ddTj = dTj.deriv() 
    dddTj = ddTj.deriv()  
    
    term1 = 24 * y * Tj(y)
    term2 = -12 * (1 - 3*y**2) * dTj(y)
    term3 = -12 * y * (1 - y**2) * ddTj(y)
    term4 = (1 - y**2)**2 * dddTj(y)
    
    return term1 + term2 + term3 + term4


"""
g_j i derivades (Chebyshev):
- g_j: funció base.
- dg_j, ddg_j: 1a i 2a derivades sense tenir en compte el pes.
- dg_j_b, ddg_j_b: 1a i 2a derivades amb el pes de Chebyshev.
"""
def g_j(y,j): 
    j = j + 2
    Tj0 = np_cheb.Chebyshev.basis(j)
    Tjp2 = np_cheb.Chebyshev.basis(j+2) 
    Tjl2 = np_cheb.Chebyshev.basis(j-2)
    
    return ( Tjp2(y) / (j * (j + 1)) - 2 * Tj0(y) / ((j+1)*(j-1)) + Tjl2(y) / (j * (j-1)) ) / 4

def dg_j(y,j): 
    j = j + 2
    Tj0 = np_cheb.Chebyshev.basis(j)
    Tjp2 = np_cheb.Chebyshev.basis(j+2)
    Tjl2 = np_cheb.Chebyshev.basis(j-2)
    
    return ( Tjp2.deriv()(y) / (j * (j + 1)) - 2 * Tj0.deriv()(y) / ((j+1)*(j-1)) + Tjl2.deriv()(y) / (j * (j-1)) ) / 4

def ddg_j(y,j): 
    j = j+2
    Tj0 = np_cheb.Chebyshev.basis(j).deriv()
    Tjp2 = np_cheb.Chebyshev.basis(j+2).deriv()
    Tjl2 = np_cheb.Chebyshev.basis(j-2).deriv()

    return ( Tjp2.deriv()(y) / (j * (j + 1)) - 2 * Tj0.deriv()(y) / ((j+1)*(j-1)) + Tjl2.deriv()(y) / (j * (j-1)) ) / 4

def dg_j_b(y,j): 
    return dg_j(y,j) + y * g_j(y,j) / (1 - y**2)

def ddg_j_b(y,j): 
    return ddg_j(y,j) + y * dg_j(y,j) / (1 - y**2) + (g_j(y,j) + y * dg_j(y,j)) / (1 - y**2) + 3*y**2 * g_j(y,j) / (1 - y**2)**2

# --------------------------
# Calculs matrius A i B 
# --------------------------
def a_ij(i,j, kx):
    # Calcula a_ij = <u_j, xi_i> amb el producte escalar segons la notació de Moser.
    N = 4 * ((j + 4) + (i + 2))  # Nombre de nodes usats per augmentar la resolucio. 
    if kx == 0: 
        points, weights = np_cheb.chebgauss(N)
        f_points  = h_j(points, j) * P_j(points,i)

    else: 
        points, weights = np_cheb.chebgauss(N)
        f_points = -(ddf_j(points, j) - kx**2 * f_j(points, j) + df_j(points, j) * points / (1 - points**2) ) * g_j(points,i)
        #f_points = f_j(points,j) * f_j(points,i) # Represnetacio ortogonal per f_j (usat per la Figura 9)
    
    if np.abs(np.sum(f_points * weights)) < 10**(-7): 
        return 0 
    
    else: 
        return np.sum(f_points * weights)


def b_ij(i,j, kx):
    # Calcula B_ij = <d^2u_j/dy^2 - k^2 u_j, xi_i> amb el producte escalar.
    N = 5 * ((j + 4) + (j + 2))
    points, weights = np_cheb.chebgauss(N)
    if kx == 0: 
        f_points = (ddh_j(points, j) - kx**2 * h_j(points, j)) * P_j(points, i)
    else: 
        f_points = dddf_j(points, j) * dg_j(points, i) - kx**2 * df_j(points, j) * dg_j(points, i) + kx**2 * ddf_j(points, j) * g_j(points, i) - kx**4 * f_j(points, j) * g_j(points, i)
    
    if np.abs(np.sum(f_points * weights)) < 10**(-7): 
        return 0 
    
    else: 
        return np.sum(f_points * weights)


# Matriu de diferenciacio de Txebitxev (tal i com s'ha definit a l'exemple 3.4)
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