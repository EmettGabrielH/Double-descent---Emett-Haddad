from numpy import random, trace, linalg, dot, linspace, meshgrid, vectorize, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def Z(P,N):
    return random.normal(loc=0, scale=1, size=(N, P))
def f(x) : return log(1+x)

def simulation_exp(delta,gamma):
    N = 500
    M = 1
    E, P = int(delta*N), int(gamma*N)
    res = 0
    for _ in range(M):
        Z_E = Z(E,N)
        Z_P = Z_E[:, :P]
        res += trace(dot(dot(Z_E,Z_E.T), linalg.pinv(dot(Z_P,Z_P.T))))

    res /= M * min(N,P)
    print(res)
    return res

def simulation_th(delta,gamma):
    if gamma != 1:
        res = abs((1-delta)/(1-gamma))
    else:
        res = 10
    return res

# Créer une grille de points
DELTA, GAMMA = meshgrid(linspace(1, 2, 10), linspace(0.1, 2, 10))

Z_th = vectorize(simulation_th)(DELTA, GAMMA)
Z_exp = vectorize(simulation_exp)(DELTA, GAMMA)

Z_diff = abs(Z_exp - Z_th)

def afficher(ntype):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if ntype == 0:
        surface = ax.plot_surface(DELTA, GAMMA, Z_exp, cmap=cm.coolwarm, linewidth=0)
        #ax.set_title(r"$log(1 + \frac{1}{min(N,P)}\mathbb{E}(tr(Z_EZ_E^T(Z_PZ_P^T)^{\dagger})))$")
        ax.set_title(r"$\frac{1}{min(N,P)}tr(Z_EZ_E^T(Z_PZ_P^T)^{\dagger})$")
    elif ntype == 1:
        surface = ax.plot_surface(DELTA, GAMMA, Z_th, cmap=cm.coolwarm, linewidth=0)
        #ax.set_title(r"$|log(1 + \frac{\delta - 1}{\gamma - 1}|)$")
        ax.set_title(r"$|\frac{\delta - 1}{\gamma - 1}|$")
    else:
        surface = ax.plot_surface(DELTA, GAMMA, Z_diff, cmap=cm.coolwarm, linewidth=0)
        #ax.set_title(r"$ |\frac{\delta - 1}{\gamma - 1}|- \frac{1}{min(N,P)}tr(Z_EZ_E^T(Z_PZ_P^T)^{\dagger})$")
        
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel('tr')

    plt.show()


afficher(0)
