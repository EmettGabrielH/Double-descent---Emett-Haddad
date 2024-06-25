from numpy import random, trace, linalg, dot, linspace, meshgrid, vectorize, log, abs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def excess_risk(delta,gamma):
    res = 1 +(min(gamma,1)/ delta)*(-2 + abs((1-delta)/(1-gamma)))
    return res

def excess_risk2(delta,gamma):
    res = ((abs(1 - gamma))**(-1)) * ( max(1, gamma) - gamma/delta)
    return res

def excess_risk3(delta,gamma):
    if gamma < 1:
        res = ((abs(1 - gamma))**(-1)) * (1 - gamma/delta)
    if gamma > 1:
        res = 1 - 2/delta + (1/delta) * ((delta - 1)/(gamma - 1))
    return res

def diff(delta,gamma):
    return abs(excess_risk(delta,gamma) - excess_risk3(delta,gamma))
# Créer une grille de points
DELTA, GAMMA = meshgrid(linspace(1, 3, 30), linspace(0.1, 2, 30))



def afficher():
    Z = vectorize(excess_risk)(DELTA, GAMMA)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(DELTA, GAMMA, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_title(r"$\frac{\overline{\mathcal{E}}(\gamma,\delta)}{\sigma_{\Phi}^{2} ||\beta^{\star}||^2} =  [1 - 2\frac{min(\gamma,1)}{\delta} + \frac{min(\gamma,1)}{\delta}|\frac{1 - \delta}{1 - \gamma}| ]$")

    ax.set_xlabel(r'$\delta = lim \; \frac{E}{N}$')
    ax.set_ylabel(r'$\gamma = lim \; \frac{P}{N}$')
    ax.set_zlabel(r'$\frac{\overline{\mathcal{E}}(\gamma,\delta)}{\sigma_{\Phi}^{2} ||\beta^{\star}||^2}$')
    ax.view_init(elev=30, azim=-30)

    plt.savefig('E_Haddad_modele_lineaire.png')
    plt.show()
    
def afficher2():
    Z = vectorize(excess_risk2)(DELTA, GAMMA)
    titre_func = r"\overline{\mathcal{E}}(\gamma,\delta)"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(DELTA, GAMMA, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_title(r"$\frac{"+titre_func+r"} {\sigma_{\Phi}^{2} ||\beta^{\star}||^2} =  \frac{1}{|1 - \gamma|} [max(1,\gamma)  - \frac{\gamma}{\delta}  ]$")
    
    ax.set_xlabel(r'$\delta = lim \; \frac{E}{N}$')
    ax.set_ylabel(r'$\gamma = lim \; \frac{P}{N}$')
    ax.set_zlabel(r"$\frac{"+titre_func+r"}{||\beta^{\star}||^2}$")
    ax.view_init(elev=30, azim=-30)
    
    plt.savefig('F_Bach_modele_lineaire.png')
    
    plt.show()
    
def afficher3():
    Z = vectorize(excess_risk3)(DELTA, GAMMA)
    titre_func = r"\mathbb{E}_{X}(\mathcal{R}(\hat{y}_{\hat{\beta}}))"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(DELTA, GAMMA, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_title(r"$\frac{"+titre_func+r"} {||\beta^{\star}||^2}$")
    
    ax.set_xlabel(r'$\delta = lim \; \frac{E}{N}$')
    ax.set_ylabel(r'$\gamma = lim \; \frac{P}{N}$')
    ax.set_zlabel(r"$\frac{"+titre_func+r"}{||\beta^{\star}||^2}$")
    ax.view_init(elev=30, azim=-30)
    
    plt.savefig('Belkin_modele_lineaire.png')
    
    plt.show()

def afficher4():
    Z = vectorize(diff)(DELTA, GAMMA)
    titre_func = r"difference\_modeles"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(DELTA, GAMMA, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_title(r"$\frac{"+titre_func+r"} {||\beta^{\star}||^2}$")
    
    ax.set_xlabel(r'$\delta = lim \; \frac{E}{N}$')
    ax.set_ylabel(r'$\gamma = lim \; \frac{P}{N}$')
    ax.set_zlabel(r"$\frac{"+titre_func+r"}{||\beta^{\star}||^2}$")
    ax.view_init(elev=30, azim=-30)
    
    plt.savefig('Diff_Belk_EH_modele_lineaire.png')
    
    plt.show()
afficher4()
