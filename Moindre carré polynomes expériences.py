from numpy import linalg, linspace , random
from numpy import ones, outer, array,diag, dot, vectorize
from numpy import exp,cos, sqrt,log

from decimal import Decimal,getcontext

from matplotlib.pyplot import pause, show, plot, scatter, axes, xlabel,ylabel,clf, legend,subplot
from sys import stdout

from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#On fixe les calculs sur la classe Decimal avec 100 chiffres
getcontext().prec = 100

# Classe de polynomes a D variables
class Polynomial:
    def __init__(self,dimension, coefficients):
        self.dimension = dimension
        self.coefficients = coefficients  # coefficients est un dictionnaire {(degré_X1, degré_X2,..., degré_Xd): coefficient}

    def update(self):
        L_a_supprimer = []
        for degrees, coeff in self.coefficients.items():
            if abs(coeff) < 10**(-30):
                L_a_supprimer.append(degrees)
        for d in L_a_supprimer:
            del self.coefficients[d]
        
        if len(self.coefficients) == 0:
            self.coefficients[tuple([0]*self.dimension)] = Decimal(0)
    def copy(self):
        return Polynomial(self.dimension, self.coefficients.copy())
    def __repr__(self):
        res = ""
        for degrees, coeff in self.coefficients.items():
            res += f"{coeff}"
            for i in range(self.dimension):
                if degrees[i] != 0 : res += f"*X{i+1}^{degrees[i]}"
            res += " + "
        return res[:-3]
    def __str__(self):
        res = ""
        for degrees, coeff in self.coefficients.items():
            res += f"{coeff}"
            for i in range(self.dimension):
                if degrees[i] != 0 : res += f"*X{i+1}^{degrees[i]}"
            res += " + "
        return res[:-3]
    
    def __call__(self, x):    
        res = 0
        for degrees, coeff in self.coefficients.items():
            prod = coeff
            for i in range(self.dimension):
                prod *= x[i]**degrees[i]
            res += prod
        return res 

    def add_term(self, degrees, coeff):
        if degrees in self.coefficients:
            self.coefficients[degrees] += coeff
        else:
            self.coefficients[degrees] = coeff
        
    def __add__(self, other):
        result = self.copy()
        for degrees, coeff in other.coefficients.items():
            result.add_term(degrees, coeff)
        result.update()
        return result

    def __mul__(self, other):
        result = Polynomial(self.dimension,{})
        for degrees1, coeff1 in self.coefficients.items():
            for degrees2, coeff2 in other.coefficients.items():
                new_degrees = tuple([degrees1[i] + degrees2[i] for i in range(self.dimension)])
                new_coeff = coeff1 * coeff2
                result.add_term(new_degrees,new_coeff)
        result.update()
        return result
    def __rmul__(self, x):
        return self*Polynomial(self.dimension, {tuple([0]*self.dimension): x})

    def __sub__(self,other):
        result = self.copy()
        for degrees, coeff in other.coefficients.items():
            result.add_term(degrees,-coeff)
        result.update()
        return result
    def primitiver(self,i):
        result = Polynomial(self.dimension,{})
        for degrees, coeff in self.coefficients.items():
            new_degrees = list(degrees)
            new_degrees[i-1] += 1
            result.add_term(tuple(new_degrees),coeff/new_degrees[i-1])
        result.update()
        return result
    def deriver(self,i):
        result = Polynomial(self.dimension,{})
        for degrees, coeff in self.coefficients.items():
            if degrees[i-1] != 0:
                new_degrees = list(degrees)
                new_degrees[i-1] -= 1
                result.add_term(tuple(new_degrees),coeff*degrees[i-1])
                
        result.update()                
        return result
    def eval(self,x,i):
        result = Polynomial(self.dimension-1,{})
        for degrees, coeff in self.coefficients.items():
            if degrees[i-1] != 0:
                new_degrees = list(degrees)
                degree = new_degrees.pop(i-1)
                result.add_term(tuple(new_degrees),coeff*(x**degree))
            else:
                new_degrees = list(degrees)
                new_degrees.pop(i-1)
                result.add_term(tuple(new_degrees),coeff)
    
        result.update()
        return result

    def integrer_polynome(self,C):
        result = self.copy()
        for i in range(self.dimension):
            result =   result.primitiver(1)
            result = result.eval(C[i][1],1) - result.eval(C[i][0],1)
        return result.coefficients[()]
    
def generer_base(dimension,degre):
    Liste_degrees = list(product([i for i in range(degre+1)],repeat=dimension))
    Base = []
    for l_degrees in Liste_degrees:
        if sum(l_degrees) <= degre:
            Base.append(Polynomial(dimension, {tuple(l_degrees) : Decimal(1)}))
    return Base


def produit_scalaire(C,P,Q):
    return (P*Q).integrer_polynome(C)
def norme(C,P):
    return Decimal(sqrt((P*P).integrer_polynome(C)))
def orthonormaliser_Base(B,C):
    nB = []
    for e in B:
        ne = e.copy()
        for nb in nB:
            ne = ne - produit_scalaire(C,nb,ne)*nb
        ne = (1/norme(C,ne))*ne
        
        nB.append(ne)
    return nB
        
def Decimal_to_float_Base(Base):
    nBase = []
    dimension = Base[0].dimension
    for P in Base:
        nP = Polynomial(dimension,{})
        for degrees, coeff in P.coefficients.items():
            nP.coefficients[degrees] = float(coeff)
        nBase.append(nP)
    return nBase


f = lambda X: exp(2 - (X[0]**2 + X[1]**2 ))
M = 70
n = 40
degree = 12
ratio = 0.2
C = [[-1,1],[-1,1]]


D = len(C)
stdout.write("Base \n")
Base_orthonormale = Decimal_to_float_Base(orthonormaliser_Base(generer_base(D,degree),C))
stdout.write("Base orthonormale de cardinal :" + str(len(Base_orthonormale))+"\n")

#Création du dataset, avec seed fixée pour reproductabilité
random.seed(23334)
U = random.uniform(0,1,size = (M,D))
X = ones((M,D))@diag([C[d][0] for d in range(D)])+U@diag([C[d][1]-C[d][0] for d in range(D)])

Y = list(map(f,X))
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=ratio, random_state=42)
N = len(X_train)
print("M: " + str(M) + "| N:" + str(N))

# Maillage pour le calcul de l'erreur globale et pour la visualisation
x = array(list(product(linspace(C[0][0],C[0][1],n),repeat = D)))
if D == 2:
    u1 = linspace(C[0][0],C[0][1],n)
    u = outer(linspace(C[0][0], C[0][1], n), ones(n))

P_min, P_max = 1, len(Base_orthonormale)
erreur_train, erreur_test,erreur_global,norme_beta  = [], [], [], []

liste_P = []
Y_global = list(map(f,x))

def Phi(Base_orthonormale,P,t):
    return array([Base_orthonormale[j](t) for j in range(P)])

for P in range(P_min, P_max):
    liste_P.append(P/N)
    stdout.write("P: "+ str(P) + " P/N: "+ str(P/N) + "\n")

    Z = array([Phi(Base_orthonormale,P,X_train[i]) for i in range(N)])
    beta_chap = linalg.pinv(Z)@Y_train
    Y_chap = Z@beta_chap
    y_chap =  lambda t: dot(Phi(Base_orthonormale,P,t), beta_chap)

    erreur_train.append(log(1+mean_squared_error(Y_train,Y_chap)))
    norme_beta.append(log(1+sum(beta_chap**2)))
    erreur_test.append(log(1+mean_squared_error(Y_test, list(map(y_chap,X_test)))))
    erreur_global.append(log(1+mean_squared_error(Y_global, list(map(y_chap,x)))))

    stdout.write("Erreur train: "+ str(erreur_train[-1])+ "\n")
    stdout.write("Norme de beta: " + str(norme_beta[-1])+ "\n")
    stdout.write("Erreur test: " + str(erreur_test[-1])+ "\n")
    stdout.write("Erreur global: " + str(erreur_global[-1])+ "\n")
    
    if  P > N:
        stdout.write("Minimum sous-paramétré: "+ str(min(erreur_test[:N-P_min])) + "| Minimum sur-paramétré: " + str(min(erreur_test[N-P_min:]))+"\n")
    else:
        stdout.write("Minimum sous-paramétré: "+ str(min(erreur_test)) + "\n")

        
    subplot(2,1,1)
    if D == 2:
        # Visualisation des graphes en 2D
        Y_visualisation = array([[f((u1[i],u1[j])) for i in range(n)] for j in range(n)])
        Y_visualisation_approx = array([[y_chap((u1[i],u1[j])) for i in range(n)] for j in range(n)])
        
        ax = axes(projection='3d')
        ax.set_title('Graphe 3d des fonctions: approximation en rouge')
        ax.plot_surface(u,u.T,Y_visualisation , cmap='viridis',edgecolor='green')
        ax.plot_surface(u,u.T,Y_visualisation_approx , cmap='viridis',edgecolor='red')
        

    if D == 1:
        # Visualisation des graphes en 1D
        plot(x,list(map(y_chap,x)))
        plot(x,Y_global)
        scatter(X, Y)
        legend(["Approximation", "Fonction a approximer"])

    # Affichage des courbes d'erreurs
    subplot(2,1,2)
    plot(liste_P, norme_beta)
    plot(liste_P, erreur_train)
    plot(liste_P, erreur_test)
    plot(liste_P,erreur_global)
    xlabel("Rapport P/N ou P:nb parametres, N: nb de points")
    ylabel("log(1+MSE)")
    legend(["Norme beta", "Erreur train", "Erreur test","Erreur global"])


    pause(0.1)
    clf()

