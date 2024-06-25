from numpy import linalg, linspace , random
from numpy import ones, outer, array,diag, dot, vectorize, eye, concatenate,zeros
from numpy import exp,cos, sqrt,log
from numpy import mean, std

from decimal import Decimal,getcontext

from matplotlib.pyplot import plot, scatter, axes, xlabel,ylabel,clf, legend,subplot, savefig,title, fill_between
from sys import stdout

from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#The calculations are based on the Decimal class with 100 digits, so that a large number of polynomials can be orthonormalised.
getcontext().prec = 100
#We set an error that cancels out the coefficients that are too low
erreur_calcul = 10**(-30)


""" --------------------------------------------- """

# Class of polynomials with D variables
class Polynomial:
    def __init__(self,dimension, coefficients):
        self.dimension = dimension
        self.coefficients = coefficients  # coefficients est un dictionnaire {(degré_X1, degré_X2,..., degré_XD): coefficient}

    def update(self):
        L_a_supprimer = []
        for degrees, coeff in self.coefficients.items():
            if abs(coeff) < erreur_calcul:
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
    


def product_scalar(C,P,Q):
    return (P*Q).integrer_polynome(C)
def norm(C,P):
    return Decimal(sqrt((P*P).integrer_polynome(C)))
def orthonormalize_Basis(Basis,C):
    orthonormal_basis = []
    for polynome_cano in Basis:
        n_polynome_ortho = polynome_cano.copy()
        for polynome_ortho in orthonormal_basis:
            n_polynome_ortho = n_polynome_ortho - product_scalar(C,polynome_ortho,n_polynome_ortho)*polynome_ortho
        n_polynome_ortho = (1/norm(C,n_polynome_ortho))*n_polynome_ortho
        
        orthonormal_basis.append(n_polynome_ortho)
    return orthonormal_basis
        
def Decimal_to_float_Base(Base):
    nBase = []
    dimension = Base[0].dimension
    for P in Base:
        nP = Polynomial(dimension,{})
        for degrees, coeff in P.coefficients.items():
            nP.coefficients[degrees] = float(coeff)
        nBase.append(nP)
    return nBase

""" --------------------------------------------- """



def generate_canonical_basis(D,Deg):
    List_degrees = list(product([i for i in range(Deg+1)],repeat=D))
    Basis = []
    for l_degrees in List_degrees:
        if sum(l_degrees) <= Deg:
            Basis.append(Polynomial(D, {tuple(l_degrees) : Decimal(1)}))
    return Decimal_to_float_Base(Basis)

def generate_orthonormal_basis(D,Deg,C):
    List_degrees = list(product([i for i in range(Deg+1)],repeat=D))
    Basis = []
    for l_degrees in List_degrees:
        if sum(l_degrees) <= Deg:
            Basis.append(Polynomial(D, {tuple(l_degrees) : Decimal(1)}))
    return Decimal_to_float_Base(orthonormalize_Basis(Basis, C))


""" --------------------------------------------- """

def Phi(Base,P,t):
    return array([Base[j](t) for j in range(P)])

def dataset_initialisation(y,C,M,ratio_data,Random_seed):
    random.seed(Random_seed)
    D = len(C)
    U = random.uniform(0,1,size = (M,D))
    X = ones((M,D))@diag([C[d][0] for d in range(D)])+U@diag([C[d][1]-C[d][0] for d in range(D)])
    Y = list(map(y,X))
    
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=ratio_data, random_state=42)
    return X_train, X_test, Y_train, Y_test

def ridge_regression(M_min, M_max, Features, Lambda, Data_global, Data_func):
    
    X_global, Y_global = Data_global
    Train_error, Test_error,Global_error,Beta_norm  = [], [], [], []
    y,C,ratio_data,Random_seed = Data_func
    
    for M in range(M_min, M_max):
        #Dataset creation, with seed set for reproducibility        
        X_train, X_test, Y_train, Y_test = dataset_initialisation(y,C,M,ratio_data,Random_seed)
        
        Z_P = array([Phi(Features,P,x_train) for x_train in X_train])
        #Z_P_test = array([Phi(Features,P,x_test) for x_test in X_test])
        #Z_P_global = array([Phi(Features,P,x_global) for x_global in X_global])
        
        if Lambda == 0:
            beta_chap = linalg.pinv(Z_P)@Y_train
        else:
            W_P = concatenate((Z_P,sqrt(Lambda*N)*eye(P)), axis = 0)
            beta_chap = linalg.pinv(W_P)@concatenate((Y_train,zeros(P,)), axis = 0)
            
        y_chap =  lambda t: dot(Phi(Features,P,t), beta_chap)

        Train_error.append(log(1+mean_squared_error(Y_train, list(map(y_chap,X_train)))))
        Test_error.append(log(1+mean_squared_error(Y_test, list(map(y_chap,X_test)))))
        Global_error.append(log(1+mean_squared_error(Y_global, list(map(y_chap,X_global)))))
        Beta_norm.append(log(1+sum(beta_chap**2)))
        stdout.write("Train_error: "+ str(Train_error[-1])+ "| Test_error: " + str(Test_error[-1])+"| Global_error: " + str(Global_error[-1])+"\n")
        print(P/M)
    if len(C) == 1:
        # 1D graph visualization
        clf()
        plot(X_global,list(map(y_chap,X_global)), label="Estimator")
        plot(X_global,Y_global, label = "Target")
        scatter(X_train, Y_train, label = "Interpolation points")
        scatter(X_test, Y_test, label = "Test points")
        xlabel("x")
        ylabel("y")
        title("Functions plot")
        legend()
        savefig('Function_plot'+str(Random_seed)+'.png')
        clf()
    return Train_error, Test_error, Beta_norm, Global_error, y_chap
    
def main(M_min,M_max, Lambda,C, y ,Features, ratio_data,Random_seed):
    
    # Uniform mesh for global error calculation and visualization, only works for cubes in dimension 2
    P,D = len(Features), len(C)
    X_global = array(list(product(linspace(C[0][0],C[0][1],n),repeat = D)))
    Y_global = list(map(y,X_global))
    Data_global = X_global, Y_global
    
        
    #Ridge regression
    Data_func = y,C,ratio_data,Random_seed
    Train_error, Test_error, Beta_norm, Global_error, y_chap = ridge_regression(M_min, M_max, Features, Lambda, Data_global, Data_func)
    
    
    clf()
    
    liste_M = [ P/int((M*(1-ratio_data))) for M in range(M_min,M_max) ]
    plot(liste_M, Beta_norm, label = "Beta_norm")
    plot(liste_M, Train_error, label="Train_error")
    plot(liste_M, Test_error, label="Test_error")
    plot(liste_M,Global_error, label="Global_error")
    xlabel("P/N ratio with P:nb parameters, N: nb points")
    ylabel("log(1+MSE)")
    title("Errors plot with P fixed")
    legend()
    
    savefig('Graph_error_on_N.png')
    return Beta_norm, Train_error, Test_error, Global_error
    
a, n_example = 1, 1
type_polynome = 2
D, Deg= 2,5

n = 20
ratio_data = 0.2
Lambda = 0
M_min, M_max = 15, 250

Liste_seeds = [1000,12334,13902,29389,1233983,18938309,22387928,37289749,232382,42,3297]

if n_example == 1:
    y = lambda X: 2*exp(D - sum([X[i]**2 for i in range(D)]))
if n_example == 2:
    y = lambda X: sum([3*cos(40*X[i]) + cos(10*X[i])  - X[i]**2 for i in range(D)])
     

if type_polynome == 2:
    stdout.write("Orthonormalized basis\n")
    Features = generate_orthonormal_basis(D,Deg,array([[Decimal(-a),Decimal(a)] for _ in range(D)]))
if type_polynome == 1:
    stdout.write("Canonical basis\n")
    Features = generate_canonical_basis(D,Deg)

P =  len(Features)
C = array([[-a,a] for _ in range(D)])
liste_M = [P/int((M*(1-ratio_data))) for M in range(M_min,M_max)]


Liste_Beta_norm, Liste_Train_error, Liste_Test_error, Liste_Global_error = [],[],[],[]
for Random_seed in Liste_seeds:
    Beta_norm, Train_error, Test_error, Global_error = main(M_min,M_max, Lambda,C, y ,Features, ratio_data,Random_seed)
    Liste_Beta_norm.append(Beta_norm)
    Liste_Train_error.append(Train_error)
    Liste_Test_error.append(Test_error)
    Liste_Global_error.append(Global_error)

Beta_norm_min, Beta_norm_max = mean(Liste_Beta_norm,axis=0)-(1/2)*std(Liste_Beta_norm, axis=0), mean(Liste_Beta_norm,axis=0)+(1/2)*std(Liste_Beta_norm, axis=0)
Train_error_min, Train_error_max = mean(Liste_Train_error,axis=0)-(1/2)*std(Liste_Train_error, axis=0), mean(Liste_Train_error,axis=0)+(1/2)*std(Liste_Train_error, axis=0)
Test_error_min, Test_error_max = mean(Liste_Test_error,axis=0)-(1/2)*std(Liste_Test_error, axis=0), mean(Liste_Test_error,axis=0)+(1/2)*std(Liste_Test_error, axis=0)
Global_error_min, Global_error_max = mean(Liste_Global_error,axis=0)-(1/2)*std(Liste_Global_error, axis=0), mean(Liste_Global_error,axis=0)+(1/2)*std(Liste_Global_error, axis=0)

clf()
fill_between(liste_M, Beta_norm_min, Beta_norm_max, color='blue', alpha=0.3, label='Beta_norm_std')
fill_between(liste_M, Train_error_min, Train_error_max, color='orange', alpha=0.3, label='Train_error_std')
fill_between(liste_M, Test_error_min, Test_error_max, color='green', alpha=0.3, label='Test_error_std')
fill_between(liste_M, Global_error_min, Global_error_max, color='red', alpha=0.3, label='Global_error_std')
plot(liste_M, mean(Liste_Beta_norm,axis=0), color='blue', label = "Beta_norm_mean")
plot(liste_M, mean(Liste_Train_error,axis=0), color='orange', label="Train_error_mean")
plot(liste_M, mean(Liste_Test_error,axis=0), color='green', label="Test_error_mean")
plot(liste_M,mean(Liste_Global_error,axis=0), color='red', label="Global_error_mean")
xlabel("P/N ratio with P:nb parameters, N: nb points")
ylabel("log(1+MSE)")
title("Mean errors plot, P fixed")
legend()
savefig('Graph_error_results_on_N.png')

