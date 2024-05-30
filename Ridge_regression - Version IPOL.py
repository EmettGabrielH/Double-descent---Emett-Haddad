from numpy import linalg, linspace , random
from numpy import ones, outer, array,diag, dot, vectorize, eye, concatenate,zeros
from numpy import exp,cos, sqrt,log

from decimal import Decimal,getcontext

from matplotlib.pyplot import pause, show, plot, scatter, axes, xlabel,ylabel,clf, legend,subplot, savefig
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

def ridge_regression(P_min, P_max,Features,Lambda,Data,Data_global):
    N = len(Data[0])
    X_train, X_test, Y_train, Y_test = Data
    X_global, Y_global = Data_global
    Train_error, Test_error,Global_error,Beta_norm  = [], [], [], []
    
    for P in range(P_min, P_max):
        stdout.write("P: "+ str(P) + " P/N: "+ str(P/N) + "\n")

        Z_P = array([Phi(Features,P,x_train) for x_train in X_train])
        Z_P_test = array([Phi(Features,P,x_test) for x_test in X_test])
        Z_P_global = array([Phi(Features,P,x_global) for x_global in X_global])
        
        if Lambda == 0:
            beta_chap_P = linalg.pinv(Z_P)@Y_train
        else:
            W_P = concatenate((Z_P,sqrt(Lambda*N)*eye(P)), axis = 0)
            beta_chap_P = linalg.pinv(W_P)@concatenate((Y_train,zeros(P,)), axis = 0)
            
        
        Train_error.append(log(1+mean_squared_error(Y_train,Z_P@beta_chap_P)))
        Test_error.append(log(1+mean_squared_error(Y_test, Z_P_test@beta_chap_P)))
        Global_error.append(log(1+mean_squared_error(Y_global, Z_P_global@beta_chap_P)))
        Beta_norm.append(log(1+sum(beta_chap_P**2)))

        stdout.write("Train_error: "+ str(Train_error[-1])+ "| Test_error: " + str(Test_error[-1])+"| Global_error: " + str(Global_error[-1])+"\n")
        stdout.write("Beta_norm: " + str(Beta_norm[-1])+ "\n")
        
        if  P > N:
            stdout.write("Under-parameterised minimum test: "+ str(min(Test_error[:N-P_min])) + "| Over-parameterised minimum test: " + str(min(Test_error[N-P_min:]))+"\n")
            
    y_chap =  lambda t: dot(Phi(Features,P,t), beta_chap_P)
    return Train_error, Test_error, Beta_norm, Global_error, y_chap
    
def main(n, M, D, Lambda,C, y ,Features, ratio_data,Random_seed):
    
    #Dataset creation, with seed set for reproducibility
    
    Data = dataset_initialisation(y,C,M,ratio_data,Random_seed)

    N = len(Data[0])
    stdout.write("M: " + str(M) + "| N:" + str(N)+"\n")
    
    # Uniform mesh for global error calculation and visualization, only works for cubes in dimension 2
    X_global = array(list(product(linspace(C[0][0],C[0][1],n),repeat = D)))
    Y_global = list(map(y,X_global))
    Data_global = X_global, Y_global
    
    if D == 2:
        u1 = linspace(C[0][0],C[0][1],n)
        u = outer(linspace(C[0][0], C[0][1], n), ones(n))
    P_min, P_max = 1, len(Features)

    #Ridge regression
    Train_error, Test_error, Beta_norm, Global_error, y_chap = ridge_regression(P_min, P_max,Features,Lambda,Data,Data_global)
    
    # Visualization
    
    if D == 2:
        # 2D graph visualization
        Y_visualization = array([[y((u1[i],u1[j])) for i in range(n)] for j in range(n)])
        Y_visualization_approx = array([[y_chap((u1[i],u1[j])) for i in range(n)] for j in range(n)])
        
        ax = axes(projection='3d')
        ax.set_title('Estimator: red and target: green,  lambda ='+ str(Lambda))
        ax.plot_surface(u,u.T,Y_visualization , cmap='viridis',edgecolor='green')
        ax.plot_surface(u,u.T,Y_visualization_approx , cmap='viridis',edgecolor='red')
        

    if D == 1:
        # 1D graph visualization
        plot(X_global,list(map(y_chap,X_global)))
        plot(X_global,Y_global)
        scatter(Data[0], Data[2])
        legend(["Estimator", "Target"])
        
    savefig('Function_plot.png')
    # Display of error curves
    
    clf()

    liste_P = [P/N for P in range(P_min,P_max)]
    plot(liste_P, Beta_norm)
    plot(liste_P, Train_error)
    plot(liste_P, Test_error)
    plot(liste_P,Global_error)
    xlabel("P/N ratio with P:nb parameters, N: nb points")
    ylabel("log(1+MSE)")
    legend(["Beta_norm", "Train_error", "Test_error","Global_error"])
    
    savefig('Graph_error.png')
    
#"""
a, n_example, type_polynome, D, Deg,r = 1, 1, 2, 2,9,0.2 
n = 50
M = 50
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
    
C = array([[-a,a] for _ in range(D)])
main(n,M, D, 0,C, y, Features, r,1000)
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--Lambda", type=float, required=True)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--Deg", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--Random_seed", type=int, required=True)
    parser.add_argument("--n_example", type=int, required=True)
    parser.add_argument("--type_polynome", type=int, required=True)
    parser.add_argument("--r", type=float, required=True)
    args = parser.parse_args()

    C = array([[-args.a,args.a] for _ in range(args.D)])
    
    
    if args.n_example == 1:
        y = lambda X: 2*exp(args.D - sum([X[i]**2 for i in range(args.D)]))
    if args.n_example == 2:
        y = lambda X: sum([3*cos(40*X[i]) + cos(10*X[i])  - X[i]**2 for i in range(args.D)])
         
    
    if args.type_polynome == 2:
        stdout.write("Orthonormalized basis\n")
        Features = generate_orthonormal_basis(args.D,Deg,array([[Decimal(-args.a),Decimal(args.a)] for _ in range(args.D)]))
    if args.type_polynome == 1:
        stdout.write("Canonical basis\n")
        Features = generate_canonical_basis(args.D,args.Deg)
        
    
    main(args.n,args.M, args.D, args.Lambda,C, y, Features, args.r, args.Random_seed)
"""
